# Face Worker Implementation Task

## Mission

Create a robust face detection worker for Windows that:
1. **Never loses faces** - failures are tracked and retried, not marked as "processed"
2. **Handles GPU OOM** - progressive downscaling and HOG fallback
3. **Conservative settings** - prioritizes accuracy over speed for GTX 1650 4GB

## Context

You are working in `C:\VisionFaceService` on a Windows machine (GAME @ 192.168.12.75).
This worker connects to a PostgreSQL database on the Vision server (192.168.12.20) and processes images from an SMB share.

The existing face worker has a critical bug: when face detection fails (OOM, timeout, etc.), it saves `face_count=0` to `face_metadata`, marking the image as "processed". These images are never retried, permanently losing any faces.

## Your Task

Create `face_worker.py` with these components:

### 1. Configuration Constants

```python
# Database
DB_HOST = "192.168.12.20"
DB_PORT = 5432
DB_NAME = "vision"
DB_USER = "vision"
DB_PASS = "vision123"

# Image Access
IMAGE_MOUNT = r"\\192.168.12.20\vision_images"

# Conservative GPU Settings (GTX 1650 4GB)
MAX_IMAGE_DIMENSION = 1280        # Aggressive downscaling
SCALE_CASCADE = [1.0, 0.75, 0.5, 0.33, 0.25]  # Progressive scales
MIN_SCALE = 0.25
MAX_WORKERS = 1                   # Single worker for 4GB GPU
BATCH_SIZE = 5                    # Small batches
DETECTION_TIMEOUT = 120           # 2 minutes per image
MEMORY_CLEANUP_INTERVAL = 3       # gc.collect every N images

# Model Settings
USE_CNN_MODEL = True              # Use CNN (GPU) by default
HOG_FALLBACK_ON_OOM = True        # Fall back to HOG if OOM
CNN_UPSAMPLE_TIMES = 0            # No upsampling (saves VRAM)

# Retry Settings
MAX_RETRY_COUNT = 5
BASE_RETRY_DELAY_SECONDS = 60
RETRY_BACKOFF_MULTIPLIER = 2
MAX_RETRY_DELAY_SECONDS = 3600
```

### 2. Worker Initialization with HEIC Support

```python
def init_face_worker():
    """Initialize worker process with HEIC support."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
```

### 3. Path Conversion Functions

```python
def windows_to_linux_path(windows_path: str) -> str:
    """Convert Windows UNC path to relative path for database."""
    rel_path = windows_path.replace(IMAGE_MOUNT, "").lstrip("\\").replace("\\", "/")
    return rel_path.lstrip("/")

def linux_to_windows_path(linux_path: str) -> str:
    """Convert database relative path to Windows UNC path."""
    return os.path.join(IMAGE_MOUNT, linux_path.replace("/", "\\"))
```

### 4. Image Preprocessing with Downscaling

```python
def preprocess_image(
    image_path: str,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    target_scale: float = 1.0
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Load and preprocess image with downscaling.

    Returns:
        - Preprocessed numpy array
        - Actual scale factor applied
        - Original dimensions (width, height)
    """
    from PIL import Image

    with Image.open(image_path) as img:
        original_size = img.size  # (width, height)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Calculate required scale
        max_orig_dim = max(original_size)
        dimension_scale = min(1.0, max_dimension / max_orig_dim)
        final_scale = min(dimension_scale, target_scale)

        if final_scale < 1.0:
            new_size = (
                int(original_size[0] * final_scale),
                int(original_size[1] * final_scale)
            )
            img = img.resize(new_size, Image.LANCZOS)

        return np.array(img), final_scale, original_size
```

### 5. GPU Memory Cleanup

```python
def cleanup_gpu_memory():
    """Aggressive GPU memory cleanup."""
    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
```

### 6. Coordinate Scaling

```python
def scale_face_locations(
    face_locations: List[Tuple[int, int, int, int]],
    scale_factor: float
) -> List[Tuple[int, int, int, int]]:
    """Scale face locations back to original image coordinates."""
    if scale_factor >= 1.0:
        return face_locations

    inverse_scale = 1.0 / scale_factor
    return [
        (
            int(top * inverse_scale),
            int(right * inverse_scale),
            int(bottom * inverse_scale),
            int(left * inverse_scale)
        )
        for top, right, bottom, left in face_locations
    ]
```

### 7. Progressive Cascade Detection (CRITICAL)

```python
def detect_faces_with_cascade(image_path: str) -> Dict[str, Any]:
    """
    Attempt face detection with progressive downscaling.
    Tries each scale in SCALE_CASCADE, falling back on OOM.
    """
    import face_recognition

    last_error = None
    last_error_type = None
    attempted_scale = None
    original_size = None

    for scale in SCALE_CASCADE:
        attempted_scale = scale
        cleanup_gpu_memory()

        try:
            # Preprocess image
            image_array, actual_scale, original_size = preprocess_image(
                image_path,
                max_dimension=MAX_IMAGE_DIMENSION,
                target_scale=scale
            )

            # Try CNN model first (GPU accelerated)
            try:
                face_locations = face_recognition.face_locations(
                    image_array,
                    model="cnn",
                    number_of_times_to_upsample=CNN_UPSAMPLE_TIMES
                )
            except RuntimeError as e:
                error_str = str(e).lower()
                if "cuda" in error_str or "out of memory" in error_str:
                    if HOG_FALLBACK_ON_OOM:
                        logger.info(f"CNN OOM at scale {scale}, trying HOG fallback...")
                        cleanup_gpu_memory()
                        face_locations = face_recognition.face_locations(
                            image_array,
                            model="hog"
                        )
                    else:
                        raise
                else:
                    raise

            # Success! Scale coordinates back
            scaled_locations = scale_face_locations(face_locations, actual_scale)

            # Get encodings
            face_encodings = face_recognition.face_encodings(
                image_array,
                face_locations  # Use original locations for encoding
            )

            return {
                "success": True,
                "face_count": len(face_locations),
                "face_locations": [list(loc) for loc in scaled_locations],
                "face_encodings": [enc.tolist() for enc in face_encodings],
                "scale_used": actual_scale,
                "original_size": original_size,
            }

        except RuntimeError as e:
            error_str = str(e).lower()
            if "cuda" in error_str or "out of memory" in error_str:
                last_error = str(e)
                last_error_type = "oom"
                logger.warning(f"OOM at scale {scale}, trying smaller...")
                cleanup_gpu_memory()
                continue
            else:
                last_error = str(e)
                last_error_type = "runtime"
                break

        except Exception as e:
            last_error = str(e)
            last_error_type = "unknown"
            logger.error(f"Detection error at scale {scale}: {e}")
            break

    # All scales failed
    return {
        "success": False,
        "face_count": 0,
        "error": last_error,
        "error_type": last_error_type,
        "attempted_scale": attempted_scale,
        "original_size": original_size,
    }
```

### 8. Failure Recording (NOT to face_metadata!)

```python
async def record_detection_failure(
    conn,
    image_path: str,
    error: str,
    error_type: str,
    original_size: Tuple[int, int] = None,
    attempted_scale: float = None
):
    """Record failure for later retry - NOT to face_metadata!"""
    await conn.execute("""
        INSERT INTO face_detection_failures
            (image_path, last_error, last_error_type,
             original_width, original_height, last_attempted_scale,
             retry_count, last_failure_at, next_retry_after)
        VALUES ($1, $2, $3, $4, $5, $6, 1, CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP + INTERVAL '1 second' * $7)
        ON CONFLICT (image_path) DO UPDATE SET
            retry_count = face_detection_failures.retry_count + 1,
            last_error = EXCLUDED.last_error,
            last_error_type = EXCLUDED.last_error_type,
            last_attempted_scale = EXCLUDED.last_attempted_scale,
            last_failure_at = CURRENT_TIMESTAMP,
            next_retry_after = CURRENT_TIMESTAMP + INTERVAL '1 second' *
                LEAST($8 * POWER($9, face_detection_failures.retry_count), $10),
            permanently_failed = (face_detection_failures.retry_count >= $11)
    """,
        image_path,
        error,
        error_type,
        original_size[0] if original_size else None,
        original_size[1] if original_size else None,
        attempted_scale,
        BASE_RETRY_DELAY_SECONDS,
        BASE_RETRY_DELAY_SECONDS,
        RETRY_BACKOFF_MULTIPLIER,
        MAX_RETRY_DELAY_SECONDS,
        MAX_RETRY_COUNT
    )

async def clear_detection_failure(conn, image_path: str):
    """Remove failure record on success."""
    await conn.execute(
        "DELETE FROM face_detection_failures WHERE image_path = $1",
        image_path
    )
```

### 9. Modified Pending Image Query

```python
async def get_pending_images(pool, batch_size: int) -> List[Dict]:
    """Get images needing detection, including retriable failures."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                a.image_path as file_path,
                COALESCE(ff.retry_count, 0) as retry_count,
                ff.last_attempted_scale
            FROM ai_metadata a
            LEFT JOIN face_metadata f ON a.image_path = f.image_path
            LEFT JOIN face_detection_failures ff ON a.image_path = ff.image_path
            WHERE a.is_processed = TRUE
              AND f.image_path IS NULL
              AND (ff.image_path IS NULL
                   OR (ff.permanently_failed = FALSE
                       AND ff.next_retry_after <= CURRENT_TIMESTAMP))
            ORDER BY
                ff.retry_count NULLS FIRST,
                ff.next_retry_after
            LIMIT $1
        """, batch_size)
        return [dict(row) for row in rows]
```

### 10. Main Worker Class

Implement `FaceWorker` class with:
- `__init__`: Initialize pool and executor with `init_face_worker`
- `connect`: Create asyncpg pool to PostgreSQL
- `disconnect`: Clean shutdown
- `get_pending_images`: Query with retry support
- `save_face_data`: ONLY save successful detections
- `process_image`: Handle single image with cascade and failure tracking
- `process_batch`: Process batch with memory cleanup
- `run`: Main loop with idle detection

### 11. Key Behavior Changes

**OLD (broken):**
```python
# On failure, saved to face_metadata with 0 faces
await self.save_face_data(linux_path, {"face_count": 0})  # BAD!
```

**NEW (correct):**
```python
if result.get("success"):
    # Only save successful detections
    await self.save_face_data(conn, linux_path, result)
    await clear_detection_failure(conn, linux_path)
else:
    # Track failure for retry - do NOT save to face_metadata
    await record_detection_failure(conn, linux_path, ...)
```

## Testing

After creating the worker:

1. Run manually: `python face_worker.py`
2. Watch logs for successful detections vs failures
3. Check database:
   ```sql
   SELECT COUNT(*) FROM face_metadata;  -- Should grow
   SELECT COUNT(*) FROM face_detection_failures WHERE permanently_failed = FALSE;  -- Pending retries
   SELECT last_error_type, COUNT(*) FROM face_detection_failures GROUP BY last_error_type;
   ```

## Success Criteria

1. No more "0/5 saved" or similar in logs for OOM failures
2. `face_detection_failures` table populated with retriable failures
3. Images with OOM are retried at progressively smaller scales
4. Only truly successful detections (even with 0 faces) go to `face_metadata`
5. Memory cleaned between batches, no runaway GPU memory

## Files to Create

1. `face_worker.py` - Main worker with all above components
2. `requirements.txt`:
   ```
   asyncpg
   numpy
   Pillow
   pillow-heif
   face_recognition
   ```

Note: dlib with CUDA should already be installed from the previous VisionFaceService setup.
