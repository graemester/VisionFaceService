# VisionFaceService Face Worker Documentation

## Overview

The Vision Face Detection Worker is a hardened, GPU-accelerated face detection service designed to run on Windows with an NVIDIA GPU. It processes images from a Vision AI system running on Linux, detecting faces and storing results in a shared PostgreSQL database.

**Key Features:**
- GPU-accelerated face detection using dlib's CNN model with CUDA
- Progressive downscaling cascade to handle GPU memory constraints
- Automatic failure tracking with exponential backoff retry
- Circuit breaker pattern for auto-healing
- Session blacklisting to prevent infinite retry loops
- Graceful shutdown with state preservation
- HEIC/HEIF image support

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Vision Server (Linux @ 192.168.12.20)                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     PostgreSQL Database                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │
│  │  │   images    │  │ ai_metadata │  │     face_metadata       │   │   │
│  │  │ (all files) │  │ (processed) │  │ (detection results)     │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │   │
│  │                                    ┌─────────────────────────┐   │   │
│  │                                    │ face_detection_failures │   │   │
│  │                                    │   (retry tracking)      │   │   │
│  │                                    └─────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              SMB Share: \\192.168.12.20\vision_images            │   │
│  │                    → /mnt/icculus/pictures                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Network
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                    GAME Worker (Windows @ 192.168.12.75)                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      face_worker.py                              │   │
│  │  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────┐     │   │
│  │  │ FaceWorker  │  │ CircuitBreaker  │  │  SessionBlacklist │     │   │
│  │  │  (main)     │  │ (auto-healing)  │  │ (loop prevention) │     │   │
│  │  └─────────────┘  └─────────────────┘  └───────────────────┘     │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────────────┐   │   │
│  │  │ ExecutorManager │  │     ProcessPoolExecutor              │   │   │
│  │  │ (crash recovery)│  │  (subprocess for face_recognition)   │   │   │
│  │  └─────────────────┘  └──────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    NVIDIA GTX 1650 (4GB VRAM)                    │   │
│  │                    dlib CNN + CUDA acceleration                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Configuration

All configuration is defined at the top of `face_worker.py`:

### Database Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `DB_HOST` | `192.168.12.20` | PostgreSQL server |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `vision` | Database name |
| `DB_USER` | `vision` | Database user |
| `DB_PASS` | `vision123` | Database password |

### Image Access

| Setting | Value | Description |
|---------|-------|-------------|
| `IMAGE_MOUNT` | `\\192.168.12.20\vision_images` | UNC path to image share |

### GPU Settings (GTX 1650 4GB)

| Setting | Value | Description |
|---------|-------|-------------|
| `MAX_IMAGE_DIMENSION` | `1280` | Pre-scale images to this max dimension |
| `SCALE_CASCADE` | `[1.0, 0.75, 0.5, 0.33, 0.25]` | Progressive downscaling on OOM |
| `MIN_FACE_SIZE` | `20` | Minimum detectable face size (pixels) |
| `MAX_WORKERS` | `1` | Single worker to avoid GPU memory contention |
| `BATCH_SIZE` | `5` | Images per batch |
| `DETECTION_TIMEOUT` | `180` | Seconds before timeout per image |
| `MIN_GPU_MEMORY_MIB` | `1500` | Minimum free VRAM to start processing |

### Model Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `USE_CNN_MODEL` | `True` | Use GPU-accelerated CNN model |
| `HOG_FALLBACK_ON_OOM` | `True` | Fall back to CPU HOG if CNN fails |
| `CNN_UPSAMPLE_TIMES` | `0` | No upsampling (saves VRAM) |

### Retry Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `MAX_RETRY_COUNT` | `5` | Retries before permanent failure |
| `BASE_RETRY_DELAY_SECONDS` | `60` | Initial retry delay |
| `RETRY_BACKOFF_MULTIPLIER` | `2` | Exponential backoff multiplier |
| `MAX_RETRY_DELAY_SECONDS` | `3600` | Maximum retry delay (1 hour) |

### Circuit Breaker Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `0.8` | 80% failure rate triggers pause |
| `CIRCUIT_BREAKER_WINDOW_SIZE` | `5` | Batches tracked for failure rate |
| `CIRCUIT_BREAKER_PAUSE_SECONDS` | `300` | Pause duration (5 minutes) |
| `CIRCUIT_BREAKER_CONSECUTIVE_FAILURES` | `10` | Consecutive failures trigger pause |

### Session Blacklist Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `SESSION_BLACKLIST_MAX_FAILURES` | `3` | Failures before session blacklist |
| `SESSION_BLACKLIST_DURATION_SECONDS` | `3600` | Blacklist duration (1 hour) |

### Executor Recovery Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `EXECUTOR_MAX_CONSECUTIVE_CRASHES` | `5` | Crashes before executor restart |
| `EXECUTOR_RESTART_DELAY_SECONDS` | `10` | Delay before restart |

## Database Schema

### face_metadata (Detection Results)

```sql
CREATE TABLE face_metadata (
    image_path TEXT PRIMARY KEY,
    face_count INTEGER DEFAULT 0,
    face_locations TEXT,      -- JSON: [[top, right, bottom, left], ...]
    face_encodings TEXT,      -- JSON: [[128 floats], ...]
    recognized_persons TEXT,  -- JSON: ["name1", "name2", ...]
    orientation_hint INTEGER,
    detection_date TIMESTAMP
);
```

### face_detection_failures (Retry Tracking)

```sql
CREATE TABLE face_detection_failures (
    id SERIAL PRIMARY KEY,
    image_path TEXT NOT NULL UNIQUE,
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,
    last_error_type TEXT,     -- 'oom', 'timeout', 'crash', 'file_error', etc.
    original_width INTEGER,
    original_height INTEGER,
    last_attempted_scale REAL,
    first_failure_at TIMESTAMP,
    last_failure_at TIMESTAMP,
    next_retry_after TIMESTAMP,
    permanently_failed BOOLEAN DEFAULT FALSE
);
```

**Error Types:**
| Type | Description |
|------|-------------|
| `oom` | GPU out of memory |
| `timeout` | Detection exceeded timeout |
| `crash` | Worker process crashed |
| `file_error` | File not found, corrupted, or unsupported |
| `too_small` | Image too small for face detection |
| `save_error` | Detection succeeded but DB save failed |
| `session_blacklist` | Blacklisted after repeated session failures |
| `circuit_breaker` | Skipped due to circuit breaker |
| `shutdown` | Skipped due to worker shutdown |
| `forced_permanent` | Force-failed to break retry loop |

## Key Components

### FaceWorker (Main Class)

The main worker class that orchestrates everything:

```python
class FaceWorker:
    def __init__(self):
        self.pool                # asyncpg connection pool
        self.running             # shutdown flag
        self.circuit_breaker     # CircuitBreaker instance
        self.session_blacklist   # SessionBlacklist instance
        self.executor_manager    # ExecutorManager instance

    async def run()              # Main loop
    async def process_batch()    # Process a batch of images
    async def process_image()    # Process single image
    async def record_failure()   # Record failure for retry
    async def save_face_data()   # Save successful detection
```

### CircuitBreaker (Auto-Healing)

Prevents the worker from spinning when something is systematically wrong:

```python
class CircuitBreaker:
    def record_batch(successes, failures)  # Track batch results
    def can_proceed() -> bool              # Check if processing allowed
    def _trip(reason)                      # Trigger pause
    def _reset()                           # Resume processing
```

**Triggers:**
1. Failure rate exceeds 80% over 5 batches
2. 10+ consecutive failures

**Behavior:** Pauses processing for 5 minutes, then resets and retries.

### SessionBlacklist (Loop Prevention)

Prevents the same image from being retried endlessly within a session:

```python
class SessionBlacklist:
    def record_failure(path) -> bool    # Returns True if now blacklisted
    def is_blacklisted(path) -> bool    # Check if blacklisted
    def clear_on_success(path)          # Clear on successful processing
```

**Behavior:** After 3 failures in the same session, the image is blacklisted for 1 hour and recorded as permanently failed in the database.

### ExecutorManager (Crash Recovery)

Manages the ProcessPoolExecutor with automatic recovery:

```python
class ExecutorManager:
    def create_executor()       # Create new executor
    def record_success()        # Reset crash counter
    def record_crash() -> bool  # Returns True if restart needed
    def shutdown()              # Clean shutdown
```

**Behavior:** After 5 consecutive crashes, recreates the executor with a fresh subprocess.

## Face Detection Pipeline

### 1. Image Selection

```sql
SELECT a.image_path
FROM ai_metadata a
LEFT JOIN face_metadata f ON a.image_path = f.image_path
LEFT JOIN face_detection_failures ff ON a.image_path = ff.image_path
WHERE a.is_processed = TRUE
  AND f.image_path IS NULL
  AND (ff.image_path IS NULL
       OR (ff.permanently_failed = FALSE
           AND ff.next_retry_after <= CURRENT_TIMESTAMP))
ORDER BY ff.retry_count NULLS FIRST
LIMIT 5
```

### 2. Path Conversion

```
Database path:  apple/IMG_1234.JPG
Windows path:   \\192.168.12.20\vision_images\apple\IMG_1234.JPG
```

### 3. Image Preprocessing

1. Load image with PIL (HEIC support via pillow-heif)
2. Convert to RGB if needed
3. Calculate scale factor: `min(1.0, 1280 / max_dimension)`
4. Resize with LANCZOS resampling
5. Convert to numpy array

### 4. Face Detection Cascade

```
┌─────────────────────────────────────────────────────────────┐
│                    Scale Cascade                            │
│                                                             │
│  Scale 1.0 (100%) ──► CNN detect ──► Success? ──► Done     │
│         │                              │                    │
│         │                           OOM?                    │
│         ▼                              │                    │
│  Scale 0.75 (75%) ──► CNN detect ──► Success? ──► Done     │
│         │                              │                    │
│         │                           OOM?                    │
│         ▼                              │                    │
│  Scale 0.5 (50%) ──► CNN detect ──► Success? ──► Done      │
│         │                              │                    │
│         │                           OOM?                    │
│         ▼                              │                    │
│  Scale 0.33 (33%) ──► CNN detect ──► Success? ──► Done     │
│         │                              │                    │
│         │                           OOM?                    │
│         ▼                              │                    │
│  Scale 0.25 (25%) ──► CNN detect ──► Success? ──► Done     │
│         │                              │                    │
│         │                           OOM?                    │
│         ▼                              │                    │
│  HOG Fallback (CPU) ──────────────► Success? ──► Done      │
│                                        │                    │
│                                     Failed                  │
│                                        │                    │
│                                        ▼                    │
│                              Record Failure                 │
└─────────────────────────────────────────────────────────────┘
```

### 5. Coordinate Scaling

Face coordinates are scaled back to original image dimensions:

```python
scaled_location = (
    int(top / scale_factor),
    int(right / scale_factor),
    int(bottom / scale_factor),
    int(left / scale_factor)
)
```

### 6. Face Encoding

128-dimensional face encodings are extracted for each detected face using `face_recognition.face_encodings()`.

## Failure Handling

### Retry Logic

Failures use exponential backoff:

```
Attempt 1: Wait 60 seconds
Attempt 2: Wait 120 seconds (60 * 2^1)
Attempt 3: Wait 240 seconds (60 * 2^2)
Attempt 4: Wait 480 seconds (60 * 2^3)
Attempt 5: Wait 960 seconds (60 * 2^4)
Attempt 6: Permanently failed
```

Maximum delay capped at 3600 seconds (1 hour).

### Force Permanent Failure

When normal failure recording fails, `_force_permanent_failure()` ensures the image is marked permanent:

```python
async def _force_permanent_failure(self, image_path: str, reason: str):
    """Last-resort method to permanently fail an image."""
    await conn.execute("""
        INSERT INTO face_detection_failures (...)
        VALUES (..., permanently_failed=TRUE, retry_count=99, ...)
        ON CONFLICT DO UPDATE SET permanently_failed=TRUE
    """)
```

### Guaranteed Recording

All code paths that skip images ensure database recording:

1. **Session blacklist skip** → `record_failure()` with fallback to `_force_permanent_failure()`
2. **Circuit breaker abort** → Records all remaining batch images
3. **Graceful shutdown** → Records all remaining batch images
4. **save_face_data() failure** → `record_failure()` with `save_error` type

## GPU Memory Management

### Strategies

1. **Pre-scaling**: All images scaled to max 1280px before detection
2. **Progressive cascade**: Try smaller scales on OOM
3. **Single worker**: Only one detection at a time
4. **Memory cleanup**: `gc.collect()` and `torch.cuda.empty_cache()` between images
5. **GPU monitoring**: Check free VRAM before processing

### Memory Check

```python
def check_gpu_ready() -> bool:
    info = get_gpu_memory_info()  # via nvidia-smi
    return info['free'] >= MIN_GPU_MEMORY_MIB  # 1500 MiB
```

## Installation

### Prerequisites

- Windows 10/11
- NVIDIA GPU with CUDA support
- Python 3.8+
- Network access to Vision server

### Setup

```powershell
# Create directory
mkdir C:\VisionFaceService
cd C:\VisionFaceService

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install asyncpg numpy Pillow pillow-heif face_recognition

# dlib with CUDA (requires CUDA toolkit)
pip install dlib  # or build from source with CUDA
```

### Verify CUDA

```python
import dlib
print(f"CUDA enabled: {dlib.DLIB_USE_CUDA}")
```

## Running the Service

### Manual Execution

```powershell
cd C:\VisionFaceService
.\venv\Scripts\Activate.ps1
python face_worker.py
```

### As Windows Service (NSSM)

```powershell
# Install NSSM
# Download from https://nssm.cc/

# Install service
nssm install VisionFaceWorker C:\VisionFaceService\venv\Scripts\python.exe
nssm set VisionFaceWorker AppDirectory C:\VisionFaceService
nssm set VisionFaceWorker AppParameters face_worker.py
nssm set VisionFaceWorker DisplayName "Vision Face Detection Worker"
nssm set VisionFaceWorker Description "GPU-accelerated face detection for Vision AI"
nssm set VisionFaceWorker Start SERVICE_AUTO_START

# Start service
nssm start VisionFaceWorker
```

### Service Management

```powershell
# Start/Stop
nssm start VisionFaceWorker
nssm stop VisionFaceWorker

# Check status
nssm status VisionFaceWorker

# View logs
Get-Content C:\VisionFaceService\worker.log -Tail 50 -Wait
```

## Monitoring

### Log File

`C:\VisionFaceService\worker.log`

```
2025-12-26 14:01:05 - INFO - Vision Face Detection Worker v2 - HARDENED
2025-12-26 14:01:05 - INFO - Database: 192.168.12.20:5432/vision
2025-12-26 14:01:05 - INFO - CUDA: ENABLED
2025-12-26 14:01:05 - INFO - GPU Memory: 1960MiB free
2025-12-26 14:01:06 - INFO - Processing batch of 5 images...
2025-12-26 14:01:08 - INFO - Found 2 face(s) in apple/photo.jpg [scale=75%, model=cnn]
2025-12-26 14:01:10 - INFO - Batch: 4/5 OK, 3 faces. Total: 100 processed, 50 faces, 5 failures
```

### Health Status File

`C:\VisionFaceService\health_status.json`

```json
{
  "timestamp": "2025-12-26T14:01:10",
  "status": "running",
  "processed_count": 100,
  "face_count": 50,
  "failure_count": 5,
  "has_failure_table": true,
  "uptime_seconds": 3600,
  "circuit_breaker": {
    "is_open": false,
    "consecutive_failures": 0,
    "trip_count": 0
  },
  "blacklist": {
    "blacklisted_count": 2,
    "tracked_count": 5
  },
  "executor_restarts": 1
}
```

### Database Queries

```sql
-- Processing progress
SELECT COUNT(*) as detected FROM face_metadata;

-- Failure summary
SELECT
    last_error_type,
    permanently_failed,
    COUNT(*) as count
FROM face_detection_failures
GROUP BY last_error_type, permanently_failed
ORDER BY count DESC;

-- Recent activity
SELECT image_path, detection_date
FROM face_metadata
ORDER BY detection_date DESC
LIMIT 10;

-- Pending retries
SELECT image_path, retry_count, next_retry_after
FROM face_detection_failures
WHERE permanently_failed = FALSE
ORDER BY next_retry_after;

-- Images with most faces
SELECT image_path, face_count
FROM face_metadata
WHERE face_count > 0
ORDER BY face_count DESC
LIMIT 10;
```

## Troubleshooting

### Common Issues

#### "No pending images" but images exist

Check that images are in `ai_metadata` with `is_processed = TRUE`:
```sql
SELECT COUNT(*) FROM ai_metadata WHERE is_processed = TRUE;
```

#### GPU Out of Memory

1. Check GPU memory: `nvidia-smi`
2. Reduce `MAX_IMAGE_DIMENSION`
3. Increase `SCALE_CASCADE` steps
4. Kill other GPU processes

#### "Circuit breaker tripped"

Something is systematically failing. Check:
1. Database connectivity
2. SMB share access
3. GPU health
4. Recent error types in `face_detection_failures`

#### "Cannot access shared folder" (WinError 1272)

SMB credentials not configured:
```powershell
net use \\192.168.12.20\vision_images /user:vision vision123 /persistent:yes
```

#### Infinite retry loop

This should be fixed by the hardening. If it occurs:
1. Check `face_detection_failures` for images with high `retry_count`
2. Mark stuck images as permanently failed:
```sql
UPDATE face_detection_failures
SET permanently_failed = TRUE
WHERE retry_count >= 5;
```

### Debug Mode

Add to `face_worker.py`:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Maintenance Scripts

### analyze_failures.py
Analyze permanent failures by error type.

### fix_stuck.py
Mark stuck retrying images as permanently failed.

### fix_problem_filenames.py
Find and blacklist images with problematic filenames.

### check_tiny.py
Investigate 1x1 pixel images (tracking pixels).

## Version History

### v2 - Hardened (Current)
- Added CircuitBreaker for auto-healing
- Added SessionBlacklist to prevent infinite loops
- Added ExecutorManager for crash recovery
- Added `_force_permanent_failure()` as last resort
- Fixed off-by-one in `permanently_failed` check
- Fixed all code paths to guarantee failure recording
- Added proper error_type tracking for small images

### v1 - Initial
- Basic face detection with retry logic
- GPU memory management with scale cascade
- HOG fallback on CNN failure
