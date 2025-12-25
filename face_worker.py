#!/usr/bin/env python3
"""
Vision Face Detection Worker v2 - Hardened for Reliability
===========================================================
Designed for rock-solid GPU processing on Windows with:
- Orphan worker cleanup on startup
- Single worker model (no GPU memory contention)
- Image preprocessing with downscaling
- Progressive cascade scaling on OOM
- Failure tracking with retry logic
- GPU memory cleanup between images
- HOG fallback after cascade exhausted
- Graceful shutdown with signal handlers
"""

import asyncio
import atexit
import gc
import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import numpy as np
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================
DB_HOST = "192.168.12.20"
DB_PORT = 5432
DB_NAME = "vision"
DB_USER = "vision"
DB_PASS = "vision123"

# Image access - V: mapped drive works in some contexts, UNC always works
IMAGE_MOUNT = r"\\192.168.12.20\vision_images"

# Conservative GPU Settings (GTX 1650 4GB)
MAX_IMAGE_DIMENSION = 1280          # Pre-scale all images to this max
SCALE_CASCADE = [1.0, 0.75, 0.5, 0.33, 0.25]  # Progressive downscaling on OOM
MIN_FACE_SIZE = 20                  # Minimum face size in pixels after scaling
MAX_WORKERS = 1                     # Single worker - no GPU memory contention
BATCH_SIZE = 5                      # Small batches for memory management
DETECTION_TIMEOUT = 180             # 3 minutes per image (generous for cascade)
MEMORY_CLEANUP_INTERVAL = 1         # gc.collect after every N images

# Model Settings
USE_CNN_MODEL = True                # Use CNN (GPU) by default
HOG_FALLBACK_ON_OOM = True          # Fall back to HOG if all CNN scales fail
CNN_UPSAMPLE_TIMES = 0              # No upsampling (saves VRAM)

# Retry Settings
MAX_RETRY_COUNT = 5
BASE_RETRY_DELAY_SECONDS = 60
RETRY_BACKOFF_MULTIPLIER = 2
MAX_RETRY_DELAY_SECONDS = 3600      # 1 hour max

# GPU Memory Threshold (MiB) - don't start if less than this available
MIN_GPU_MEMORY_MIB = 1500

# Paths
WORKER_DIR = Path(r"C:\VisionFaceService")
PID_FILE = WORKER_DIR / "worker.pid"
LOG_FILE = WORKER_DIR / "worker.log"
NVIDIA_SMI = Path(r"C:\Windows\System32\nvidia-smi.exe")

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# HEIC SUPPORT
# ============================================================
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False
    logger.warning("pillow-heif not installed - HEIC files will be skipped")


# ============================================================
# GPU UTILITIES
# ============================================================

def get_gpu_memory_info() -> Optional[Dict[str, int]]:
    """Get GPU memory info via nvidia-smi."""
    if not NVIDIA_SMI.exists():
        return None
    try:
        result = subprocess.run(
            [str(NVIDIA_SMI), '--query-gpu=memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            return {
                'total': int(parts[0].strip()),
                'used': int(parts[1].strip()),
                'free': int(parts[2].strip())
            }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
    return None


def cleanup_gpu_memory():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def check_gpu_ready() -> bool:
    """Check if GPU has enough free memory to proceed."""
    info = get_gpu_memory_info()
    if info is None:
        logger.warning("Cannot check GPU memory - proceeding anyway")
        return True

    if info['free'] < MIN_GPU_MEMORY_MIB:
        logger.warning(f"GPU memory low: {info['free']}MiB free, need {MIN_GPU_MEMORY_MIB}MiB")
        return False

    logger.info(f"GPU memory OK: {info['free']}MiB free of {info['total']}MiB")
    return True


# ============================================================
# ORPHAN CLEANUP
# ============================================================

def kill_orphaned_workers():
    """Kill any leftover face_worker process from a previous run using PID file."""
    my_pid = os.getpid()

    if not PID_FILE.exists():
        return 0

    try:
        old_pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError) as e:
        logger.warning(f"Could not read PID file: {e}")
        return 0

    if old_pid == my_pid:
        return 0

    # Check if that specific PID is still running
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {old_pid}', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True, timeout=10
        )

        if 'python.exe' not in result.stdout.lower():
            # Process not running, just clean up stale PID file
            logger.info(f"Removing stale PID file (PID {old_pid} not running)")
            PID_FILE.unlink(missing_ok=True)
            return 0

        # Verify it's actually a face_worker before killing
        cmd_result = subprocess.run(
            ['wmic', 'process', 'where', f'ProcessId={old_pid}',
             'get', 'CommandLine', '/VALUE'],
            capture_output=True, text=True, timeout=5
        )

        if 'face_worker' not in cmd_result.stdout.lower():
            logger.info(f"PID {old_pid} is not a face_worker, removing stale PID file")
            PID_FILE.unlink(missing_ok=True)
            return 0

        # Kill the old worker
        logger.info(f"Killing previous worker PID {old_pid}")
        subprocess.run(['taskkill', '/F', '/PID', str(old_pid)],
                      capture_output=True, timeout=5)

        # Clean up
        PID_FILE.unlink(missing_ok=True)
        time.sleep(2)  # Wait for GPU memory to be released
        cleanup_gpu_memory()
        logger.info("Previous worker terminated, GPU memory cleaned")
        return 1

    except Exception as e:
        logger.warning(f"Error during orphan cleanup: {e}")
        return 0


def write_pid_file():
    """Write current PID to file for tracking."""
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
    except Exception as e:
        logger.warning(f"Failed to write PID file: {e}")


def remove_pid_file():
    """Remove PID file on exit."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass


# ============================================================
# PATH UTILITIES
# ============================================================

def linux_to_windows_path(linux_path: str) -> str:
    """Convert database relative path to Windows UNC path."""
    return os.path.join(IMAGE_MOUNT, linux_path.replace("/", "\\"))


def is_heic_file(path: str) -> bool:
    """Check if file is HEIC/HEIF format."""
    return path.lower().endswith((".heic", ".heif"))


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_path: str,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    target_scale: float = 1.0
) -> Tuple[np.ndarray, float, Tuple[int, int], int]:
    """
    Load and preprocess image with downscaling.

    Returns:
        - Preprocessed numpy array (RGB)
        - Actual scale factor applied
        - Original dimensions (width, height)
        - File size in bytes
    """
    file_size = os.path.getsize(image_path)

    with Image.open(image_path) as img:
        original_size = img.size  # (width, height)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Calculate required scale
        max_orig_dim = max(original_size)
        dimension_scale = min(1.0, max_dimension / max_orig_dim)
        final_scale = min(dimension_scale, target_scale)

        if final_scale < 1.0:
            new_size = (
                max(1, int(original_size[0] * final_scale)),
                max(1, int(original_size[1] * final_scale))
            )
            img = img.resize(new_size, Image.LANCZOS)

        return np.array(img), final_scale, original_size, file_size


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


# ============================================================
# WORKER SUBPROCESS FUNCTIONS
# ============================================================

def init_face_worker():
    """Initialize worker subprocess with HEIC support."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass


def detect_faces_cascade(image_path: str) -> Dict[str, Any]:
    """
    Detect faces with progressive downscaling cascade.
    Tries each scale in SCALE_CASCADE, falling back on OOM.
    """
    import face_recognition

    last_error = None
    last_error_type = None
    attempted_scale = None
    original_size = None
    file_size = 0

    # First, preprocess to get dimensions and check file
    try:
        # Initial load just to get metadata
        test_array, _, original_size, file_size = preprocess_image(
            image_path, MAX_IMAGE_DIMENSION, 1.0
        )
        del test_array
        gc.collect()
    except FileNotFoundError:
        return {
            "success": False,
            "error": "File not found",
            "error_type": "file_error",
            "face_count": 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "file_error",
            "face_count": 0,
        }

    # Try each scale in the cascade
    for scale in SCALE_CASCADE:
        attempted_scale = scale
        gc.collect()  # Clean before each attempt

        try:
            # Preprocess image at this scale
            image_array, actual_scale, _, _ = preprocess_image(
                image_path,
                max_dimension=MAX_IMAGE_DIMENSION,
                target_scale=scale
            )

            # Check minimum size
            h, w = image_array.shape[:2]
            if min(h, w) < MIN_FACE_SIZE * 2:
                # Image too small at this scale
                continue

            # Try CNN model first (GPU accelerated)
            try:
                face_locations = face_recognition.face_locations(
                    image_array,
                    model="cnn",
                    number_of_times_to_upsample=CNN_UPSAMPLE_TIMES
                )
                model_used = "cnn"
            except RuntimeError as e:
                error_str = str(e).lower()
                if "cuda" in error_str or "out of memory" in error_str or "memory" in error_str:
                    # OOM - try next scale
                    last_error = str(e)
                    last_error_type = "oom"
                    del image_array
                    gc.collect()
                    continue
                else:
                    raise

            # SUCCESS!
            # Scale coordinates back to original dimensions
            scaled_locations = scale_face_locations(face_locations, actual_scale)

            # Get face encodings (on the scaled image, locations are pre-scaled)
            face_encodings = []
            if face_locations:
                try:
                    encodings = face_recognition.face_encodings(
                        image_array, face_locations
                    )
                    face_encodings = [enc.tolist() for enc in encodings]
                except Exception as enc_err:
                    logger.warning(f"Encoding error (non-fatal): {enc_err}")

            del image_array
            gc.collect()

            return {
                "success": True,
                "face_count": len(face_locations),
                "face_locations": [list(loc) for loc in scaled_locations],
                "face_encodings": face_encodings,
                "scale_used": actual_scale,
                "model_used": model_used,
                "original_size": original_size,
                "file_size": file_size,
            }

        except RuntimeError as e:
            error_str = str(e).lower()
            if "cuda" in error_str or "out of memory" in error_str or "memory" in error_str:
                last_error = str(e)
                last_error_type = "oom"
                gc.collect()
                continue
            else:
                last_error = str(e)
                last_error_type = "runtime"
                break

        except Exception as e:
            last_error = str(e)
            last_error_type = "unknown"
            break

    # CNN cascade exhausted - try HOG fallback on smallest scale
    if HOG_FALLBACK_ON_OOM and last_error_type == "oom":
        try:
            gc.collect()
            # Use smallest scale for HOG
            min_scale = min(SCALE_CASCADE)
            image_array, actual_scale, _, _ = preprocess_image(
                image_path,
                max_dimension=MAX_IMAGE_DIMENSION,
                target_scale=min_scale
            )

            face_locations = face_recognition.face_locations(
                image_array, model="hog"
            )

            scaled_locations = scale_face_locations(face_locations, actual_scale)

            face_encodings = []
            if face_locations:
                try:
                    encodings = face_recognition.face_encodings(
                        image_array, face_locations
                    )
                    face_encodings = [enc.tolist() for enc in encodings]
                except Exception:
                    pass

            del image_array
            gc.collect()

            return {
                "success": True,
                "face_count": len(face_locations),
                "face_locations": [list(loc) for loc in scaled_locations],
                "face_encodings": face_encodings,
                "scale_used": actual_scale,
                "model_used": "hog_fallback",
                "original_size": original_size,
                "file_size": file_size,
            }

        except Exception as hog_err:
            last_error = f"HOG fallback failed: {hog_err}"
            last_error_type = "hog_failed"

    # All attempts failed
    gc.collect()
    return {
        "success": False,
        "face_count": 0,
        "error": last_error,
        "error_type": last_error_type,
        "attempted_scale": attempted_scale,
        "original_size": original_size,
        "file_size": file_size,
    }


# ============================================================
# MAIN WORKER CLASS
# ============================================================

class FaceWorker:
    """Hardened face detection worker."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.executor: Optional[ProcessPoolExecutor] = None
        self.running = True
        self.processed_count = 0
        self.face_count = 0
        self.failure_count = 0
        self.has_failure_table = False

    async def connect(self):
        """Connect to PostgreSQL and check table access."""
        dsn = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        self.pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)
        logger.info(f"Connected to PostgreSQL at {DB_HOST}")

        # Check if we have access to failure tracking table
        async with self.pool.acquire() as conn:
            try:
                await conn.fetchval("SELECT 1 FROM face_detection_failures LIMIT 1")
                self.has_failure_table = True
                logger.info("Failure tracking table available")
            except asyncpg.exceptions.InsufficientPrivilegeError:
                self.has_failure_table = False
                logger.warning("No access to face_detection_failures table - "
                             "failures will be logged but not tracked for retry")
            except asyncpg.exceptions.UndefinedTableError:
                self.has_failure_table = False
                logger.warning("face_detection_failures table does not exist")

    async def disconnect(self):
        """Clean shutdown."""
        self.running = False
        if self.pool:
            await self.pool.close()
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)

    async def get_pending_images(self, batch_size: int) -> List[Dict]:
        """Get images needing face detection, including retriable failures."""
        async with self.pool.acquire() as conn:
            if self.has_failure_table:
                # Include retry logic
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
                        ff.next_retry_after NULLS FIRST
                    LIMIT $1
                """, batch_size)
            else:
                # Simple query without failure tracking
                rows = await conn.fetch("""
                    SELECT a.image_path as file_path, 0 as retry_count, NULL as last_attempted_scale
                    FROM ai_metadata a
                    LEFT JOIN face_metadata f ON a.image_path = f.image_path
                    WHERE a.is_processed = TRUE
                      AND f.image_path IS NULL
                    LIMIT $1
                """, batch_size)

            return [dict(row) for row in rows]

    async def save_face_data(self, image_path: str, data: Dict[str, Any]) -> bool:
        """Save successful face detection results."""
        async with self.pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO face_metadata
                        (image_path, face_count, face_locations, face_encodings,
                         recognized_persons, orientation_hint, detection_date)
                    VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                    ON CONFLICT (image_path) DO UPDATE SET
                        face_count = EXCLUDED.face_count,
                        face_locations = EXCLUDED.face_locations,
                        face_encodings = EXCLUDED.face_encodings,
                        detection_date = CURRENT_TIMESTAMP
                """,
                    image_path,
                    data.get("face_count", 0),
                    json.dumps(data.get("face_locations", [])),
                    json.dumps(data.get("face_encodings", [])),
                    json.dumps([]),  # recognized_persons filled by clustering
                    data.get("orientation_hint"),
                )

                # Clear any failure record
                if self.has_failure_table:
                    await conn.execute(
                        "DELETE FROM face_detection_failures WHERE image_path = $1",
                        image_path
                    )

                return True
            except Exception as e:
                logger.error(f"Save error for {image_path}: {e}")
                return False

    async def record_failure(
        self,
        image_path: str,
        error: str,
        error_type: str,
        original_size: Optional[Tuple[int, int]] = None,
        file_size: int = 0,
        attempted_scale: Optional[float] = None
    ):
        """Record detection failure for later retry."""
        if not self.has_failure_table:
            logger.warning(f"FAILURE (not tracked): {image_path} - {error_type}: {error}")
            return

        async with self.pool.acquire() as conn:
            try:
                # Calculate next retry delay with exponential backoff
                await conn.execute("""
                    INSERT INTO face_detection_failures
                        (image_path, last_error, last_error_type,
                         original_width, original_height, file_size_bytes,
                         last_attempted_scale, retry_count,
                         first_failure_at, last_failure_at, next_retry_after)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, 1,
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP + INTERVAL '1 second' * $8)
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
                    error[:500] if error else None,  # Truncate long errors
                    error_type,
                    original_size[0] if original_size else None,
                    original_size[1] if original_size else None,
                    file_size,
                    attempted_scale,
                    BASE_RETRY_DELAY_SECONDS,
                    RETRY_BACKOFF_MULTIPLIER,
                    MAX_RETRY_DELAY_SECONDS,
                    MAX_RETRY_COUNT
                )
            except Exception as e:
                logger.error(f"Failed to record failure for {image_path}: {e}")

    async def process_image(self, image_info: Dict) -> Tuple[bool, int]:
        """Process a single image for face detection."""
        linux_path = image_info['file_path']
        windows_path = linux_to_windows_path(linux_path)

        # Check file exists
        if not os.path.exists(windows_path):
            logger.warning(f"File not found: {windows_path}")
            await self.record_failure(
                linux_path, "File not found", "file_error"
            )
            return False, 0

        # Skip HEIC if not supported
        if is_heic_file(windows_path) and not HEIC_AVAILABLE:
            logger.debug(f"Skipping HEIC (not supported): {linux_path}")
            await self.record_failure(
                linux_path, "HEIC not supported", "file_error"
            )
            return False, 0

        loop = asyncio.get_event_loop()

        try:
            # Run detection in subprocess with timeout
            future = loop.run_in_executor(
                self.executor, detect_faces_cascade, windows_path
            )
            result = await asyncio.wait_for(future, timeout=DETECTION_TIMEOUT)

            if result.get("success"):
                # SUCCESS - save to face_metadata
                if await self.save_face_data(linux_path, result):
                    faces = result.get("face_count", 0)
                    scale = result.get("scale_used", 1.0)
                    model = result.get("model_used", "unknown")
                    if faces > 0:
                        logger.info(f"Found {faces} face(s) in {linux_path} "
                                  f"[scale={scale:.0%}, model={model}]")
                    return True, faces
                else:
                    return False, 0
            else:
                # FAILURE - record for retry, do NOT save to face_metadata
                error = result.get("error", "Unknown error")
                error_type = result.get("error_type", "unknown")
                original_size = result.get("original_size")
                file_size = result.get("file_size", 0)
                attempted_scale = result.get("attempted_scale")

                logger.warning(f"Detection failed for {linux_path}: "
                             f"[{error_type}] {error[:100]}")

                await self.record_failure(
                    linux_path, error, error_type,
                    original_size, file_size, attempted_scale
                )
                return False, 0

        except asyncio.TimeoutError:
            logger.warning(f"Timeout ({DETECTION_TIMEOUT}s) for {linux_path}")
            await self.record_failure(
                linux_path, f"Timeout after {DETECTION_TIMEOUT}s", "timeout"
            )
            return False, 0

        except Exception as e:
            logger.error(f"Error processing {linux_path}: {e}")
            await self.record_failure(
                linux_path, str(e), "crash"
            )
            return False, 0

    async def process_batch(self, image_infos: List[Dict]) -> Tuple[int, int]:
        """Process a batch of images with memory cleanup."""
        processed = 0
        total_faces = 0

        for i, info in enumerate(image_infos):
            if not self.running:
                break

            success, faces = await self.process_image(info)
            if success:
                processed += 1
                total_faces += faces
                self.processed_count += 1
                self.face_count += faces
            else:
                self.failure_count += 1

            # Cleanup between images
            if (i + 1) % MEMORY_CLEANUP_INTERVAL == 0:
                cleanup_gpu_memory()

        return processed, total_faces

    async def run(self):
        """Main worker loop."""
        # Startup
        kill_orphaned_workers()
        write_pid_file()
        atexit.register(remove_pid_file)

        # Check GPU
        if not check_gpu_ready():
            logger.error("GPU not ready - waiting 30s then retrying")
            await asyncio.sleep(30)
            if not check_gpu_ready():
                logger.error("GPU still not ready - exiting")
                return

        await self.connect()

        # Create executor with single worker
        self.executor = ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_face_worker
        )

        try:
            idle_count = 0

            while self.running:
                # Periodic GPU check
                if self.processed_count > 0 and self.processed_count % 50 == 0:
                    if not check_gpu_ready():
                        logger.warning("GPU memory low - pausing 30s for cleanup")
                        cleanup_gpu_memory()
                        await asyncio.sleep(30)
                        continue

                # Get pending images
                image_infos = await self.get_pending_images(BATCH_SIZE)

                if not image_infos:
                    idle_count += 1
                    if idle_count == 1:
                        logger.info(f"No pending images. Stats: "
                                  f"{self.processed_count} processed, "
                                  f"{self.face_count} faces, "
                                  f"{self.failure_count} failures. Waiting...")
                    await asyncio.sleep(30)
                    continue

                idle_count = 0
                retry_info = ""
                if any(info.get('retry_count', 0) > 0 for info in image_infos):
                    retries = sum(1 for i in image_infos if i.get('retry_count', 0) > 0)
                    retry_info = f" ({retries} retries)"

                logger.info(f"Processing batch of {len(image_infos)} images{retry_info}...")

                processed, faces = await self.process_batch(image_infos)

                logger.info(f"Batch: {processed}/{len(image_infos)} OK, {faces} faces. "
                          f"Total: {self.processed_count} processed, "
                          f"{self.face_count} faces, {self.failure_count} failures")

                # Brief pause between batches
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown requested (Ctrl+C)...")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self.running = False
            await self.disconnect()
            remove_pid_file()
            logger.info("Worker stopped")


# ============================================================
# SIGNAL HANDLERS
# ============================================================

def handle_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum} - shutting down...")
    # The running flag will be checked in the main loop
    raise KeyboardInterrupt()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

async def main():
    # Check dlib CUDA status
    try:
        import dlib
        cuda_status = "ENABLED" if dlib.DLIB_USE_CUDA else "DISABLED"
    except Exception:
        cuda_status = "UNKNOWN"

    gpu_info = get_gpu_memory_info()
    gpu_str = f"{gpu_info['free']}MiB free" if gpu_info else "unknown"

    logger.info("=" * 70)
    logger.info("Vision Face Detection Worker v2 - HARDENED")
    logger.info("=" * 70)
    logger.info(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    logger.info(f"Image mount: {IMAGE_MOUNT}")
    logger.info(f"Workers: {MAX_WORKERS} (single-worker mode)")
    logger.info(f"Max image dimension: {MAX_IMAGE_DIMENSION}px")
    logger.info(f"Scale cascade: {SCALE_CASCADE}")
    logger.info(f"CUDA: {cuda_status}")
    logger.info(f"GPU Memory: {gpu_str}")
    logger.info(f"HEIC support: {HEIC_AVAILABLE}")
    logger.info("=" * 70)

    worker = FaceWorker()
    await worker.run()


if __name__ == "__main__":
    # Setup signal handlers (Windows compatible)
    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except (AttributeError, OSError):
        pass  # Some signals not available on Windows

    asyncio.run(main())
