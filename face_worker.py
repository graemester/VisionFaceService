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

# Circuit Breaker Settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 0.8  # 80% failure rate triggers pause
CIRCUIT_BREAKER_WINDOW_SIZE = 5          # Number of batches to track
CIRCUIT_BREAKER_PAUSE_SECONDS = 300      # 5 minutes pause when triggered
CIRCUIT_BREAKER_CONSECUTIVE_FAILURES = 10  # Pause after 10 consecutive failures

# In-Memory Blacklist (for session)
SESSION_BLACKLIST_MAX_FAILURES = 3       # Blacklist after 3 failures in same session
SESSION_BLACKLIST_DURATION_SECONDS = 3600  # 1 hour blacklist

# Executor Recovery
EXECUTOR_MAX_CONSECUTIVE_CRASHES = 5     # Restart executor after 5 crashes
EXECUTOR_RESTART_DELAY_SECONDS = 10      # Wait before restarting

# Paths (defined early for use in other constants)
WORKER_DIR = Path(r"C:\VisionFaceService")
PID_FILE = WORKER_DIR / "worker.pid"

# Health Monitoring
HEALTH_STATUS_FILE = WORKER_DIR / "health_status.json"
METRICS_FILE = WORKER_DIR / "metrics.json"
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
# CIRCUIT BREAKER & AUTO-HEALING
# ============================================================

class CircuitBreaker:
    """Tracks failure rates and triggers pauses when thresholds are exceeded."""

    def __init__(
        self,
        failure_threshold: float = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        window_size: int = CIRCUIT_BREAKER_WINDOW_SIZE,
        pause_seconds: int = CIRCUIT_BREAKER_PAUSE_SECONDS,
        max_consecutive: int = CIRCUIT_BREAKER_CONSECUTIVE_FAILURES
    ):
        self.failure_threshold = failure_threshold
        self.window_size = window_size
        self.pause_seconds = pause_seconds
        self.max_consecutive = max_consecutive
        self.batch_results: List[Tuple[int, int]] = []  # (successes, failures)
        self.consecutive_failures = 0
        self.is_open = False
        self.last_trip_time: Optional[datetime] = None
        self.trip_count = 0

    def record_batch(self, successes: int, failures: int):
        """Record batch results and check if circuit should trip."""
        self.batch_results.append((successes, failures))

        # Keep only last window_size batches
        if len(self.batch_results) > self.window_size:
            self.batch_results.pop(0)

        # Track consecutive failures
        if failures > 0 and successes == 0:
            self.consecutive_failures += failures
        else:
            self.consecutive_failures = 0

        # Check if we should trip
        self._check_trip()

    def _check_trip(self):
        """Check if circuit breaker should trip."""
        if len(self.batch_results) < self.window_size:
            return

        total_successes = sum(s for s, f in self.batch_results)
        total_failures = sum(f for s, f in self.batch_results)
        total = total_successes + total_failures

        if total == 0:
            return

        failure_rate = total_failures / total

        if failure_rate >= self.failure_threshold:
            self._trip(f"Failure rate {failure_rate:.1%} exceeds threshold {self.failure_threshold:.1%}")
        elif self.consecutive_failures >= self.max_consecutive:
            self._trip(f"Consecutive failures ({self.consecutive_failures}) exceeds threshold")

    def _trip(self, reason: str):
        """Trip the circuit breaker."""
        self.is_open = True
        self.last_trip_time = datetime.now()
        self.trip_count += 1
        logger.warning(f"CIRCUIT BREAKER TRIPPED (#{self.trip_count}): {reason}")
        logger.warning(f"Pausing for {self.pause_seconds} seconds...")

    def can_proceed(self) -> bool:
        """Check if processing can proceed."""
        if not self.is_open:
            return True

        elapsed = (datetime.now() - self.last_trip_time).total_seconds()
        if elapsed >= self.pause_seconds:
            self._reset()
            return True

        return False

    def _reset(self):
        """Reset the circuit breaker."""
        self.is_open = False
        self.batch_results.clear()
        self.consecutive_failures = 0
        logger.info("Circuit breaker reset - resuming processing")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "is_open": self.is_open,
            "consecutive_failures": self.consecutive_failures,
            "batch_count": len(self.batch_results),
            "trip_count": self.trip_count,
            "last_trip_time": self.last_trip_time.isoformat() if self.last_trip_time else None,
        }


class SessionBlacklist:
    """In-memory blacklist for images that fail repeatedly in the same session."""

    def __init__(
        self,
        max_failures: int = SESSION_BLACKLIST_MAX_FAILURES,
        duration_seconds: int = SESSION_BLACKLIST_DURATION_SECONDS
    ):
        self.max_failures = max_failures
        self.duration_seconds = duration_seconds
        self.failure_counts: Dict[str, int] = {}  # image_path -> failure count
        self.blacklist: Dict[str, datetime] = {}  # image_path -> blacklist_until

    def record_failure(self, image_path: str) -> bool:
        """Record a failure. Returns True if image is now blacklisted."""
        self.failure_counts[image_path] = self.failure_counts.get(image_path, 0) + 1

        if self.failure_counts[image_path] >= self.max_failures:
            self.blacklist[image_path] = datetime.now() + timedelta(seconds=self.duration_seconds)
            logger.warning(f"Image blacklisted for session: {image_path} "
                         f"(failed {self.failure_counts[image_path]} times)")
            return True
        return False

    def is_blacklisted(self, image_path: str) -> bool:
        """Check if image is currently blacklisted."""
        if image_path not in self.blacklist:
            return False

        if datetime.now() >= self.blacklist[image_path]:
            # Blacklist expired
            del self.blacklist[image_path]
            self.failure_counts.pop(image_path, None)
            return False

        return True

    def clear_on_success(self, image_path: str):
        """Clear failure count when an image succeeds."""
        self.failure_counts.pop(image_path, None)
        self.blacklist.pop(image_path, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get blacklist statistics."""
        return {
            "blacklisted_count": len(self.blacklist),
            "tracked_count": len(self.failure_counts),
            "blacklisted_paths": list(self.blacklist.keys())[:10],  # Sample
        }


class ExecutorManager:
    """Manages ProcessPoolExecutor with automatic recovery."""

    def __init__(self, max_crashes: int = EXECUTOR_MAX_CONSECUTIVE_CRASHES):
        self.max_crashes = max_crashes
        self.consecutive_crashes = 0
        self.executor: Optional[ProcessPoolExecutor] = None
        self.total_restarts = 0

    def create_executor(self) -> ProcessPoolExecutor:
        """Create a new executor."""
        if self.executor:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        self.executor = ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_face_worker
        )
        self.total_restarts += 1
        logger.info(f"Created new ProcessPoolExecutor (restart #{self.total_restarts})")
        return self.executor

    def record_success(self):
        """Record successful execution."""
        self.consecutive_crashes = 0

    def record_crash(self) -> bool:
        """Record a crash. Returns True if executor should be restarted."""
        self.consecutive_crashes += 1

        if self.consecutive_crashes >= self.max_crashes:
            logger.warning(f"Executor has crashed {self.consecutive_crashes} times - will restart")
            self.consecutive_crashes = 0
            return True
        return False

    def get_executor(self) -> ProcessPoolExecutor:
        """Get current executor, creating one if needed."""
        if self.executor is None:
            return self.create_executor()
        return self.executor

    def shutdown(self):
        """Shutdown executor."""
        if self.executor:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self.executor = None


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
            "original_size": None,
            "file_size": 0,
            "attempted_scale": None,
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            "error_type": "file_error",
            "face_count": 0,
            "original_size": original_size,
            "file_size": file_size,
            "attempted_scale": None,
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

            # DEFENSIVE: Check image_array is valid
            if image_array is None:
                last_error = f"Preprocessing returned None at scale {scale}"
                last_error_type = "file_error"
                continue

            # DEFENSIVE: Check dimensions
            if image_array.ndim < 2:
                last_error = f"Invalid image dimensions: ndim={image_array.ndim}"
                last_error_type = "file_error"
                continue

            # Check minimum size
            h, w = image_array.shape[:2]
            if h == 0 or w == 0:
                last_error = f"Zero dimension image: {image_array.shape}"
                last_error_type = "file_error"
                continue

            if min(h, w) < MIN_FACE_SIZE * 2:
                # Image too small at this scale - set error for tracking
                last_error = f"Image too small at scale {scale}: {w}x{h} (min {MIN_FACE_SIZE * 2}px needed)"
                last_error_type = "too_small"
                continue

            # Try CNN model first (GPU accelerated)
            try:
                face_locations = face_recognition.face_locations(
                    image_array,
                    model="cnn",
                    number_of_times_to_upsample=CNN_UPSAMPLE_TIMES
                )

                # DEFENSIVE: face_locations should never be None
                if face_locations is None:
                    face_locations = []

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
    """Hardened face detection worker with auto-healing."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.running = True
        self.processed_count = 0
        self.face_count = 0
        self.failure_count = 0
        self.has_failure_table = False
        self.start_time = datetime.now()

        # Auto-healing components
        self.circuit_breaker = CircuitBreaker()
        self.session_blacklist = SessionBlacklist()
        self.executor_manager = ExecutorManager()

    @property
    def executor(self) -> Optional[ProcessPoolExecutor]:
        """Get executor from manager for backward compatibility."""
        return self.executor_manager.executor if hasattr(self, 'executor_manager') else None

    async def connect(self):
        """Connect to PostgreSQL and ensure failure table exists."""
        dsn = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        self.pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)
        logger.info(f"Connected to PostgreSQL at {DB_HOST}")

        # Try to create failure tracking table if it doesn't exist
        async with self.pool.acquire() as conn:
            # First try to create the table
            try:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS face_detection_failures (
                        id SERIAL PRIMARY KEY,
                        image_path TEXT NOT NULL UNIQUE,
                        retry_count INTEGER DEFAULT 0,
                        last_error TEXT,
                        last_error_type TEXT,
                        original_width INTEGER,
                        original_height INTEGER,
                        file_size_bytes BIGINT,
                        last_attempted_scale REAL,
                        first_failure_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_failure_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        next_retry_after TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        permanently_failed BOOLEAN DEFAULT FALSE
                    )
                """)
                logger.info("Ensured face_detection_failures table exists")
            except asyncpg.exceptions.InsufficientPrivilegeError:
                logger.warning("Cannot create face_detection_failures table - insufficient privileges")
            except Exception as e:
                logger.warning(f"Could not create failure table: {e}")

            # Now check if we have access
            try:
                await conn.fetchval("SELECT 1 FROM face_detection_failures LIMIT 1")
                self.has_failure_table = True
                logger.info("Failure tracking table available and accessible")
            except asyncpg.exceptions.InsufficientPrivilegeError:
                self.has_failure_table = False
                logger.error("No access to face_detection_failures table - "
                           "failures will be logged but not tracked for retry. "
                           "Run: GRANT ALL ON face_detection_failures TO vision;")
            except asyncpg.exceptions.UndefinedTableError:
                self.has_failure_table = False
                logger.error("face_detection_failures table does not exist and cannot be created")

    async def disconnect(self):
        """Clean shutdown."""
        self.running = False
        if self.pool:
            await self.pool.close()
        if hasattr(self, 'executor_manager'):
            self.executor_manager.shutdown()

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
                # Note: file_size_bytes column removed as it doesn't exist in production table
                await conn.execute("""
                    INSERT INTO face_detection_failures
                        (image_path, last_error, last_error_type,
                         original_width, original_height,
                         last_attempted_scale, retry_count,
                         first_failure_at, last_failure_at, next_retry_after)
                    VALUES ($1, $2, $3, $4, $5, $6, 1,
                            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP + INTERVAL '1 second' * $7)
                    ON CONFLICT (image_path) DO UPDATE SET
                        retry_count = face_detection_failures.retry_count + 1,
                        last_error = EXCLUDED.last_error,
                        last_error_type = EXCLUDED.last_error_type,
                        last_attempted_scale = EXCLUDED.last_attempted_scale,
                        last_failure_at = CURRENT_TIMESTAMP,
                        next_retry_after = CURRENT_TIMESTAMP + INTERVAL '1 second' *
                            LEAST($7 * POWER($8, face_detection_failures.retry_count), $9),
                        permanently_failed = (face_detection_failures.retry_count + 1 >= $10)
                """,
                    image_path,
                    error[:500] if error else None,  # Truncate long errors
                    error_type,
                    original_size[0] if original_size else None,
                    original_size[1] if original_size else None,
                    attempted_scale,
                    BASE_RETRY_DELAY_SECONDS,
                    RETRY_BACKOFF_MULTIPLIER,
                    MAX_RETRY_DELAY_SECONDS,
                    MAX_RETRY_COUNT
                )
            except Exception as e:
                logger.error(f"Failed to record failure for {image_path}: {e}")
                raise  # Re-raise so caller can handle

    async def _force_permanent_failure(self, image_path: str, reason: str):
        """Last-resort method to permanently fail an image and break retry loops."""
        if not self.has_failure_table:
            logger.error(f"Cannot force-fail {image_path}: no failure table")
            return

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO face_detection_failures
                        (image_path, last_error, last_error_type, permanently_failed,
                         retry_count, next_retry_after)
                    VALUES ($1, $2, 'forced_permanent', TRUE, 99,
                            CURRENT_TIMESTAMP + INTERVAL '100 years')
                    ON CONFLICT (image_path) DO UPDATE SET
                        permanently_failed = TRUE,
                        last_error = EXCLUDED.last_error,
                        last_error_type = 'forced_permanent',
                        next_retry_after = CURRENT_TIMESTAMP + INTERVAL '100 years'
                """, image_path, reason[:500])
                logger.warning(f"Force-permanent failure: {image_path}")
        except Exception as e:
            logger.error(f"Even force_permanent_failure failed for {image_path}: {e}")

    async def process_image(self, image_info: Dict) -> Tuple[bool, int]:
        """Process a single image for face detection."""
        linux_path = image_info['file_path']
        windows_path = linux_to_windows_path(linux_path)

        # Check file exists - missing files should be IMMEDIATELY permanent (no retry)
        if not os.path.exists(windows_path):
            logger.warning(f"File not found (permanent): {windows_path}")
            await self._force_permanent_failure(
                linux_path, "File does not exist on disk"
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
                    # Detection succeeded but DB save failed - record this
                    await self.record_failure(
                        linux_path,
                        "Detection succeeded but save_face_data failed",
                        "save_error",
                        original_size=(result.get('width'), result.get('height')),
                        file_size=0,
                        attempted_scale=result.get('scale_used')
                    )
                    return False, 0
            else:
                # FAILURE - record for retry, do NOT save to face_metadata
                error = result.get("error", "Unknown error")
                error_type = result.get("error_type", "unknown")
                original_size = result.get("original_size")
                file_size = result.get("file_size", 0)
                attempted_scale = result.get("attempted_scale")

                error_msg = error[:100] if error else "Unknown error"
                logger.warning(f"Detection failed for {linux_path}: "
                             f"[{error_type}] {error_msg}")

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
            import traceback
            full_error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error processing {linux_path}: {full_error}", exc_info=True)
            await self.record_failure(
                linux_path, full_error, "crash",
                original_size=None, file_size=0, attempted_scale=None
            )
            return False, 0

    async def process_batch(self, image_infos: List[Dict]) -> Tuple[int, int]:
        """Process a batch of images with circuit breaker and blacklist support."""
        processed = 0
        total_faces = 0
        batch_failures = 0
        batch_start = time.time()

        for i, info in enumerate(image_infos):
            if not self.running:
                # Record remaining images as shutdown-skipped so they don't loop forever
                logger.info("Shutdown requested - recording remaining batch images")
                for remaining in image_infos[i:]:
                    try:
                        await self.record_failure(
                            remaining['file_path'],
                            "Worker shutdown during batch",
                            "shutdown"
                        )
                    except Exception:
                        await self._force_permanent_failure(
                            remaining['file_path'],
                            "Shutdown - could not record normally"
                        )
                break

            image_path = info['file_path']

            # DEFENSIVE: Check session blacklist
            try:
                if hasattr(self, 'session_blacklist') and self.session_blacklist.is_blacklisted(image_path):
                    logger.debug(f"Skipping blacklisted image: {image_path}")
                    # Record to DB so it doesn't keep coming back
                    try:
                        await self.record_failure(
                            image_path,
                            "Session blacklisted - repeated failures",
                            "session_blacklist"
                        )
                    except Exception as e:
                        logger.error(f"CRITICAL: Cannot record blacklist for {image_path}: {e}")
                        # Force permanent failure to break the loop
                        await self._force_permanent_failure(image_path, f"Blacklist record failed: {e}")
                    batch_failures += 1
                    continue
            except Exception as e:
                logger.warning(f"Blacklist check failed (continuing): {e}")

            # DEFENSIVE: Check circuit breaker
            try:
                if hasattr(self, 'circuit_breaker') and not self.circuit_breaker.can_proceed():
                    logger.warning(f"Circuit breaker open - waiting {CIRCUIT_BREAKER_PAUSE_SECONDS}s...")
                    await asyncio.sleep(CIRCUIT_BREAKER_PAUSE_SECONDS)
                    if not self.circuit_breaker.can_proceed():
                        logger.warning("Circuit breaker still open - recording remaining images")
                        for remaining in image_infos[i:]:
                            try:
                                await self.record_failure(
                                    remaining['file_path'],
                                    "Circuit breaker pause - batch aborted",
                                    "circuit_breaker"
                                )
                            except Exception:
                                await self._force_permanent_failure(
                                    remaining['file_path'],
                                    "Circuit breaker - could not record normally"
                                )
                        break
            except Exception as e:
                logger.warning(f"Circuit breaker check failed (continuing): {e}")

            # Process image with exception handling to prevent crashes
            try:
                success, faces = await self.process_image(info)
            except Exception as e:
                logger.error(f"Unhandled error for {image_path}: {e}")
                try:
                    await self._force_permanent_failure(image_path, f"Unhandled: {e}")
                except Exception:
                    pass
                success, faces = False, 0

            if success:
                processed += 1
                total_faces += faces
                self.processed_count += 1
                self.face_count += faces

                # Clear from blacklist on success
                try:
                    if hasattr(self, 'session_blacklist'):
                        self.session_blacklist.clear_on_success(image_path)
                    if hasattr(self, 'executor_manager'):
                        self.executor_manager.record_success()
                except Exception:
                    pass
            else:
                batch_failures += 1
                self.failure_count += 1

                # Record in session blacklist
                try:
                    if hasattr(self, 'session_blacklist'):
                        self.session_blacklist.record_failure(image_path)
                except Exception:
                    pass

                # Check if executor needs restart
                try:
                    if hasattr(self, 'executor_manager') and self.executor_manager.record_crash():
                        logger.info("Restarting executor after consecutive crashes...")
                        self.executor_manager.create_executor()
                        await asyncio.sleep(EXECUTOR_RESTART_DELAY_SECONDS)
                except Exception as e:
                    logger.warning(f"Executor recovery failed: {e}")

            # Cleanup between images
            if (i + 1) % MEMORY_CLEANUP_INTERVAL == 0:
                cleanup_gpu_memory()

        # Record batch in circuit breaker
        try:
            if hasattr(self, 'circuit_breaker'):
                self.circuit_breaker.record_batch(processed, batch_failures)
        except Exception:
            pass

        # Write health status
        try:
            self.write_health_status()
        except Exception:
            pass

        return processed, total_faces

    def write_health_status(self):
        """Write health status to file for external monitoring."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "status": "running" if self.running else "stopped",
            "processed_count": self.processed_count,
            "face_count": self.face_count,
            "failure_count": self.failure_count,
            "has_failure_table": self.has_failure_table,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }

        # Add circuit breaker status
        if hasattr(self, 'circuit_breaker'):
            status["circuit_breaker"] = self.circuit_breaker.get_status()

        # Add blacklist stats
        if hasattr(self, 'session_blacklist'):
            status["blacklist"] = self.session_blacklist.get_stats()

        # Add executor info
        if hasattr(self, 'executor_manager'):
            status["executor_restarts"] = self.executor_manager.total_restarts

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=WORKER_DIR,
                suffix='.json',
                delete=False
            ) as f:
                json.dump(status, f, indent=2)
                temp_path = f.name
            os.replace(temp_path, HEALTH_STATUS_FILE)
        except Exception as e:
            logger.debug(f"Failed to write health status: {e}")

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

        # Create executor using manager
        self.executor_manager.create_executor()

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
