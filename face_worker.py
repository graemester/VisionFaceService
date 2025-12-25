#!/usr/bin/env python3
"""
Vision Face Detection Worker for Windows
Detects faces and computes encodings, stores in shared PostgreSQL database.
Supports GPU acceleration via CUDA-enabled dlib.
"""

import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:\\VisionFaceService\\worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
DB_HOST = "192.168.12.20"  # Vision server
DB_PORT = 5432
DB_NAME = "vision"
DB_USER = "vision"
DB_PASS = "vision123"

IMAGE_MOUNT = r"\\192.168.12.20\vision_images"  # UNC path (mapped drives unreliable)

# GPU Memory Tuning:
# - GTX 1650 4GB:  BATCH_SIZE=10, MAX_WORKERS=2
# - RTX 3060 8GB:  BATCH_SIZE=20, MAX_WORKERS=4
# - RTX 3080 12GB: BATCH_SIZE=30, MAX_WORKERS=6
BATCH_SIZE = 10
MAX_WORKERS = 2  # Keep low for 4GB VRAM to avoid OOM
DETECTION_TIMEOUT = 60  # Seconds per image
# ============================================================


# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False
    logger.warning("pillow-heif not installed - HEIC files will be skipped")


def windows_to_linux_path(windows_path: str) -> str:
    """Convert Windows path to relative path matching images table.

    The images table stores RELATIVE paths (e.g., 'subfolder/image.jpg'),
    not full Linux paths. This function extracts the relative path.
    """
    # \\192.168.12.20\vision_images\subfolder\image.jpg -> subfolder/image.jpg
    rel_path = windows_path.replace(IMAGE_MOUNT, "").lstrip("\\").replace("\\", "/")
    return rel_path.lstrip("/")


def linux_to_windows_path(linux_path: str) -> str:
    """Convert relative path from images table to Windows UNC path."""
    # subfolder/image.jpg -> \\192.168.12.20\vision_images\subfolder\image.jpg
    return os.path.join(IMAGE_MOUNT, linux_path.replace("/", "\\"))


def is_heic_file(path: str) -> bool:
    """Check if file is HEIC/HEIF format."""
    return path.lower().endswith((".heic", ".heif"))


def init_face_worker():
    """Initialize worker process with HEIC support. Called once per subprocess."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass


def detect_faces_worker(image_path: str) -> Dict[str, Any]:
    """
    Worker function to detect faces in an image.
    Runs in a separate process for crash isolation.
    """
    try:
        import face_recognition

        # Handle HEIC files
        if is_heic_file(image_path):
            if not HEIC_AVAILABLE:
                return {
                    "success": False,
                    "error": "HEIC not supported",
                    "face_count": 0,
                }
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                image = np.array(img)
        else:
            image = face_recognition.load_image_file(image_path)

        # Detect faces using HOG model (faster) or CNN model (more accurate)
        # Use "cnn" for GPU acceleration, "hog" for CPU
        model = "cnn" if hasattr(face_recognition, 'face_locations') else "hog"
        try:
            face_locations = face_recognition.face_locations(image, model="cnn")
        except Exception:
            # Fall back to HOG if CNN fails (no GPU)
            face_locations = face_recognition.face_locations(image, model="hog")

        if not face_locations:
            return {
                "success": True,
                "face_count": 0,
                "face_locations": [],
                "face_encodings": [],
                "orientation_hint": None,
            }

        # Get face encodings (128-dimensional vectors)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_encodings_list = [enc.tolist() for enc in face_encodings]

        # Calculate orientation hint based on facial landmarks
        orientation_hint = calculate_orientation_hint(image, face_locations)

        return {
            "success": True,
            "face_count": len(face_locations),
            "face_locations": [list(loc) for loc in face_locations],
            "face_encodings": face_encodings_list,
            "orientation_hint": orientation_hint,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "face_count": 0,
        }


def calculate_orientation_hint(image: np.ndarray, face_locations: list) -> Optional[int]:
    """
    Determine if image needs rotation based on face orientation.
    Returns: 0, 90, 180, or 270 degrees, or None if undetermined.
    """
    if not face_locations:
        return None

    try:
        import face_recognition

        landmarks = face_recognition.face_landmarks(image, face_locations)
        if not landmarks:
            return None

        # Check if faces are upright
        upright_score = calculate_upright_score(landmarks)

        if upright_score > 0.7:
            return 0  # Already upright

        # Try rotations
        best_rotation = 0
        best_score = upright_score

        for rotation in [90, 180, 270]:
            if rotation == 90:
                rotated = np.rot90(image, k=3)
            elif rotation == 180:
                rotated = np.rot90(image, k=2)
            else:
                rotated = np.rot90(image, k=1)

            rotated_locations = face_recognition.face_locations(rotated, model="hog")
            if rotated_locations:
                rotated_landmarks = face_recognition.face_landmarks(rotated, rotated_locations)
                rotated_score = calculate_upright_score(rotated_landmarks)

                if rotated_score > best_score:
                    best_rotation = rotation
                    best_score = rotated_score

        return best_rotation if best_score > 0.6 else 0

    except Exception:
        return None


def calculate_upright_score(landmarks_list: list) -> float:
    """Calculate how upright faces appear (0-1 scale)."""
    if not landmarks_list:
        return 0.0

    scores = []
    for landmarks in landmarks_list:
        if all(k in landmarks for k in ["left_eye", "right_eye", "nose_tip"]):
            try:
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose_tip = landmarks["nose_tip"]

                # Eye centers
                left_center = (
                    sum(p[0] for p in left_eye) / len(left_eye),
                    sum(p[1] for p in left_eye) / len(left_eye)
                )
                right_center = (
                    sum(p[0] for p in right_eye) / len(right_eye),
                    sum(p[1] for p in right_eye) / len(right_eye)
                )
                nose_center = (
                    sum(p[0] for p in nose_tip) / len(nose_tip),
                    sum(p[1] for p in nose_tip) / len(nose_tip)
                )

                # Eyes should be level
                eye_height_diff = abs(left_center[1] - right_center[1])
                eye_width = abs(left_center[0] - right_center[0])
                eye_level_score = max(0, 1 - (eye_height_diff / eye_width)) if eye_width > 0 else 0

                # Nose should be below eyes
                eye_avg_y = (left_center[1] + right_center[1]) / 2
                nose_below = 1.0 if nose_center[1] > eye_avg_y else 0.0

                scores.append(eye_level_score * 0.6 + nose_below * 0.4)
            except (IndexError, ZeroDivisionError):
                continue

    return sum(scores) / len(scores) if scores else 0.0


class FaceWorker:
    """Windows face detection worker service."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.executor = ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_face_worker)
        self.running = True

    async def connect(self):
        """Connect to PostgreSQL on Vision server."""
        dsn = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        self.pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        logger.info(f"Connected to PostgreSQL at {DB_HOST}")

    async def disconnect(self):
        """Clean shutdown."""
        if self.pool:
            await self.pool.close()
        self.executor.shutdown(wait=True)

    async def get_pending_images(self, batch_size: int) -> List[str]:
        """Get images that need face detection."""
        async with self.pool.acquire() as conn:
            # Find processed images without face metadata
            rows = await conn.fetch("""
                SELECT a.image_path as file_path
                FROM ai_metadata a
                LEFT JOIN face_metadata f ON a.image_path = f.image_path
                WHERE f.image_path IS NULL
                  AND a.is_processed = TRUE
                LIMIT $1
            """, batch_size)
            return [row['file_path'] for row in rows]

    async def save_face_data(self, linux_path: str, data: Dict[str, Any]) -> bool:
        """Save face detection results to database."""
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
                        recognized_persons = EXCLUDED.recognized_persons,
                        orientation_hint = EXCLUDED.orientation_hint,
                        detection_date = CURRENT_TIMESTAMP
                """,
                    linux_path,
                    data.get("face_count", 0),
                    json.dumps(data.get("face_locations", [])),
                    json.dumps(data.get("face_encodings", [])),
                    json.dumps([]),  # recognized_persons populated by clustering
                    data.get("orientation_hint"),
                )
                return True
            except Exception as e:
                logger.error(f"Save error for {linux_path}: {e}")
                return False

    async def process_image(self, linux_path: str) -> Tuple[bool, int]:
        """Process a single image for face detection."""
        windows_path = linux_to_windows_path(linux_path)

        if not os.path.exists(windows_path):
            logger.warning(f"File not found: {windows_path}")
            # Save empty result to avoid retrying
            await self.save_face_data(linux_path, {"face_count": 0})
            return False, 0

        loop = asyncio.get_event_loop()

        try:
            future = loop.run_in_executor(
                self.executor, detect_faces_worker, windows_path
            )
            result = await asyncio.wait_for(future, timeout=DETECTION_TIMEOUT)

            if result.get("success"):
                await self.save_face_data(linux_path, result)
                return True, result.get("face_count", 0)
            else:
                logger.warning(f"Detection failed for {linux_path}: {result.get('error')}")
                # Save empty result
                await self.save_face_data(linux_path, {"face_count": 0})
                return False, 0

        except asyncio.TimeoutError:
            logger.warning(f"Timeout processing {linux_path}")
            await self.save_face_data(linux_path, {"face_count": 0})
            return False, 0
        except Exception as e:
            logger.error(f"Error processing {linux_path}: {e}")
            return False, 0

    async def process_batch(self, linux_paths: List[str]) -> Tuple[int, int]:
        """Process a batch of images."""
        processed = 0
        total_faces = 0

        for path in linux_paths:
            success, faces = await self.process_image(path)
            if success:
                processed += 1
                total_faces += faces

        return processed, total_faces

    async def run(self):
        """Main worker loop."""
        await self.connect()

        try:
            total_processed = 0
            total_faces = 0
            idle_count = 0

            while self.running:
                paths = await self.get_pending_images(BATCH_SIZE)

                if not paths:
                    idle_count += 1
                    if idle_count == 1:
                        logger.info(f"No pending images. Total: {total_processed} images, {total_faces} faces. Waiting...")
                    await asyncio.sleep(30)
                    continue

                idle_count = 0
                logger.info(f"Processing batch of {len(paths)} images...")

                processed, faces = await self.process_batch(paths)
                total_processed += processed
                total_faces += faces

                logger.info(f"Batch: {processed}/{len(paths)}, {faces} faces. Total: {total_processed} images, {total_faces} faces")
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
        finally:
            self.running = False
            await self.disconnect()


async def main():
    # Check for CUDA
    try:
        import dlib
        cuda_status = "ENABLED" if dlib.DLIB_USE_CUDA else "DISABLED"
    except Exception:
        cuda_status = "UNKNOWN"

    logger.info("=" * 60)
    logger.info("Vision Face Detection Worker Starting")
    logger.info(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    logger.info(f"Image mount: {IMAGE_MOUNT}")
    logger.info(f"Workers: {MAX_WORKERS}")
    logger.info(f"CUDA: {cuda_status}")
    logger.info("=" * 60)

    worker = FaceWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())