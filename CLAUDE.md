# CLAUDE.md - VisionFaceService

## Project Overview

**VisionFaceService** - A robust Windows service for GPU-accelerated face detection, designed to work with the Vision AI system on Linux. This worker offloads face detection from the Vision server to a Windows machine with an NVIDIA GPU.

## Architecture

```
Vision Server (Linux @ 192.168.12.20)
├── PostgreSQL Database (vision)
│   ├── images - All registered images
│   ├── ai_metadata - Processing status
│   ├── face_metadata - Successful face detections
│   └── face_detection_failures - Failed detections for retry
└── SMB Share: \\192.168.12.20\vision_images → /mnt/icculus/pictures

GAME Worker (Windows @ 192.168.12.75)
├── C:\VisionFaceService\
│   ├── face_worker.py - Main worker script
│   ├── CLAUDE.md - This file
│   └── worker.log - Runtime logs
└── GPU: NVIDIA GTX 1650 (4GB VRAM)
```

## Database Schema

### Table: images
```sql
CREATE TABLE images (
    image_path TEXT PRIMARY KEY,  -- Relative path like 'apple/IMG_1234.JPG'
    file_size BIGINT,
    last_modified REAL,
    updated_at TIMESTAMP
);
```

### Table: ai_metadata
```sql
CREATE TABLE ai_metadata (
    image_path TEXT PRIMARY KEY,
    is_processed BOOLEAN DEFAULT FALSE,
    -- ... other AI processing fields
);
```

### Table: face_metadata
```sql
CREATE TABLE face_metadata (
    image_path TEXT PRIMARY KEY,
    face_count INTEGER DEFAULT 0,
    face_locations TEXT,      -- JSON array of [top, right, bottom, left]
    face_encodings TEXT,      -- JSON array of 128-d vectors
    recognized_persons TEXT,  -- JSON array of person names
    orientation_hint INTEGER,
    detection_date TIMESTAMP
);
```

### Table: face_detection_failures (NEW)
```sql
CREATE TABLE face_detection_failures (
    id SERIAL PRIMARY KEY,
    image_path TEXT NOT NULL UNIQUE,
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,
    last_error_type TEXT,     -- 'oom', 'timeout', 'crash', 'unknown'
    original_width INTEGER,
    original_height INTEGER,
    last_attempted_scale REAL,
    first_failure_at TIMESTAMP,
    last_failure_at TIMESTAMP,
    next_retry_after TIMESTAMP,
    permanently_failed BOOLEAN DEFAULT FALSE
);
```

## Key Concepts

### Path Handling
- Database stores **relative paths**: `apple/IMG_1234.JPG`
- Windows accesses via UNC: `\\192.168.12.20\vision_images\apple\IMG_1234.JPG`
- Never use mapped drives (unreliable for services)

### Face Detection Flow
1. Query `ai_metadata` for processed images without `face_metadata` entry
2. Also query `face_detection_failures` for retriable images
3. Load image, downscale if needed, detect faces
4. On SUCCESS: Save to `face_metadata`, delete from `face_detection_failures`
5. On FAILURE: Record in `face_detection_failures` with retry delay

### GPU Memory Management (GTX 1650 4GB)
- **Critical constraint**: Only 4GB VRAM
- Pre-scale images to max 1280px dimension
- Use progressive cascade: 100% → 75% → 50% → 33% → 25%
- Fall back to CPU HOG detector if CNN OOMs
- Cleanup memory between images: `gc.collect()`, `torch.cuda.empty_cache()`

## Configuration Reference

```python
# Database
DB_HOST = "192.168.12.20"
DB_PORT = 5432
DB_NAME = "vision"
DB_USER = "vision"
DB_PASS = "vision123"

# Image Access
IMAGE_MOUNT = r"\\192.168.12.20\vision_images"

# Conservative GPU Settings
MAX_IMAGE_DIMENSION = 1280
SCALE_CASCADE = [1.0, 0.75, 0.5, 0.33, 0.25]
MAX_WORKERS = 1
BATCH_SIZE = 5
DETECTION_TIMEOUT = 120

# Retry Settings
MAX_RETRY_COUNT = 5
BASE_RETRY_DELAY = 60
RETRY_BACKOFF_MULTIPLIER = 2
```

## Dependencies

```
asyncpg
numpy
Pillow
pillow-heif
face_recognition
dlib (with CUDA)
```

## Reference: Vision Server faces.py Patterns

The Vision server's face processor uses these patterns:

```python
# Process isolation for crash protection
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

_face_executor = ProcessPoolExecutor(
    max_workers=1,
    mp_context=multiprocessing.get_context("spawn")
)

# HEIC handling
def _load_image(image_path: str) -> np.ndarray:
    if image_path.lower().endswith(('.heic', '.heif')):
        from pillow_heif import register_heif_opener
        register_heif_opener()

    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

# Face detection with face_recognition
import face_recognition

def detect_faces(image_array):
    face_locations = face_recognition.face_locations(image_array, model="hog")
    face_encodings = face_recognition.face_encodings(image_array, face_locations)
    return face_locations, face_encodings
```

## Service Management

```powershell
# Run manually
cd C:\VisionFaceService
.\venv\Scripts\Activate.ps1
python face_worker.py

# View logs
Get-Content C:\VisionFaceService\worker.log -Tail 50 -Wait

# Install as Windows service with NSSM
C:\nssm.exe install VisionFaceWorker
# Path: C:\VisionFaceService\venv\Scripts\python.exe
# Startup directory: C:\VisionFaceService
# Arguments: face_worker.py
```

## Monitoring

```sql
-- Check progress
SELECT COUNT(*) as detected FROM face_metadata;
SELECT COUNT(*) as pending_retry FROM face_detection_failures WHERE permanently_failed = FALSE;
SELECT COUNT(*) as permanent_failures FROM face_detection_failures WHERE permanently_failed = TRUE;

-- Recent activity
SELECT image_path, detection_date FROM face_metadata ORDER BY detection_date DESC LIMIT 10;

-- Failure breakdown
SELECT last_error_type, COUNT(*) FROM face_detection_failures GROUP BY last_error_type;
```
