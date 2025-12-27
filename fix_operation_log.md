# VisionFaceService Fix Operation Log

**Date**: 2025-12-26
**Operator**: Claude Code

## Problem Summary

The face detection worker has been failing for hours with 120,500+ failures. Root causes:
1. Missing `face_detection_failures` table
2. NoneType subscript error on 5 specific PNG files
3. Infinite retry loop without failure tracking

## Operations Performed

### 1. Service Stop (11:01)

**Action**: Stopped VisionFaceWorker Windows service
```
net stop VisionFaceWorker
```
**Result**: Service stopped successfully

**Note**: Service is managed by NSSM at `C:\nssm.exe`

### 2. Log Archive (11:03)

**Action**: Archived 43MB spam log
```
cp worker.log worker.log.pre-fix-backup
echo > worker.log
```
**Result**: Log archived, fresh log started

### 3. Database Setup Attempt (11:05)

**Action**: Created and ran `db_setup.py`

**Findings**:
- Table `face_detection_failures` either doesn't exist or vision user can't see it
- `vision` user has insufficient privileges to create indexes

**Required Actions on Linux Vision Server (192.168.12.20)**:
Run as PostgreSQL superuser (postgres):
```sql
-- Connect to vision database
\c vision

-- Create table if needed
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
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_fdf_next_retry ON face_detection_failures(next_retry_after) WHERE permanently_failed = FALSE;
CREATE INDEX IF NOT EXISTS idx_fdf_image_path ON face_detection_failures(image_path);
CREATE INDEX IF NOT EXISTS idx_fdf_error_type ON face_detection_failures(last_error_type);

-- Grant permissions
GRANT ALL ON face_detection_failures TO vision;
GRANT USAGE, SELECT ON SEQUENCE face_detection_failures_id_seq TO vision;
```

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `db_setup.py` | Created | Autonomous database setup and diagnostics |
| `worker.log` | Cleared | Fresh log for debugging |
| `worker.log.pre-fix-backup` | Created | Archived 43MB of failure spam |
| `fix_operation_log.md` | Created | This operation log |

## BLOCKER: Database Permissions Required

The `face_detection_failures` table EXISTS but `vision` user doesn't have permission.

**Required Action on Linux Vision Server (192.168.12.20)**:
```sql
-- Run as postgres superuser
GRANT ALL ON face_detection_failures TO vision;
GRANT USAGE, SELECT ON SEQUENCE face_detection_failures_id_seq TO vision;
```

### 4. face_worker.py Hardening (11:15-11:30)

**Changes Made**:
1. Added **CircuitBreaker class** - Tracks failure rates, pauses processing at 80% failure threshold
2. Added **SessionBlacklist class** - In-memory blacklist for images failing repeatedly
3. Added **ExecutorManager class** - Auto-restarts ProcessPoolExecutor after consecutive crashes
4. Updated `connect()` - Attempts to create failure table, graceful degradation if no permissions
5. Updated `process_batch()` - Defensive integration of all auto-healing components
6. Added `write_health_status()` - Writes JSON status file for external monitoring
7. All new code wrapped in `try/except` with `hasattr()` checks for safety

### 5. Test Run (11:31)

**Result**: Worker starts successfully and operates in degraded mode

**Observations**:
- Connected to PostgreSQL ✓
- GPU detected (1960MiB free) ✓
- CUDA enabled ✓
- Failure table exists but vision user has NO PERMISSION ✗
- Processing batches, finding many missing files
- Executor auto-restart triggered after 5 failures ✓

**Log Sample**:
```
2025-12-26 11:31:10 - ERROR - No access to face_detection_failures table -
    failures will be logged but not tracked for retry.
    Run: GRANT ALL ON face_detection_failures TO vision;
2025-12-26 11:32:39 - WARNING - Executor has crashed 5 times - will restart
2025-12-26 11:32:39 - INFO - Created new ProcessPoolExecutor (restart #2)
```

## BLOCKER: Database Permissions Required

The `face_detection_failures` table EXISTS but `vision` user doesn't have permission.

**Required Action on Linux Vision Server (192.168.12.20)**:
```sql
-- Run as postgres superuser
GRANT ALL ON face_detection_failures TO vision;
GRANT USAGE, SELECT ON SEQUENCE face_detection_failures_id_seq TO vision;
```

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `face_worker.py` | Modified | Added circuit breaker, session blacklist, executor recovery, defensive checks |
| `face_worker_old.py` | Exists | Original backup (already existed) |
| `db_setup.py` | Created | Autonomous database setup and diagnostics |
| `test_db.py` | Created | Quick database access test |
| `investigate_images.py` | Created | Script to investigate failing PNG files |
| `worker.log` | Cleared | Fresh log for debugging |
| `worker.log.pre-fix-backup` | Created | Archived 43MB of failure spam |
| `fix_operation_log.md` | Created | This operation log |

## New Features Added to face_worker.py

1. **CircuitBreaker**: Pauses processing when failure rate exceeds 80% for 5 minutes
2. **SessionBlacklist**: Temporarily blacklists images that fail 3+ times in same session
3. **ExecutorManager**: Auto-restarts executor after 5 consecutive crashes
4. **Defensive coding**: All new code wrapped in try/except, falls through to existing behavior
5. **Health status file**: `health_status.json` written for external monitoring

## Remaining Issues

### Issue 1: Database Permissions (BLOCKER)
**Status**: BLOCKED - Requires Linux server admin
**Action Required**: Run `GRANT ALL ON face_detection_failures TO vision;` on Linux server

### Issue 2: Many Missing Files
**Status**: Investigation needed on Linux server
**Finding**: Many files in `ai_metadata` no longer exist on disk:
- `apple/IMG_0073(1).PNG`
- `apple/IMG_0157(2).PNG`
- `apple/IMG_0528(3).PNG`
- etc.

**Recommendation**: Run cleanup script on Linux server to remove orphaned ai_metadata entries

### Issue 3: Known Bad PNG Files
**Status**: Could not investigate (SMB share access issues from Windows)
**Files**:
- `attachments/2100/23005.png`
- `attachments/2152/30692.png`
- `attachments/2152/30700.png`
- `attachments/2152/30705.png`
- `attachments/2152/30712.png`

**Recommendation**: After permissions are fixed, run `python db_setup.py blacklist` to permanently mark these

### 6. Database Permissions Fixed (12:38)

**Action**: User granted permissions on Linux server
**Result**: `test_db.py` confirmed table accessible with 0 rows

### 7. Blacklisted Known Bad Images (12:38)

**Action**: Ran `db_setup.py blacklist`
**Result**: 5 known problematic images marked as permanently failed
```
  [OK] Blacklisted: attachments/2100/23005.png
  [OK] Blacklisted: attachments/2152/30692.png
  [OK] Blacklisted: attachments/2152/30700.png
  [OK] Blacklisted: attachments/2152/30705.png
  [OK] Blacklisted: attachments/2152/30712.png
```

### 8. First Start - Schema Mismatch Found (12:38)

**Issue Found**: Production table was missing `file_size_bytes` column
**Error**: `column "file_size_bytes" of relation "face_detection_failures" does not exist`
**Root Cause**: Table created by mainline app with different schema

**Fixes Applied**:
1. Removed `file_size_bytes` from `record_failure()` SQL query
2. Fixed `error[:100]` crash when error is None (added null check)

### 9. Service Restart and Validation (12:40)

**Action**: Restarted VisionFaceWorker service
**Result**: SUCCESS - Worker now processing correctly

**Final Statistics** (12:41):
```
Face Detection Results:
  Total images processed:  83,465
  Total faces found:       27,204
  Average faces/image:     0.33
  Last detection:          2025-12-26 12:41:19

Failure Tracking:
  Total failures:          18
  Permanently failed:      5
  Average retries:         28.2

Pending Queue:
  Ready to process:        2,104
```

---

## RESOLUTION COMPLETE

### Summary of Changes

| Component | Change |
|-----------|--------|
| `face_worker.py` | Added CircuitBreaker, SessionBlacklist, ExecutorManager classes |
| `face_worker.py` | Fixed None error handling, removed file_size_bytes reference |
| Database | Permissions granted to vision user |
| Database | 5 bad images permanently blacklisted |

### Auto-Healing Features Added

1. **CircuitBreaker**: Pauses processing when failure rate exceeds 80% for 5 minutes
2. **SessionBlacklist**: Temporarily blacklists images that fail 3+ times in same session
3. **ExecutorManager**: Auto-restarts executor after 5 consecutive crashes
4. **Defensive coding**: All new code wrapped in try/except, falls through to existing behavior

### Verification Status

| Feature | Status |
|---------|--------|
| Failure tracking | WORKING (18 failures recorded with retry scheduling) |
| Face detection | WORKING (finding faces at various scales) |
| Blacklist | WORKING (5 permanently failed images excluded) |
| Auto-healing | WORKING (executor restart triggered on crashes) |
| Circuit breaker | READY (will trigger at 80% failure rate) |

### Monitoring Commands

```powershell
# Live logs
Get-Content C:\VisionFaceService\worker.log -Tail 50 -Wait

# Database stats
cd C:\VisionFaceService
.\venv\Scripts\python.exe db_setup.py stats

# Full diagnostic report
.\venv\Scripts\python.exe db_setup.py diagnose

# Service control
net stop VisionFaceWorker
net start VisionFaceWorker
```

### Files Created During Fix

| File | Purpose |
|------|---------|
| `db_setup.py` | Database setup, diagnostics, blacklisting |
| `test_db.py` | Quick database access test |
| `fix_schema.py` | Schema migration helper |
| `investigate_images.py` | Image investigation script |
| `fix_operation_log.md` | This operation log |
| `worker.log.pre-fix-backup` | Archived 43MB spam log |
