#!/usr/bin/env python3
"""
VisionFaceService Database Setup & Diagnostics
===============================================
Autonomous database management script for setting up failure tracking,
diagnosing issues, and managing the face detection backlog.

Usage:
    python db_setup.py setup      # Create table and indexes
    python db_setup.py diagnose   # Run diagnostic queries
    python db_setup.py blacklist  # Add known bad images to blacklist
    python db_setup.py stats      # Show current statistics
    python db_setup.py reset      # Reset permanently failed for retry
    python db_setup.py all        # Run all setup steps
"""

import asyncio
import sys
from datetime import datetime
from typing import Optional

import asyncpg

# Database configuration (same as face_worker.py)
DB_HOST = "192.168.12.20"
DB_PORT = 5432
DB_NAME = "vision"
DB_USER = "vision"
DB_PASS = "vision123"

# Known problematic images from log analysis
KNOWN_BAD_IMAGES = [
    "attachments/2100/23005.png",
    "attachments/2152/30692.png",
    "attachments/2152/30700.png",
    "attachments/2152/30705.png",
    "attachments/2152/30712.png",
]


async def get_connection() -> asyncpg.Connection:
    """Get a database connection."""
    dsn = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return await asyncpg.connect(dsn)


async def setup_table(conn: asyncpg.Connection) -> bool:
    """Create the face_detection_failures table if it doesn't exist."""
    print("Setting up face_detection_failures table...")

    try:
        # Create table
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
        print("  [OK] Table created/verified")

        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fdf_next_retry
            ON face_detection_failures(next_retry_after)
            WHERE permanently_failed = FALSE
        """)
        print("  [OK] Index idx_fdf_next_retry created")

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fdf_image_path
            ON face_detection_failures(image_path)
        """)
        print("  [OK] Index idx_fdf_image_path created")

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fdf_error_type
            ON face_detection_failures(last_error_type)
        """)
        print("  [OK] Index idx_fdf_error_type created")

        return True

    except asyncpg.exceptions.InsufficientPrivilegeError as e:
        print(f"  [ERROR] Insufficient privileges: {e}")
        print("  Run as database superuser or grant CREATE permission to vision user")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to create table: {e}")
        return False


async def diagnose(conn: asyncpg.Connection):
    """Run diagnostic queries to understand current state."""
    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT")
    print("="*60)

    # Check if table exists
    table_exists = await conn.fetchval("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'face_detection_failures'
        )
    """)
    print(f"\nface_detection_failures table exists: {table_exists}")

    if not table_exists:
        print("  [!] Table does not exist - run 'python db_setup.py setup' first")
        return

    # Overall counts
    print("\n--- Overall Status ---")
    processed = await conn.fetchval("SELECT COUNT(*) FROM face_metadata")
    ai_processed = await conn.fetchval("SELECT COUNT(*) FROM ai_metadata WHERE is_processed = TRUE")
    failures = await conn.fetchval("SELECT COUNT(*) FROM face_detection_failures")
    permanent = await conn.fetchval("SELECT COUNT(*) FROM face_detection_failures WHERE permanently_failed = TRUE")
    retriable = await conn.fetchval("SELECT COUNT(*) FROM face_detection_failures WHERE permanently_failed = FALSE")

    print(f"  AI-processed images:     {ai_processed:,}")
    print(f"  Face metadata entries:   {processed:,}")
    print(f"  Failure records:         {failures:,}")
    print(f"    - Permanently failed:  {permanent:,}")
    print(f"    - Retriable:           {retriable:,}")

    pending = ai_processed - processed - failures
    print(f"  Pending (no status):     {pending:,}")

    # Error type breakdown
    print("\n--- Failure Breakdown by Error Type ---")
    rows = await conn.fetch("""
        SELECT
            last_error_type,
            COUNT(*) as count,
            COUNT(*) FILTER (WHERE permanently_failed) as permanent,
            AVG(retry_count)::numeric(10,1) as avg_retries
        FROM face_detection_failures
        GROUP BY last_error_type
        ORDER BY count DESC
    """)

    if rows:
        print(f"  {'Error Type':<20} {'Count':>10} {'Permanent':>10} {'Avg Retries':>12}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12}")
        for row in rows:
            etype = row['last_error_type'] or 'NULL'
            print(f"  {etype:<20} {row['count']:>10} {row['permanent']:>10} {row['avg_retries'] or 0:>12.1f}")
    else:
        print("  No failures recorded")

    # Images stuck in pending (not in face_metadata, not in failures)
    print("\n--- Pending Images Sample (not in face_metadata or failures) ---")
    stuck = await conn.fetch("""
        SELECT a.image_path
        FROM ai_metadata a
        LEFT JOIN face_metadata f ON a.image_path = f.image_path
        LEFT JOIN face_detection_failures ff ON a.image_path = ff.image_path
        WHERE a.is_processed = TRUE
          AND f.image_path IS NULL
          AND ff.image_path IS NULL
        LIMIT 10
    """)

    if stuck:
        for row in stuck:
            print(f"  - {row['image_path']}")
        if len(stuck) == 10:
            print("  ... (showing first 10)")
    else:
        print("  None - all images have a status")

    # Most recent failures
    print("\n--- Most Recent Failures ---")
    recent = await conn.fetch("""
        SELECT image_path, last_error_type, retry_count, last_failure_at
        FROM face_detection_failures
        ORDER BY last_failure_at DESC
        LIMIT 5
    """)

    if recent:
        for row in recent:
            ts = row['last_failure_at'].strftime('%Y-%m-%d %H:%M') if row['last_failure_at'] else 'N/A'
            print(f"  [{row['last_error_type']}] {row['image_path'][:50]} (retries: {row['retry_count']}, at: {ts})")
    else:
        print("  No failures recorded")

    # Check for known bad images
    print("\n--- Known Bad Images Status ---")
    for img in KNOWN_BAD_IMAGES:
        status = await conn.fetchrow("""
            SELECT retry_count, permanently_failed, last_error_type
            FROM face_detection_failures
            WHERE image_path = $1
        """, img)

        if status:
            perm = "PERMANENT" if status['permanently_failed'] else "retriable"
            print(f"  {img}: {status['last_error_type']} ({perm}, {status['retry_count']} retries)")
        else:
            # Check if in face_metadata
            has_meta = await conn.fetchval(
                "SELECT 1 FROM face_metadata WHERE image_path = $1", img
            )
            if has_meta:
                print(f"  {img}: IN face_metadata (processed)")
            else:
                print(f"  {img}: NOT TRACKED (will retry infinitely!)")


async def blacklist_bad_images(conn: asyncpg.Connection):
    """Add known problematic images to the blacklist."""
    print("\nBlacklisting known problematic images...")

    for img in KNOWN_BAD_IMAGES:
        try:
            result = await conn.execute("""
                INSERT INTO face_detection_failures
                    (image_path, last_error, last_error_type, permanently_failed, retry_count,
                     first_failure_at, last_failure_at, next_retry_after)
                VALUES ($1, $2, $3, TRUE, 99, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP + INTERVAL '100 years')
                ON CONFLICT (image_path) DO UPDATE SET
                    permanently_failed = TRUE,
                    retry_count = 99,
                    last_error = EXCLUDED.last_error,
                    next_retry_after = CURRENT_TIMESTAMP + INTERVAL '100 years'
            """, img, "Blacklisted - known problematic image causing NoneType errors", "blacklisted")
            print(f"  [OK] Blacklisted: {img}")
        except Exception as e:
            print(f"  [ERROR] Failed to blacklist {img}: {e}")


async def show_stats(conn: asyncpg.Connection):
    """Show current processing statistics."""
    print("\n" + "="*60)
    print("PROCESSING STATISTICS")
    print("="*60)

    # Face metadata stats
    face_stats = await conn.fetchrow("""
        SELECT
            COUNT(*) as total,
            SUM(face_count) as total_faces,
            AVG(face_count)::numeric(10,2) as avg_faces,
            MAX(detection_date) as last_detection
        FROM face_metadata
    """)

    print(f"\nFace Detection Results:")
    print(f"  Total images processed:  {face_stats['total']:,}")
    print(f"  Total faces found:       {face_stats['total_faces'] or 0:,}")
    print(f"  Average faces/image:     {face_stats['avg_faces'] or 0:.2f}")
    if face_stats['last_detection']:
        print(f"  Last detection:          {face_stats['last_detection']}")

    # Failure stats
    fail_stats = await conn.fetchrow("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE permanently_failed) as permanent,
            AVG(retry_count)::numeric(10,1) as avg_retries
        FROM face_detection_failures
    """)

    print(f"\nFailure Tracking:")
    print(f"  Total failures:          {fail_stats['total']:,}")
    print(f"  Permanently failed:      {fail_stats['permanent']:,}")
    print(f"  Average retries:         {fail_stats['avg_retries'] or 0:.1f}")

    # Pending queue
    pending = await conn.fetchval("""
        SELECT COUNT(*)
        FROM ai_metadata a
        LEFT JOIN face_metadata f ON a.image_path = f.image_path
        LEFT JOIN face_detection_failures ff ON a.image_path = ff.image_path
        WHERE a.is_processed = TRUE
          AND f.image_path IS NULL
          AND (ff.image_path IS NULL
               OR (ff.permanently_failed = FALSE AND ff.next_retry_after <= CURRENT_TIMESTAMP))
    """)

    print(f"\nPending Queue:")
    print(f"  Ready to process:        {pending:,}")


async def reset_permanent_failures(conn: asyncpg.Connection):
    """Reset permanently failed images for retry (useful after fixing bugs)."""
    print("\nResetting permanently failed images for retry...")

    # Get count first
    count = await conn.fetchval("""
        SELECT COUNT(*) FROM face_detection_failures
        WHERE permanently_failed = TRUE
          AND last_error_type NOT IN ('blacklisted', 'file_error')
    """)

    if count == 0:
        print("  No non-blacklisted permanent failures to reset")
        return

    confirm = input(f"  Reset {count} permanent failures? (y/n): ")
    if confirm.lower() != 'y':
        print("  Cancelled")
        return

    result = await conn.execute("""
        UPDATE face_detection_failures
        SET
            permanently_failed = FALSE,
            retry_count = 0,
            next_retry_after = CURRENT_TIMESTAMP
        WHERE permanently_failed = TRUE
          AND last_error_type NOT IN ('blacklisted', 'file_error')
    """)

    print(f"  [OK] Reset {count} failures for retry")


async def run_all(conn: asyncpg.Connection):
    """Run all setup steps."""
    await setup_table(conn)
    await diagnose(conn)
    await blacklist_bad_images(conn)
    await show_stats(conn)


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    print(f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME}...")

    try:
        conn = await get_connection()
        print("[OK] Connected\n")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        sys.exit(1)

    try:
        if command == "setup":
            await setup_table(conn)
        elif command == "diagnose":
            await diagnose(conn)
        elif command == "blacklist":
            await blacklist_bad_images(conn)
        elif command == "stats":
            await show_stats(conn)
        elif command == "reset":
            await reset_permanent_failures(conn)
        elif command == "all":
            await run_all(conn)
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            sys.exit(1)
    finally:
        await conn.close()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
