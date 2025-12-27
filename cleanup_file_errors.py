#!/usr/bin/env python3
"""
One-time script to mark all file_error failures as permanently failed.
This clears the backlog of missing files that were being retried.
"""
import asyncio
import asyncpg

# Database configuration (from face_worker.py)
DB_HOST = "192.168.12.20"
DB_PORT = 5432
DB_NAME = "vision"
DB_USER = "vision"
DB_PASS = "vision123"

async def main():
    print(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}...")

    conn = await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

    try:
        # Check current state
        before = await conn.fetchval("""
            SELECT COUNT(*) FROM face_detection_failures
            WHERE last_error_type = 'file_error' AND permanently_failed = FALSE
        """)
        print(f"Found {before} file_error failures to mark as permanent")

        if before > 0:
            # Mark all file_error failures as permanently failed
            result = await conn.execute("""
                UPDATE face_detection_failures
                SET permanently_failed = TRUE,
                    next_retry_after = CURRENT_TIMESTAMP + INTERVAL '100 years',
                    last_error = 'File does not exist on disk (cleaned up)'
                WHERE last_error_type = 'file_error'
                  AND permanently_failed = FALSE
            """)
            print(f"Updated: {result}")

        # Also mark any with 'File not found' in error message
        before2 = await conn.fetchval("""
            SELECT COUNT(*) FROM face_detection_failures
            WHERE last_error LIKE '%File not found%' AND permanently_failed = FALSE
        """)
        if before2 > 0:
            print(f"Found {before2} additional 'File not found' errors to mark")
            await conn.execute("""
                UPDATE face_detection_failures
                SET permanently_failed = TRUE,
                    next_retry_after = CURRENT_TIMESTAMP + INTERVAL '100 years'
                WHERE last_error LIKE '%File not found%'
                  AND permanently_failed = FALSE
            """)

        # Show final stats
        total_permanent = await conn.fetchval("""
            SELECT COUNT(*) FROM face_detection_failures WHERE permanently_failed = TRUE
        """)
        total_pending = await conn.fetchval("""
            SELECT COUNT(*) FROM face_detection_failures WHERE permanently_failed = FALSE
        """)

        print(f"\nFinal state:")
        print(f"  Permanently failed: {total_permanent}")
        print(f"  Pending retry: {total_pending}")

        # Show breakdown by error type
        print("\nBreakdown by error type (permanently failed):")
        rows = await conn.fetch("""
            SELECT last_error_type, COUNT(*) as cnt
            FROM face_detection_failures
            WHERE permanently_failed = TRUE
            GROUP BY last_error_type
            ORDER BY cnt DESC
        """)
        for row in rows:
            print(f"  {row['last_error_type']}: {row['cnt']}")

    finally:
        await conn.close()
        print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
