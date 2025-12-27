#!/usr/bin/env python3
"""Mark stuck retry images as permanently failed."""
import asyncio
import asyncpg

async def fix():
    dsn = "postgresql://vision:vision123@192.168.12.20:5432/vision"
    conn = await asyncpg.connect(dsn)
    print("Connected")

    # Check what's stuck
    stuck = await conn.fetch("""
        SELECT image_path, retry_count, last_error_type, next_retry_after
        FROM face_detection_failures
        WHERE permanently_failed = FALSE
        ORDER BY retry_count DESC
    """)

    print(f"\nFound {len(stuck)} retriable failures:")
    for row in stuck:
        print(f"  [{row['retry_count']} retries] {row['image_path'][:50]}... ({row['last_error_type']})")

    if not stuck:
        print("Nothing to fix")
        await conn.close()
        return

    # Mark them as permanently failed
    result = await conn.execute("""
        UPDATE face_detection_failures
        SET permanently_failed = TRUE,
            last_error = 'Marked permanent after repeated failures',
            next_retry_after = CURRENT_TIMESTAMP + INTERVAL '100 years'
        WHERE permanently_failed = FALSE
    """)

    print(f"\nMarked {len(stuck)} images as permanently failed")
    await conn.close()

asyncio.run(fix())
