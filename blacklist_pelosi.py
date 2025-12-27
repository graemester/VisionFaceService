#!/usr/bin/env python3
"""Blacklist the Nancy Pelosi image that's causing infinite loops."""
import asyncio
import asyncpg

async def blacklist():
    dsn = "postgresql://vision:vision123@192.168.12.20:5432/vision"
    conn = await asyncpg.connect(dsn)
    print("Connected")

    # Find and blacklist any image with "Nancy Pelosi" or multiple periods
    result = await conn.fetch("""
        SELECT image_path FROM ai_metadata
        WHERE image_path LIKE '%Nancy Pelosi%'
           OR image_path LIKE '%Arthur Delaney%'
           OR image_path LIKE '%....%'
        LIMIT 20
    """)

    print(f"Found {len(result)} images to blacklist:")
    for row in result:
        print(f"  - {row['image_path'][:80]}...")

        await conn.execute("""
            INSERT INTO face_detection_failures
                (image_path, last_error, last_error_type, permanently_failed,
                 retry_count, next_retry_after)
            VALUES ($1, $2, 'path_issue', TRUE, 99,
                    CURRENT_TIMESTAMP + INTERVAL '100 years')
            ON CONFLICT (image_path) DO UPDATE SET
                permanently_failed = TRUE,
                last_error = EXCLUDED.last_error,
                last_error_type = 'path_issue',
                next_retry_after = CURRENT_TIMESTAMP + INTERVAL '100 years'
        """, row['image_path'], "Blacklisted: filename with special characters causing path issues")

    print(f"\nBlacklisted {len(result)} images")
    await conn.close()

asyncio.run(blacklist())
