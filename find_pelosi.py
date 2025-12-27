#!/usr/bin/env python3
"""Find the Nancy Pelosi image."""
import asyncio
import asyncpg

async def find():
    dsn = "postgresql://vision:vision123@192.168.12.20:5432/vision"
    conn = await asyncpg.connect(dsn)

    # Search for any image with periods in name (not extension)
    result = await conn.fetch("""
        SELECT image_path FROM ai_metadata
        WHERE image_path ~ '\\.\\.\\.'
        LIMIT 20
    """)

    print(f"Found {len(result)} images with '...':")
    for row in result:
        print(f"  - {row['image_path']}")

    # Also check failure table
    failures = await conn.fetch("""
        SELECT image_path, last_error, retry_count FROM face_detection_failures
        WHERE last_error IS NULL OR last_error_type IS NULL
        ORDER BY retry_count DESC
        LIMIT 10
    """)

    print(f"\nImages with NULL errors ({len(failures)}):")
    for row in failures:
        print(f"  [{row['retry_count']}] {row['image_path'][:60]}...")

    await conn.close()

asyncio.run(find())
