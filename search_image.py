#!/usr/bin/env python3
"""Search for images with problematic patterns."""
import asyncio
import asyncpg

async def search():
    dsn = "postgresql://vision:vision123@192.168.12.20:5432/vision"
    conn = await asyncpg.connect(dsn)

    # Search ai_metadata for pattern
    print("Searching ai_metadata for images with '...' or special chars...\n")

    # Exact search for the filename pattern
    result = await conn.fetch("""
        SELECT image_path
        FROM ai_metadata
        WHERE image_path LIKE '%on Twitter%'
           OR image_path LIKE '%Delaney%'
           OR image_path LIKE '%Pelosi%'
        LIMIT 20
    """)

    if result:
        print(f"Found {len(result)} matching images in ai_metadata:")
        for row in result:
            path = row['image_path']
            print(f"  - {path}")

            # Check if in face_metadata (processed)
            fm = await conn.fetchval("SELECT 1 FROM face_metadata WHERE image_path = $1", path)
            # Check if in failures
            ff = await conn.fetchrow("""
                SELECT permanently_failed, retry_count
                FROM face_detection_failures WHERE image_path = $1
            """, path)

            if fm:
                print(f"    -> In face_metadata (PROCESSED)")
            elif ff:
                print(f"    -> In failures: perm={ff['permanently_failed']}, retries={ff['retry_count']}")
            else:
                print(f"    -> NOT TRACKED (will be picked up for processing)")
    else:
        print("No matching images found in ai_metadata")

    # Also check images table
    result2 = await conn.fetch("""
        SELECT image_path
        FROM images
        WHERE image_path LIKE '%Pelosi%' OR image_path LIKE '%Delaney%'
        LIMIT 5
    """)

    if result2:
        print(f"\nFound {len(result2)} in images table:")
        for row in result2:
            print(f"  - {row['image_path']}")

    await conn.close()

asyncio.run(search())
