#!/usr/bin/env python3
"""Check the 1x1 pixel images to understand what they are."""
import asyncio
import asyncpg
import os
from PIL import Image

IMAGE_MOUNT = r"\\192.168.12.20\vision_images"

async def main():
    conn = await asyncpg.connect("postgresql://vision:vision123@192.168.12.20:5432/vision")

    # Get the 1x1 images
    tiny = await conn.fetch("""
        SELECT image_path FROM face_detection_failures
        WHERE original_width = 1 AND original_height = 1
    """)

    print(f"Checking {len(tiny)} tiny (1x1) images:\n")

    for row in tiny[:8]:
        path = row['image_path']
        win_path = os.path.join(IMAGE_MOUNT, path.replace("/", "\\"))

        print(f"{path}")
        try:
            # File size
            size = os.path.getsize(win_path)
            print(f"  file size: {size} bytes")

            # Actual image dimensions
            with Image.open(win_path) as img:
                print(f"  actual dims: {img.size}")
                print(f"  format: {img.format}, mode: {img.mode}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    await conn.close()

asyncio.run(main())
