#!/usr/bin/env python3
"""Investigate the failing PNG images."""

import os
import sys
from PIL import Image
import numpy as np

# UNC path to SMB share
IMAGE_MOUNT = r"\\192.168.12.20\vision_images"

# Images that are failing with NoneType error
BAD_IMAGES = [
    "attachments/2100/23005.png",
    "attachments/2152/30692.png",
    "attachments/2152/30700.png",
    "attachments/2152/30705.png",
    "attachments/2152/30712.png",
]


def investigate_image(rel_path: str):
    """Check if an image exists and examine its properties."""
    # Build Windows UNC path
    win_path = rel_path.replace("/", "\\")
    full_path = os.path.join(IMAGE_MOUNT, win_path)

    print(f"\n{'='*60}")
    print(f"Image: {rel_path}")
    print(f"Full path: {full_path}")
    print(f"{'='*60}")

    # Check existence
    exists = os.path.exists(full_path)
    print(f"File exists: {exists}")

    if not exists:
        print(">>> FILE NOT FOUND - This is why it's failing!")
        print("    The file was likely deleted or moved, but still in ai_metadata")
        return {"status": "not_found", "path": rel_path}

    # Get file stats
    try:
        stat = os.stat(full_path)
        print(f"File size: {stat.st_size:,} bytes")

        if stat.st_size == 0:
            print(">>> ZERO-SIZE FILE - This would cause errors!")
            return {"status": "zero_size", "path": rel_path}
    except Exception as e:
        print(f"Stat error: {e}")
        return {"status": "stat_error", "path": rel_path, "error": str(e)}

    # Try to open with PIL
    try:
        with Image.open(full_path) as img:
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode}")
            print(f"Size: {img.size[0]}x{img.size[1]}")

            # Check for unusual modes
            if img.mode not in ('RGB', 'RGBA', 'L', 'P'):
                print(f">>> UNUSUAL MODE: {img.mode}")

            # Try RGB conversion
            try:
                rgb = img.convert('RGB')
                arr = np.array(rgb)
                print(f"RGB array shape: {arr.shape}")
                print(f"RGB dtype: {arr.dtype}")

                # Check for valid dimensions
                if arr.shape[0] == 0 or arr.shape[1] == 0:
                    print(">>> ZERO DIMENSION - This would cause NoneType errors!")
                    return {"status": "zero_dim", "path": rel_path}

            except Exception as e:
                print(f">>> RGB CONVERSION FAILED: {e}")
                return {"status": "rgb_error", "path": rel_path, "error": str(e)}

    except Exception as e:
        print(f">>> PIL OPEN FAILED: {type(e).__name__}: {e}")
        return {"status": "open_error", "path": rel_path, "error": str(e)}

    return {"status": "ok", "path": rel_path}


def main():
    print("Investigating failing PNG images...")
    print(f"SMB mount: {IMAGE_MOUNT}")

    # First check if mount is accessible
    if not os.path.exists(IMAGE_MOUNT):
        print(f"\nERROR: Cannot access {IMAGE_MOUNT}")
        print("Make sure the SMB share is accessible.")
        sys.exit(1)

    print(f"Mount accessible: {os.path.exists(IMAGE_MOUNT)}")

    results = []
    for img_path in BAD_IMAGES:
        result = investigate_image(img_path)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    not_found = [r for r in results if r["status"] == "not_found"]
    errors = [r for r in results if r["status"] not in ("ok", "not_found")]
    ok = [r for r in results if r["status"] == "ok"]

    print(f"Files not found: {len(not_found)}")
    for r in not_found:
        print(f"  - {r['path']}")

    print(f"Files with errors: {len(errors)}")
    for r in errors:
        print(f"  - {r['path']}: {r['status']}")

    print(f"Files OK: {len(ok)}")

    if not_found:
        print("\n>>> ROOT CAUSE: Files in database don't exist on disk!")
        print("    These need to be removed from ai_metadata or marked as failures")


if __name__ == "__main__":
    main()
