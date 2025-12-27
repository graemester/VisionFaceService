#!/usr/bin/env python3
"""Test share access."""
import os

shares = [
    r"\\192.168.12.20\icculus",
    r"\\192.168.12.20\icculus\pictures",
    r"\\192.168.12.20\vision_images",
]

for share in shares:
    print(f"Testing: {share}")
    try:
        items = os.listdir(share)[:5]
        print(f"  OK - contents: {items}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
