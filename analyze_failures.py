#!/usr/bin/env python3
"""Analyze permanent failures to understand root causes."""
import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect("postgresql://vision:vision123@192.168.12.20:5432/vision")

    failures = await conn.fetch("""
        SELECT image_path, last_error, last_error_type, retry_count,
               original_width, original_height
        FROM face_detection_failures
        WHERE permanently_failed = TRUE
        ORDER BY last_error_type, image_path
    """)

    print(f"=== {len(failures)} Permanent Failures ===")

    by_type = {}
    for f in failures:
        t = f['last_error_type'] or 'NULL'
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(f)

    for error_type, items in sorted(by_type.items()):
        print(f"\n[{error_type}] - {len(items)} images")
        print("-" * 60)
        for f in items[:3]:
            path = f['image_path']
            if len(path) > 60:
                path = path[:57] + "..."
            w = f['original_width']
            h = f['original_height']
            dims = f"{w}x{h}" if w else "unknown"
            err = (f['last_error'] or '')[:100]
            retries = f['retry_count']
            print(f"  {path}")
            print(f"    dims={dims} retries={retries}")
            print(f"    error: {err}")
        if len(items) > 3:
            print(f"  ... +{len(items) - 3} more")

    await conn.close()

asyncio.run(main())
