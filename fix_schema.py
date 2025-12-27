#!/usr/bin/env python3
"""Fix the face_detection_failures table schema."""
import asyncio
import asyncpg

async def fix_schema():
    dsn = "postgresql://vision:vision123@192.168.12.20:5432/vision"
    conn = await asyncpg.connect(dsn)
    print("Connected to database")

    # Check current columns
    columns = await conn.fetch("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'face_detection_failures'
        ORDER BY ordinal_position
    """)

    print("\nCurrent columns:")
    existing = set()
    for col in columns:
        print(f"  - {col['column_name']}: {col['data_type']}")
        existing.add(col['column_name'])

    # Required columns that might be missing
    required_columns = {
        'file_size_bytes': 'BIGINT',
        'original_width': 'INTEGER',
        'original_height': 'INTEGER',
        'last_attempted_scale': 'REAL',
    }

    print("\nAdding missing columns...")
    for col_name, col_type in required_columns.items():
        if col_name not in existing:
            try:
                await conn.execute(f"ALTER TABLE face_detection_failures ADD COLUMN {col_name} {col_type}")
                print(f"  [OK] Added {col_name} ({col_type})")
            except Exception as e:
                print(f"  [ERROR] {col_name}: {e}")
        else:
            print(f"  [SKIP] {col_name} already exists")

    await conn.close()
    print("\nDone")

asyncio.run(fix_schema())
