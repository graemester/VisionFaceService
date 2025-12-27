#!/usr/bin/env python3
"""Check OCR worker status and recent failures."""
import asyncio
import asyncpg
from datetime import datetime, timedelta

async def check():
    conn = await asyncpg.connect("postgresql://vision:vision123@192.168.12.20:5432/vision")

    # Check ocr_rerun_state current status
    state = await conn.fetchrow("SELECT * FROM ocr_rerun_state ORDER BY id DESC LIMIT 1")
    if state:
        print("OCR Rerun State:")
        print(f"  is_running: {state['is_running']}")
        print(f"  phase: {state['phase']}")
        print(f"  total_images: {state['total_images']}")
        print(f"  processed: {state['processed_count']}")
        print(f"  rejected: {state['rejected_count']}")
        print(f"  improved: {state['improved_count']}")
        current = state['current_file']
        if current:
            print(f"  current_file: {current[:80]}...")
        print(f"  started_at: {state['started_at']}")
        print(f"  completed_at: {state['completed_at']}")

    # Check recent OCR activity (last 10 minutes)
    cutoff = datetime.now() - timedelta(minutes=15)
    recent = await conn.fetch("""
        SELECT image_path, ocr_date, has_text
        FROM ocr_metadata
        WHERE ocr_date > $1
        ORDER BY ocr_date DESC
        LIMIT 10
    """, cutoff)

    print(f"\nRecent OCR activity (last 15 min): {len(recent)} entries")
    for r in recent:
        path = r['image_path'][:60] + "..." if len(r['image_path']) > 60 else r['image_path']
        print(f"  {r['ocr_date']} - has_text={r['has_text']} - {path}")

    # Check if there's an error log table or similar
    try:
        errors = await conn.fetch("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND (table_name LIKE '%error%' OR table_name LIKE '%log%')
        """)
        if errors:
            print("\nError/log tables found:")
            for e in errors:
                print(f"  {e['table_name']}")
    except:
        pass

    await conn.close()

asyncio.run(check())
