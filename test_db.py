#!/usr/bin/env python3
"""Quick test of database access."""
import asyncio
import asyncpg

async def test():
    dsn = "postgresql://vision:vision123@192.168.12.20:5432/vision"
    conn = await asyncpg.connect(dsn)

    print("Connected to database")

    # Test table access
    try:
        result = await conn.fetchval("SELECT COUNT(*) FROM face_detection_failures")
        print(f"Table accessible! Rows: {result}")
    except asyncpg.exceptions.InsufficientPrivilegeError as e:
        print(f"NO PERMISSION: {e}")
    except asyncpg.exceptions.UndefinedTableError as e:
        print(f"TABLE NOT FOUND: {e}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

    await conn.close()
    print("Done")

asyncio.run(test())
