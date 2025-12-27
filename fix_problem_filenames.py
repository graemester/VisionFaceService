#!/usr/bin/env python3
"""
Find and permanently fail images with problematic filenames:
- Multiple consecutive periods (... or ....)
- Emoji characters
- Extremely long filenames
- Other special character patterns that cause issues
"""
import asyncio
import asyncpg
import re

DSN = "postgresql://vision:vision123@192.168.12.20:5432/vision"

# Patterns that cause problems
PROBLEM_PATTERNS = [
    (r'\.{3,}', 'multiple_periods'),      # Three or more consecutive periods
    (r'[\U0001F000-\U0001F9FF]', 'emoji'),  # Emoji characters
    (r'[\U00002600-\U000027BF]', 'emoji'),  # More emoji/symbols
    (r'.{200,}', 'long_filename'),         # Extremely long paths (200+ chars in filename part)
]

async def main():
    conn = await asyncpg.connect(DSN)
    print("Connected to database\n")

    # Find images in ai_metadata that are NOT in face_metadata and NOT permanently failed
    # These are the ones that would be picked up for processing
    pending = await conn.fetch("""
        SELECT a.image_path
        FROM ai_metadata a
        LEFT JOIN face_metadata f ON a.image_path = f.image_path
        LEFT JOIN face_detection_failures fd ON a.image_path = fd.image_path
        WHERE f.image_path IS NULL
          AND (fd.image_path IS NULL OR fd.permanently_failed = FALSE)
    """)

    print(f"Found {len(pending)} pending images to check\n")

    problem_images = []
    for row in pending:
        path = row['image_path']
        # Extract just the filename for pattern matching
        filename = path.split('/')[-1] if '/' in path else path

        for pattern, reason in PROBLEM_PATTERNS:
            if re.search(pattern, filename):
                problem_images.append((path, reason))
                break

    if not problem_images:
        print("No problematic filenames found!")
        await conn.close()
        return

    print(f"Found {len(problem_images)} images with problematic filenames:\n")
    for path, reason in problem_images[:20]:  # Show first 20
        display = path[:70] + "..." if len(path) > 70 else path
        print(f"  [{reason}] {display}")

    if len(problem_images) > 20:
        print(f"  ... and {len(problem_images) - 20} more\n")

    # Ask for confirmation
    confirm = input(f"\nMark these {len(problem_images)} images as permanently failed? [y/N]: ")
    if confirm.lower() != 'y':
        print("Aborted")
        await conn.close()
        return

    # Mark them as permanently failed
    count = 0
    for path, reason in problem_images:
        try:
            await conn.execute("""
                INSERT INTO face_detection_failures
                    (image_path, last_error, last_error_type, permanently_failed,
                     retry_count, next_retry_after, first_failure_at, last_failure_at)
                VALUES ($1, $2, $3, TRUE, 99,
                        CURRENT_TIMESTAMP + INTERVAL '100 years',
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (image_path) DO UPDATE SET
                    permanently_failed = TRUE,
                    last_error = EXCLUDED.last_error,
                    last_error_type = EXCLUDED.last_error_type,
                    next_retry_after = CURRENT_TIMESTAMP + INTERVAL '100 years'
            """, path, f"Problematic filename: {reason}", f"filename_{reason}")
            count += 1
        except Exception as e:
            print(f"  Error for {path[:50]}: {e}")

    print(f"\nMarked {count} images as permanently failed")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
