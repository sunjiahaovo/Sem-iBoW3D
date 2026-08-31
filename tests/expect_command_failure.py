#!/usr/bin/env python3
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: expect_command_failure.py EXPECTED_SUBSTRING COMMAND [ARGS...]", file=sys.stderr)
        return 2

    expected = sys.argv[1]
    command = sys.argv[2:]
    completed = subprocess.run(command, capture_output=True, text=True)
    combined = completed.stdout + completed.stderr

    if completed.returncode == 0:
        print("Command unexpectedly succeeded", file=sys.stderr)
        print(combined, file=sys.stderr)
        return 1
    if expected not in combined:
        print(f"Expected substring not found: {expected}", file=sys.stderr)
        print(combined, file=sys.stderr)
        return 1

    print("expected failure observed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
