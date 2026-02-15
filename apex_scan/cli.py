import asyncio
import sys
from .core.vulnerability_scanner import PurePythonVulnerabilityScanner

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m apex_scan <target>")
        print("Example: python -m apex_scan https://example.com")
        return

    target = sys.argv[1]
    scanner = PurePythonVulnerabilityScanner()
    asyncio.run(scanner.full_scan(target))

if __name__ == "__main__":
    main()
