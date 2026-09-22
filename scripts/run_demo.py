"""Compatibility launcher for the installed ``legal-rag-demo`` command."""

from legal_rag.cli import demo_main

if __name__ == "__main__":
    raise SystemExit(demo_main())
