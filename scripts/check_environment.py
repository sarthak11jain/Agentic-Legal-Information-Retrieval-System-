"""Compatibility launcher for the installed ``legal-rag-check`` command."""

from legal_rag.cli import check_main

if __name__ == "__main__":
    raise SystemExit(check_main())
