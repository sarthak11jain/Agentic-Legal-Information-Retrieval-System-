# Contributing

Contributions must target the supported `src/legal_rag/` package unless they
explicitly improve archive preservation. Do not submit competition data, hidden
labels, derived labels, credentials, generated outputs, or legal advice.

Before opening a pull request, install `.[dev]` and run the same checks as CI:

```bash
ruff check src tests scripts
mypy
coverage run -m unittest discover -s tests -p "test_*.py"
coverage report
legal-rag-demo
```

Describe the public behavior changed, add deterministic tests, and update
documentation or the changelog when appropriate.
