from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryDocumentationTests(unittest.TestCase):
    def test_readme_local_links_resolve(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        targets = re.findall(r"\[[^]]+\]\((?!https?://)([^)#]+)(?:#[^)]+)?\)", readme)
        missing = [target for target in targets if not (ROOT / target).is_file()]
        self.assertEqual(missing, [])

    def test_environment_template_covers_archive_backends(self) -> None:
        template = (ROOT / ".env.example").read_text(encoding="utf-8")
        variables = (
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "FIREWORKS_API_KEY",
            "ZEROENTROPY_API_KEY",
        )
        for variable in variables:
            self.assertIn(f"{variable}=", template)


if __name__ == "__main__":
    unittest.main()
