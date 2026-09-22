from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from legal_rag.cli import check_main, demo_main
from legal_rag.config import PipelineConfig, load_config
from legal_rag.models import CitationDocument, Query
from legal_rag.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = Query("q1", "proportionality and hearing", ("Art. 5 BV", "Art. 29 BV"))
        self.documents = (
            CitationDocument("Art. 5 BV", "proportionality", ("proportionality",), ("Art. 29 BV",)),
            CitationDocument("Art. 29 BV", "right to a hearing", ("hearing",), ("Art. 5 BV",)),
            CitationDocument("Art. 9 BV", "good faith", ("good faith",)),
        )

    def test_pipeline_ranks_gold_citations_and_evaluates(self) -> None:
        result = run_pipeline(self.query, self.documents)
        self.assertEqual(
            [item.citation for item in result.ranked_citations[:2]],
            ["Art. 5 BV", "Art. 29 BV"],
        )
        self.assertEqual(result.macro_f1, 0.8)

    def test_empty_corpus_returns_empty_result(self) -> None:
        self.assertEqual(run_pipeline(self.query, ()).ranked_citations, ())

    def test_config_file_changes_supported_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                (
                    "rrf_k = 0\nseed_k = 1\nresult_limit = 2\n"
                    "[weights]\nlexical = 1\nrrf = 0\ngraph = 0\n"
                ),
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(
            config,
            PipelineConfig(
                rrf_k=0,
                seed_k=1,
                result_limit=2,
                lexical_weight=1.0,
                rrf_weight=0.0,
                graph_weight=0.0,
            ),
        )

    def test_cli_emits_machine_readable_demo(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(demo_main([]), 0)
        payload = json.loads(output.getvalue())
        expected_path = Path(__file__).parent / "fixtures" / "demo_output.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(payload, expected)

    def test_check_cli_reports_data_free_support(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(check_main([]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "ok")
        self.assertFalse(payload["competition_data_required"])


if __name__ == "__main__":
    unittest.main()
