from __future__ import annotations

import unittest

from legal_rag.config import PipelineConfig
from legal_rag.evaluation import macro_f1, micro_f1
from legal_rag.graph import expand_neighbors, neighbors_for
from legal_rag.llm import confirm_removals, parse_decisions
from legal_rag.reranking import RerankWeights, combine_weighted_scores
from legal_rag.retrieval import fuse_rankings, minmax_normalize


class PrimitiveTests(unittest.TestCase):
    def test_rrf_ignores_duplicate_candidates(self) -> None:
        self.assertEqual(fuse_rankings([["A", "A", "B"]], rrf_k=0), {"A": 1.0, "B": 0.5})

    def test_constant_scores_normalize_to_one(self) -> None:
        self.assertEqual(minmax_normalize({"a": 2.0, "b": 2.0}), {"a": 1.0, "b": 1.0})
        self.assertEqual(minmax_normalize({}), {})

    def test_graph_supports_both_directions(self) -> None:
        graph = {"outgoing": {"A": {"B"}}, "incoming": {"A": {"C"}}}
        self.assertEqual(neighbors_for("A", "both", graph), {"B", "C"})
        self.assertEqual(neighbors_for("A", "incoming", graph), {"C"})
        self.assertEqual(expand_neighbors({"A"}, graph), {"B", "C"})

    def test_metrics_handle_empty_predictions(self) -> None:
        self.assertEqual(macro_f1([[]], [["A"]]), 0.0)
        self.assertEqual(micro_f1([[]], [["A"]]), 0.0)

    def test_weighted_scores_combine_signals(self) -> None:
        scores = combine_weighted_scores(
            {"A": 1.0}, {"A": 0.5}, {"A": 0.0}, weights=RerankWeights(0.4, 0.2, 0.4)
        )
        self.assertAlmostEqual(scores["A"], 0.5)

    def test_llm_removal_requires_confirmation(self) -> None:
        decisions, error = parse_decisions('[{"label": "A", "decision": "REMOVE"}]')
        self.assertEqual(error, "")
        self.assertEqual(confirm_removals(decisions, {"A": "KEEP"}), {"A": "KEEP"})
        fallback, diagnostic = parse_decisions('{"label":"B", "decision":"KEEP"}')
        self.assertEqual(fallback, {"B": "KEEP"})
        self.assertIn("fallback", diagnostic)

    def test_pipeline_configuration_validates(self) -> None:
        PipelineConfig().validate()
        with self.assertRaises(ValueError):
            PipelineConfig(seed_k=0).validate()
        with self.assertRaises(ValueError):
            PipelineConfig(lexical_weight=0, rrf_weight=0, graph_weight=0).validate()


if __name__ == "__main__":
    unittest.main()
