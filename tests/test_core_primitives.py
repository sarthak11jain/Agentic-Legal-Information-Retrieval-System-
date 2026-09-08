from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_rag.evaluation import macro_f1, micro_f1
from legal_rag.config import PipelineConfig
from legal_rag.graph import expand_neighbors, neighbors_for
from legal_rag.llm import confirm_removals, parse_decisions
from legal_rag.retrieval import fuse_rankings, minmax_normalize
from legal_rag.reranking import RerankWeights, combine_weighted_scores


class RetrievalTests(unittest.TestCase):
    def test_rrf_rewards_candidates_repeated_across_rankings(self) -> None:
        scores = fuse_rankings([["A", "B"], ["B", "A"]], rrf_k=60)
        self.assertEqual(scores["A"], scores["B"])

    def test_rrf_ignores_duplicate_within_one_ranking(self) -> None:
        scores = fuse_rankings([["A", "A", "B"]], rrf_k=0)
        self.assertEqual(scores["A"], 1.0)
        self.assertEqual(scores["B"], 0.5)

    def test_minmax_constant_scores_are_ones(self) -> None:
        self.assertEqual(minmax_normalize({"a": 2.0, "b": 2.0}), {"a": 1.0, "b": 1.0})


class GraphTests(unittest.TestCase):
    GRAPH = {
        "outgoing": {"A": {"B"}},
        "incoming": {"A": {"C"}},
    }

    def test_neighbors_support_both_directions(self) -> None:
        self.assertEqual(neighbors_for("A", "both", self.GRAPH), {"B", "C"})

    def test_expansion_collects_neighbors_for_all_seeds(self) -> None:
        self.assertEqual(expand_neighbors({"A"}, self.GRAPH), {"B", "C"})


class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions_have_f1_one(self) -> None:
        values = [["A"], ["B", "C"]]
        self.assertEqual(macro_f1(values, values), 1.0)
        self.assertEqual(micro_f1(values, values), 1.0)

    def test_empty_prediction_is_penalized(self) -> None:
        self.assertEqual(macro_f1([[]], [["A"]]), 0.0)
        self.assertEqual(micro_f1([[]], [["A"]]), 0.0)


class RerankingTests(unittest.TestCase):
    def test_weighted_scores_combine_three_signals(self) -> None:
        scores = combine_weighted_scores(
            {"A": 1.0},
            {"A": 0.5},
            {"A": 0.0},
            weights=RerankWeights(tfidf=0.4, cosine=0.2, cross_encoder=0.4),
        )
        self.assertAlmostEqual(scores["A"], 0.5)

    def test_negative_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RerankWeights(tfidf=-0.1)


class LLMDecisionTests(unittest.TestCase):
    def test_valid_json_decisions_are_parsed(self) -> None:
        decisions, error = parse_decisions('[{"label": "A", "decision": "REMOVE"}]')
        self.assertEqual(decisions, {"A": "REMOVE"})
        self.assertEqual(error, "")

    def test_malformed_json_uses_fallback_parser(self) -> None:
        decisions, error = parse_decisions('{"label":"B", "decision":"KEEP"}')
        self.assertEqual(decisions, {"B": "KEEP"})
        self.assertIn("fallback", error)

    def test_removal_requires_confirmation(self) -> None:
        self.assertEqual(
            confirm_removals({"A": "REMOVE", "B": "KEEP"}, {"A": "KEEP"}),
            {"A": "KEEP", "B": "KEEP"},
        )


class ConfigTests(unittest.TestCase):
    def test_documented_defaults_validate(self) -> None:
        PipelineConfig().validate()

    def test_invalid_graph_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig(graph_mode="invalid").validate()


if __name__ == "__main__":
    unittest.main()
