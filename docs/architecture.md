# Architecture

The maintained package ranks supplied citation documents through five local,
deterministic stages:

1. score query-to-document lexical overlap;
2. fuse lexical and keyword rankings with reciprocal-rank fusion;
3. expand the highest-ranked seed citations by one outgoing graph hop;
4. blend lexical, RRF, and graph scores with validated TOML weights; and
5. emit ranked citations and optional citation-level Macro F1.

This is a reference architecture. It intentionally does not claim to execute
the archived multilingual embedding, cross-encoder, or LLM services.
