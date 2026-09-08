from __future__ import annotations

from collections.abc import Sequence
from textwrap import dedent


LABELS = ("A", "B", "C", "D")


def build_conservative_llm_law_reranking_prompt(
    german_query: str,
    citations: Sequence[str],
    documents: Sequence[str],
    court_contexts: Sequence[Sequence[dict[str, object]]] | None = None,
) -> str:
    """Build the conservative law-citation KEEP/REMOVE prompt for up to four candidates."""
    if len(citations) != len(documents):
        raise ValueError("citations and documents must have the same length")
    if court_contexts is not None and len(citations) != len(court_contexts):
        raise ValueError("citations and court_contexts must have the same length")
    if not citations:
        raise ValueError("at least one citation/document pair is required")
    if len(citations) > len(LABELS):
        raise ValueError(f"at most {len(LABELS)} citation/document pairs are supported")

    if court_contexts is None:
        court_contexts = [[] for _ in citations]
    has_court_context = any(bool(contexts) for contexts in court_contexts)
    candidate_blocks = []
    for label, citation, document, contexts in zip(
        LABELS,
        citations,
        documents,
        court_contexts,
    ):
        block = "\n".join(
            [
                f"{label}.",
                f"Citation: {_clean(citation)}",
                "Article text:",
                _clean(document),
            ]
        )
        if has_court_context:
            block = (
                block.rstrip()
                + "\n\nCourt usage context:\n"
                + _format_court_contexts(contexts)
            )
        candidate_blocks.append(block.strip())

    json_rows = ",\n  ".join(
        f'{{"label": "{label}", "decision": "KEEP/REMOVE"}}'
        for label in LABELS[: len(citations)]
    )

    if has_court_context:
        header = dedent(
            """
            You are checking candidate Swiss law citations for a legal retrieval system.

            Your task is NOT to decide whether an article is generally related to the query.
            Your task is NOT to choose only the best or most central citations.
            Your task is to remove only clear non-citations, while keeping articles that could reasonably support a complete legal answer.

            Use legal reasoning, but ground your decision only in the supplied materials:
            - German query: the current legal question to answer.
            - Candidate citation: the article being tested.
            - Candidate article text:
              - Gesetz identifies the statute/law title and code. It helps locate the legal area.
              - Regelungsbereich describes the section, chapter, or legal domain inside the statute. It helps understand the article's local context when present.
              - Normtext contains the actual article or paragraph text. It holds the main legal rule to evaluate.
            - Court usage context: a past court passage selected because it cites this candidate article and is similar to the current German query.

            Read these inputs together:
            1. Start from the current German query. Identify the legal issue, facts, procedural stage, requested standard, and requested consequence.
            2. Read the candidate Normtext. Identify what concrete legal role the article could play: rule, condition, standard, deadline, competence, procedural step, document duty, consequence, limitation, exception, definition, or supporting legal basis.
            3. If court usage context is present for that candidate, use it as a past usage example. Ask why the court cited this article there, and whether the same legal role is also needed for the current query.
            4. Court usage strengthens KEEP when the passage applies or discusses the article for a legal issue, sub-issue, procedural stage, standard, consequence, or reasoning step that is also present in the current query.
            5. Court usage weakens KEEP when the passage shows a different legal issue, different procedural stage, different actor, different consequence, or only an incidental citation bundle.
            6. Do not keep the article merely because court usage context exists. The article text must still contribute to the current query's legal answer.
            7. If no court usage context is available for a candidate, decide from the German query and article text only. Do not penalize the candidate merely because no court usage context was found.

            If Regelungsbereich is missing, use Gesetz and Normtext only.
            Focus mainly on whether Normtext contains a rule that maps to the query, with court usage context as supporting evidence about how that rule is used.
            """
        ).strip()
    else:
        header = dedent(
            """
            You are checking candidate Swiss law citations for a legal retrieval system.

            Your task is NOT to decide whether an article is generally related to the query.
            Your task is NOT to choose only the best or most central citations.
            Your task is to remove only clear non-citations, while keeping articles that could reasonably support a complete legal answer.

            Use legal reasoning, but ground your decision only in the supplied materials:
            - German query: the current legal question to answer.
            - Candidate citation: the article being tested.
            - Candidate article text:
              - Gesetz identifies the statute/law title and code. It helps locate the legal area.
              - Regelungsbereich describes the section, chapter, or legal domain inside the statute. It helps understand the article's local context when present.
              - Normtext contains the actual article or paragraph text. It holds the main legal rule to evaluate.

            Read these inputs together:
            1. Start from the current German query. Identify the legal issue, facts, procedural stage, requested standard, and requested consequence.
            2. Read the candidate Normtext. Identify what concrete legal role the article could play: rule, condition, standard, deadline, competence, procedural step, document duty, consequence, limitation, exception, definition, or supporting legal basis.
            3. Decide whether that role is useful for a complete legal answer to the current query.

            If Regelungsbereich is missing, use Gesetz and Normtext only.
            Focus mainly on whether Normtext contains a rule that maps to the query.
            """
        ).strip()

    if has_court_context:
        decision_checks = [
            "1. Does the article's Normtext state a legal rule, condition, standard, step, authority role, or consequence that maps to the query?",
            "2. Could it support any plausible branch, preliminary issue, required substep, alternative path, or legal consequence in a complete answer?",
            "3. Does the past court passage show the article being used for the same kind of legal role needed here?",
            "4. Or is the article clearly outside the answer path despite being in a similar legal area?",
        ]
        keep_court_lines = [
            "- Court usage context shows the article being applied or discussed for the same kind of issue, standard, procedural stage, consequence, or reasoning step needed in the current query.",
        ]
        remove_court_lines = [
            "- Court usage context shows the article being used for a different legal issue, different procedural stage, different actor, different consequence, or only as an incidental bundled citation.",
        ]
    else:
        decision_checks = [
            "1. Does the article's Normtext state a legal rule, condition, standard, step, authority role, or consequence that maps to the query?",
            "2. Could it support any plausible branch, preliminary issue, required substep, alternative path, or legal consequence in a complete answer?",
            "3. Or is the article clearly outside the answer path despite being in a similar legal area?",
        ]
        keep_court_lines = []
        remove_court_lines = []

    keep_lines = [
        "- The article states a rule, condition, right, remedy, test, deadline, competence rule, procedural step, document requirement, form requirement, authority review, or legal consequence that can support the answer.",
        "- The article defines or frames a legal institution, right, procedure, authority role, or legal effect that is part of the answer path.",
        "- The article provides a general legal basis that would reasonably be cited before, together with, or after a more specific rule.",
        "- The article supports a plausible answer branch, threshold issue, alternative legal route, or required substep suggested by the query facts.",
        *keep_court_lines,
        "- If the query asks about register, record, filing, authority, or implementation steps, keep articles that specify what must be entered, recorded, annotated, noticed, displayed, dated, referenced, or checked in that register or official record.",
        "- The article may be only partially relevant, but removing it would be risky because it could support one step in the legal answer.",
        "- You are uncertain.",
    ]
    remove_lines = [
        "- The article is about a clearly different legal institution, right, authority, register, procedure, remedy, or factual situation from the query.",
        "- The article only concerns purely internal administration, publication, archiving, formatting, or technical handling that is not legally operative and is not requested by the query.",
        "- The article only concerns a later, contingent, or exceptional path that is not raised by the query and is not needed to answer the requested steps or legal consequences.",
        *remove_court_lines,
        "- The article is same-area law but does not add any rule, condition, document requirement, authority role, formal step, or legal consequence that would reasonably be cited in an answer.",
    ]
    important_lines = [
        "- Use REMOVE only when there is a clear mismatch or no useful legal contribution to the answer path.",
        "- Do not use REMOVE merely because another candidate is more specific, more central, or a better citation.",
        "- Do not use REMOVE merely because the article is preliminary, definitional, conditional, broad, or only relevant to one plausible branch of the answer.",
        "- Do not use REMOVE merely because an article describes the content, placement, date, reference, or official recording mechanics of a legally relevant entry.",
        "- If the article is plausible but not perfect, use KEEP.",
        "- If the article is broad but still frames, enables, limits, or supports a required part of the legal answer, use KEEP.",
        "- If you cannot decide from the query and article text alone, use KEEP.",
    ]
    output_lines = [
        "Return a compact JSON array only.",
        "Each object must contain exactly these two keys: label and decision.",
        "Decision must be either KEEP or REMOVE.",
        "The JSON below shows the format only. Replace KEEP/REMOVE with one final label for each candidate.",
        "Do not output KEEP/REMOVE literally.",
        "Do not include explanation text, reason fields, reasoning fields, markdown, or comments.",
    ]
    instructions = "\n".join(
        [
            "For each candidate article, decide whether the article is citation-worthy for this query:",
            *decision_checks,
            "",
            "Decision labels:",
            "",
            "KEEP:",
            *keep_lines,
            "",
            "REMOVE:",
            *remove_lines,
            "",
            "Important:",
            *important_lines,
            "",
            "Output format:",
            *output_lines,
        ]
    )

    json_example = f"[\n  {json_rows}\n]"

    return "\n\n".join(
        [
            header,
            f"German query:\n{_clean(german_query)}",
            "Candidate law articles:\n\n" + "\n\n".join(candidate_blocks),
            instructions,
            json_example,
        ]
    )


def build_conservative_removal_confirmation_prompt(
    german_query: str,
    citations: Sequence[str],
    documents: Sequence[str],
    court_contexts: Sequence[Sequence[dict[str, object]]] | None = None,
) -> str:
    """Build the safety-check prompt for candidates marked REMOVE by the first pass."""
    if len(citations) != len(documents):
        raise ValueError("citations and documents must have the same length")
    if court_contexts is not None and len(citations) != len(court_contexts):
        raise ValueError("citations and court_contexts must have the same length")
    if not citations:
        raise ValueError("at least one citation/document pair is required")
    if len(citations) > len(LABELS):
        raise ValueError(f"at most {len(LABELS)} citation/document pairs are supported")

    if court_contexts is None:
        court_contexts = [[] for _ in citations]
    has_court_context = any(bool(contexts) for contexts in court_contexts)

    candidate_blocks = []
    for label, citation, document, contexts in zip(
        LABELS,
        citations,
        documents,
        court_contexts,
    ):
        block = "\n".join(
            [
                f"{label}.",
                f"Citation: {_clean(citation)}",
                "Article text:",
                _clean(document),
            ]
        )
        if has_court_context:
            block = (
                block.rstrip()
                + "\n\nCourt usage context:\n"
                + _format_court_contexts(contexts)
            )
        candidate_blocks.append(block.strip())

    json_rows = ",\n  ".join(
        f'{{"label": "{label}", "decision": "KEEP/REMOVE"}}'
        for label in LABELS[: len(citations)]
    )

    supplied_materials = [
        "- German query: the current legal question to answer.",
        "- Candidate article: a law citation that the first pass marked REMOVE.",
        "- Candidate article text: Gesetz gives the statute, Regelungsbereich gives local context when present, and Normtext is the main legal rule to evaluate.",
    ]
    if has_court_context:
        supplied_materials.append(
            "- Court usage context: a past court passage selected because it cites this candidate article and is similar to the current German query."
        )

    keep_lines = [
        "- The article has any plausible legal role in a complete answer: rule, condition, legal standard, authority role, procedural step, deadline, consequence, exception, definition, or supporting legal basis.",
        "- The article could support any branch, sub-issue, threshold issue, alternative path, or legal consequence raised by the query.",
        "- The article is broad, preliminary, definitional, or only partially relevant, but it could still support one legal step.",
        "- Removing it would risk losing a real citation-worthy part of the answer.",
        "- You are uncertain.",
    ]
    if has_court_context:
        keep_lines.insert(
            2,
            "- Court usage context shows the article being applied or discussed for a similar legal role needed in the current query.",
        )

    remove_lines = [
        "- The article clearly has no useful legal contribution to the current query's answer path.",
        "- The article concerns a different legal issue, actor, object, procedure, stage, remedy, consequence, exception, or factual situation.",
        "- The article is only same-topic or neighboring law and its Normtext would not reasonably be cited in a complete answer.",
    ]
    if has_court_context:
        remove_lines.append(
            "- Court usage context shows only a different legal role or an incidental bundled citation that does not help answer the current query."
        )

    return "\n\n".join(
        [
            dedent(
                """
                You are doing a final safety check before removing candidate Swiss law citations.

                The first pass marked these candidates REMOVE.
                Your job is to catch false removals.
                Change a candidate back to KEEP if it has any plausible legal role in a complete answer.
                Confirm REMOVE only when the article clearly has no useful legal contribution to the answer path.

                Use only the supplied materials:
                """
            ).strip()
            + "\n"
            + "\n".join(supplied_materials),
            f"German query:\n{_clean(german_query)}",
            "Candidate law articles:\n\n" + "\n\n".join(candidate_blocks),
            "\n".join(
                [
                    "KEEP if:",
                    *keep_lines,
                    "",
                    "REMOVE if:",
                    *remove_lines,
                    "",
                    "Output format:",
                    "Return a compact JSON array only.",
                    "Each object must contain exactly these two keys: label and decision.",
                    "Decision must be either KEEP or REMOVE.",
                    "The JSON below shows the format only. Replace KEEP/REMOVE with one final label for each candidate.",
                    "Do not output KEEP/REMOVE literally.",
                    "Do not include explanation text, reason fields, reasoning fields, markdown, or comments.",
                ]
            ),
            f"[\n  {json_rows}\n]",
        ]
    )


def build_conservative_no_court_law_reranking_prompt(
    german_query: str,
    citations: Sequence[str],
    documents: Sequence[str],
) -> str:
    """Build the no-court ablation prompt for conservative law-citation filtering."""
    candidate_blocks, json_example = _format_candidates_and_json(citations, documents)

    header = dedent(
        """
        You are checking candidate Swiss law citations for a legal retrieval system.

        Your task is NOT to decide whether an article is generally related to the query.
        Your task is NOT to choose only the best or most central citations.
        Your task is to remove only clear non-citations, while keeping articles that could reasonably support a complete legal answer.

        Use legal reasoning, but ground your decision only in the supplied materials:
        - German query: the current legal question to answer.
        - Candidate citation: the article being tested.
        - Candidate article text:
          - Gesetz identifies the statute/law title and code. It helps locate the legal area.
          - Regelungsbereich describes the section, chapter, or legal domain inside the statute. It helps understand the article's local context when present.
          - Normtext contains the actual article or paragraph text. It holds the main legal rule to evaluate.

        Read these inputs together:
        1. Start from the current German query. Identify the legal issue, facts, procedural stage, requested standard, and requested consequence.
        2. Read the candidate Normtext. Identify what concrete legal role the article could play: rule, condition, standard, deadline, competence, procedural step, document duty, consequence, limitation, exception, definition, or supporting legal basis.
        3. Decide whether that role is useful for a complete legal answer to the current query.

        If Regelungsbereich is missing, use Gesetz and Normtext only.
        Focus mainly on whether Normtext contains a rule that maps to the query.
        """
    ).strip()

    instructions = "\n".join(
        [
            "For each candidate article, decide whether the article is citation-worthy for this query:",
            "1. Does the article's Normtext state a legal rule, condition, standard, step, authority role, or consequence that maps to the query?",
            "2. Could it support any plausible branch, preliminary issue, required substep, alternative path, or legal consequence in a complete answer?",
            "3. Or is the article clearly outside the answer path despite being in a similar legal area?",
            "",
            "Decision labels:",
            "",
            "KEEP:",
            "- The article states a rule, condition, right, remedy, test, deadline, competence rule, procedural step, document requirement, form requirement, authority review, or legal consequence that can support the answer.",
            "- The article defines or frames a legal institution, right, procedure, authority role, or legal effect that is part of the answer path.",
            "- The article provides a general legal basis that would reasonably be cited before, together with, or after a more specific rule.",
            "- The article supports a plausible answer branch, threshold issue, alternative legal route, or required substep suggested by the query facts.",
            "- If the query asks about register, record, filing, authority, or implementation steps, keep articles that specify what must be entered, recorded, annotated, noticed, displayed, dated, referenced, or checked in that register or official record.",
            "- The article may be only partially relevant, but removing it would be risky because it could support one step in the legal answer.",
            "- You are uncertain.",
            "",
            "REMOVE:",
            "- The article is about a clearly different legal institution, right, authority, register, procedure, remedy, or factual situation from the query.",
            "- The article only concerns purely internal administration, publication, archiving, formatting, or technical handling that is not legally operative and is not requested by the query.",
            "- The article only concerns a later, contingent, or exceptional path that is not raised by the query and is not needed to answer the requested steps or legal consequences.",
            "- The article is same-area law but does not add any rule, condition, document requirement, authority role, formal step, or legal consequence that would reasonably be cited in an answer.",
            "",
            "Important:",
            "- Use REMOVE only when there is a clear mismatch or no useful legal contribution to the answer path.",
            "- Do not use REMOVE merely because another candidate is more specific, more central, or a better citation.",
            "- Do not use REMOVE merely because the article is preliminary, definitional, conditional, broad, or only relevant to one plausible branch of the answer.",
            "- Do not use REMOVE merely because an article describes the content, placement, date, reference, or official recording mechanics of a legally relevant entry.",
            "- If the article is plausible but not perfect, use KEEP.",
            "- If the article is broad but still frames, enables, limits, or supports a required part of the legal answer, use KEEP.",
            "- If you cannot decide from the query and article text alone, use KEEP.",
            "",
            "Output format:",
            "Return a compact JSON array only.",
            "Each object must contain exactly these two keys: label and decision.",
            "Decision must be either KEEP or REMOVE.",
            "The JSON below shows the format only. Replace KEEP/REMOVE with one final label for each candidate.",
            "Do not output KEEP/REMOVE literally.",
            "Do not include explanation text, reason fields, reasoning fields, markdown, or comments.",
        ]
    )

    return "\n\n".join(
        [
            header,
            f"German query:\n{_clean(german_query)}",
            "Candidate law articles:\n\n" + "\n\n".join(candidate_blocks),
            instructions,
            json_example,
        ]
    )


def build_conservative_no_court_removal_confirmation_prompt(
    german_query: str,
    citations: Sequence[str],
    documents: Sequence[str],
) -> str:
    """Build the no-court ablation safety-check prompt for first-pass removals."""
    candidate_blocks, json_example = _format_candidates_and_json(citations, documents)

    header = dedent(
        """
        You are doing a final safety check before removing candidate Swiss law citations.

        The first pass marked these candidates REMOVE.
        Your job is to catch false removals.
        Change a candidate back to KEEP if it has any plausible legal role in a complete answer.
        Confirm REMOVE only when the article clearly has no useful legal contribution to the answer path.

        Use only the supplied materials:
        - German query: the current legal question to answer.
        - Candidate article: a law citation that the first pass marked REMOVE.
        - Candidate article text: Gesetz gives the statute, Regelungsbereich gives local context when present, and Normtext is the main legal rule to evaluate.
        """
    ).strip()

    instructions = "\n".join(
        [
            "KEEP if:",
            "- The article has any plausible legal role in a complete answer: rule, condition, legal standard, authority role, procedural step, deadline, consequence, exception, definition, or supporting legal basis.",
            "- The article could support any branch, sub-issue, threshold issue, alternative path, or legal consequence raised by the query.",
            "- The article is broad, preliminary, definitional, or only partially relevant, but it could still support one legal step.",
            "- Removing it would risk losing a real citation-worthy part of the answer.",
            "- You are uncertain.",
            "",
            "REMOVE if:",
            "- The article clearly has no useful legal contribution to the current query's answer path.",
            "- The article concerns a different legal issue, actor, object, procedure, stage, remedy, consequence, exception, or factual situation.",
            "- The article is only same-topic or neighboring law and its Normtext would not reasonably be cited in a complete answer.",
            "",
            "Output format:",
            "Return a compact JSON array only.",
            "Each object must contain exactly these two keys: label and decision.",
            "Decision must be either KEEP or REMOVE.",
            "The JSON below shows the format only. Replace KEEP/REMOVE with one final label for each candidate.",
            "Do not output KEEP/REMOVE literally.",
            "Do not include explanation text, reason fields, reasoning fields, markdown, or comments.",
        ]
    )

    return "\n\n".join(
        [
            header,
            f"German query:\n{_clean(german_query)}",
            "Candidate law articles:\n\n" + "\n\n".join(candidate_blocks),
            instructions,
            json_example,
        ]
    )


def _format_candidates_and_json(
    citations: Sequence[str],
    documents: Sequence[str],
) -> tuple[list[str], str]:
    if len(citations) != len(documents):
        raise ValueError("citations and documents must have the same length")
    if not citations:
        raise ValueError("at least one citation/document pair is required")
    if len(citations) > len(LABELS):
        raise ValueError(f"at most {len(LABELS)} citation/document pairs are supported")

    candidate_blocks = []
    for label, citation, document in zip(LABELS, citations, documents):
        candidate_blocks.append(
            dedent(
                f"""
                {label}.
                Citation: {_clean(citation)}
                Article text:
                {_clean(document)}
                """
            ).strip()
        )

    json_rows = ",\n  ".join(
        f'{{"label": "{label}", "decision": "KEEP/REMOVE"}}'
        for label in LABELS[: len(citations)]
    )
    return candidate_blocks, f"[\n  {json_rows}\n]"


def _clean(value: object) -> str:
    return str(value).strip()


def _format_court_contexts(contexts: Sequence[dict[str, object]]) -> str:
    if not contexts:
        return "No court usage context available."
    rows = []
    for index, context in enumerate(contexts, start=1):
        court_citation = _clean(context.get("court_citation", ""))
        snippet = _clean(context.get("document", ""))
        rows.append(
            "\n".join(
                [
                    f"{index}. Court citation: {court_citation}",
                    f"   Passage: {snippet}",
                ]
            )
        )
    return "\n".join(rows)
