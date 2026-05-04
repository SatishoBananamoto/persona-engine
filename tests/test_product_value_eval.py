"""Regression tests for the product-value eval harness."""

from __future__ import annotations

from eval.product_value.run_eval import (
    band_match,
    decision_gates,
    format_ir_block,
    format_rich_ir_block,
    forbidden_marker_hit,
    matrix_status,
    score_item,
    summarize,
    summarize_ir,
)
from persona_engine import PersonaEngine


def _item(method: str = "schema_driven") -> dict:
    return {
        "method": method,
        "repeat": 0,
        "scenario_id": "s1",
        "turn_id": "t1",
        "turn_index": 0,
        "persona_id": "p1",
        "comparison_group": "g1",
        "response": "I am not a financial advisor. I would raise workflow and trust concerns.",
        "category": "enterprise_research",
        "persona_label": "Alex - Software Engineer",
        "expected": {
            "competence": "high",
            "claim": "personal_experience",
            "markers": ["workflow", "trust"],
            "forbidden_markers": ["financial advisor"],
            "min_words": None,
            "max_words": None,
        },
        "ir": {
            "competence": 0.82,
            "claim_type": "personal_experience",
            "citations": 20,
            "cannot_claim": ["financial advisor"],
        },
    }


def test_missing_required_methods_cannot_proceed() -> None:
    item = _item()
    item["auto_scores"] = score_item(item)

    summary = summarize([item])

    assert summary["matrix_status"]["complete"] is False
    assert summary["decision_gates"]["schema_driven_proceed_signal"] is False
    assert summary["decision_gates"]["proceed_signal"] is False


def test_matrix_requires_all_methods_for_same_cells() -> None:
    results = []
    for method in ("prompt_only", "schema_driven"):
        item = _item(method)
        item["auto_scores"] = score_item(item)
        results.append(item)

    matrix = matrix_status(results)

    assert matrix["complete"] is False
    assert "schema_rich" in matrix["missing_methods"]
    assert matrix["missing_cell_count"] > 0


def test_competence_bands_do_not_overlap() -> None:
    assert band_match(0.20, "low") is True
    assert band_match(0.20, "medium") is False
    assert band_match(0.50, "medium") is True
    assert band_match(0.50, "low") is False
    assert band_match(0.80, "high") is True
    assert band_match(0.80, "medium") is False


def test_negated_forbidden_marker_is_not_a_hit() -> None:
    assert forbidden_marker_hit("i am not a financial advisor.", "financial advisor") is False
    assert forbidden_marker_hit("as a financial advisor, i guarantee it.", "financial advisor") is True


def test_trace_presence_is_not_trace_correctness() -> None:
    item = _item()
    item["ir"]["competence"] = 0.10
    item["ir"]["claim_type"] = "domain_expert"

    scores = score_item(item)

    assert scores["trace_presence_score"] > 0
    assert scores["trace_correctness_score"] == 0
    assert scores["traceability_score"] == scores["trace_correctness_score"]


def test_decision_gates_fail_when_matrix_incomplete() -> None:
    method_summary = {
        "prompt_only": {"avg_product_score": 0.1, "forbidden_hits": 0, "segment_differentiation": 0.1},
        "no_ir_ablation": {"avg_product_score": 0.1},
        "stale_ir": {"avg_product_score": 0.1},
        "contradictory_ir": {"avg_product_score": 0.1},
        "shuffled_ir": {"avg_product_score": 0.1},
        "schema_driven": {"avg_product_score": 0.9, "forbidden_hits": 0, "segment_differentiation": 0.2},
        "schema_rich": {"avg_product_score": 0.9, "forbidden_hits": 0, "segment_differentiation": 0.2},
    }

    gates = decision_gates(method_summary, {"complete": False})

    assert gates["schema_driven_proceed_signal"] is False
    assert gates["schema_rich_proceed_signal"] is False
    assert gates["proceed_signal"] is False


def test_eval_serializes_research_ir_contract() -> None:
    engine = PersonaEngine.from_yaml(
        "personas/software_engineer.yaml",
        llm_provider="mock",
        seed=42,
    )
    ir = engine.plan(
        "Our company wants to require engineers to use an AI code review assistant "
        "on every pull request. What concerns would you raise?"
    )

    summary = summarize_ir(ir)
    compact = format_ir_block(ir)
    rich = format_rich_ir_block(ir)

    assert summary is not None
    assert summary["research"]["claim_basis"] == "direct_workflow_experience"
    assert "RESEARCH PARAMETERS:" in compact
    assert "false positives in review" in compact
    assert '"research"' in rich
    assert "not measured market evidence" in rich
