from __future__ import annotations

from nav4rail_graph_rag.domain import GenerationResult


def score_generation(result: GenerationResult, expected_skills: tuple[str, ...] = ()) -> dict[str, float | int | bool]:
    generated_skills = {step.skill for step in result.steps}
    expected = set(expected_skills)
    recall = len(generated_skills & expected) / len(expected) if expected else 0.0
    hallucinated = [skill for skill in generated_skills if skill not in result.retrieval.candidate_skills]
    return {
        "l1_ok": result.validation.l1_ok,
        "l2_ok": result.validation.l2_ok,
        "l3_ok": result.validation.l3_ok,
        "ok": result.validation.ok,
        "n_nodes": result.validation.n_nodes,
        "n_steps": len(result.steps),
        "skill_recall": round(recall, 3),
        "hallucinated_skills": len(hallucinated),
        "retrieved_skills": len(result.retrieval.candidate_skills),
    }
