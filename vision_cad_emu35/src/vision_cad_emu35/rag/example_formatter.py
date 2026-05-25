from __future__ import annotations


def format_reference_example(example: dict, index: int) -> str:
    score = example.get("score")
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
    return (
        f"Reference Example {index}:\n"
        f"Similarity score: {score_text}\n"
        f"Operation_Type: {example.get('operation_type', 'unknown')}\n"
        "The following images show the previous depth map and the target preview overlay for this reference example."
    )

