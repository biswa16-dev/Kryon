import json
from pathlib import Path
from typing import List, Dict, Any


def prepare_training_dataset(
    questions_file: str = "data/evaluation/questions.json",
    raw_dir: str = "data/raw",
    output_file: str = "data/training_samples.json"
) -> List[Dict[str, str]]:
    """
    Builds supervised Q&A training pairs pairing student queries with
    relevant handbook context passages and verified answers.
    """
    q_path = Path(questions_file)
    r_dir = Path(raw_dir)

    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    with open(q_path, "r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    # Read raw handbook texts
    documents: Dict[str, str] = {}
    for p in r_dir.glob("*.md"):
        try:
            documents[p.name] = p.read_text(encoding="utf-8")
        except Exception:
            pass

    samples: List[Dict[str, str]] = []

    for item in questions:
        q_text = item.get("question", "").strip()
        ans_text = item.get("ground_truth_answer", "").strip()
        in_scope = item.get("is_in_scope", True)
        sources = item.get("expected_sources", [])
        target_section = item.get("target_section", "")

        context_snippet = ""
        if in_scope and sources:
            src_doc = sources[0]
            full_doc = documents.get(src_doc, "")
            # Find the section snippet if available
            if target_section and target_section in full_doc:
                idx = full_doc.find(target_section)
                # Take 400 characters around the target section
                start = max(0, idx - 40)
                end = min(len(full_doc), idx + 450)
                context_snippet = full_doc[start:end].strip()
            elif full_doc:
                context_snippet = full_doc[:450].strip()
            else:
                context_snippet = "Academic policy regulations document excerpt."
        else:
            context_snippet = "No relevant academic policy found in official handbooks."

        # Format input using standard T5 / Flan-T5 task formatting
        model_input = f"question: {q_text} context: {context_snippet}"
        model_target = ans_text

        samples.append({
            "input_text": model_input,
            "target_text": model_target,
            "category": item.get("category", "general"),
            "is_in_scope": in_scope
        })

    # Save to disk
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[Dataset] Generated {len(samples)} training pairs -> {output_file}")
    return samples


if __name__ == "__main__":
    prepare_training_dataset()

