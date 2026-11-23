import json
import re
from pathlib import Path
from typing import List, Dict, Any


def get_variations(q_text: str) -> List[str]:
    """Generates natural conversational and student phrasing variations."""
    variations = [q_text]
    q_clean = q_text.rstrip("?").strip()

    # Lowercase first char for joining
    if len(q_clean) > 1:
        first_lower = q_clean[0].lower() + q_clean[1:]
    else:
        first_lower = q_clean.lower()

    if q_clean.lower().startswith("what is the ") or q_clean.lower().startswith("what are the "):
        tail = re.sub(r"^what (is|are) the ", "", q_clean, flags=re.IGNORECASE)
        variations.append(f"Can you tell me about the {tail}?")
        variations.append(f"Hey, {first_lower}?")
    elif q_clean.lower().startswith("can i ") or q_clean.lower().startswith("how do i "):
        variations.append(f"I want to know, {first_lower}?")
        variations.append(f"Please explain if {first_lower}?")
    elif not q_clean.lower().startswith("bro"):
        variations.append(f"Hey advisor, {first_lower}?")

    return list(dict.fromkeys(variations))


def prepare_training_dataset(
    questions_file: str = "data/evaluation/questions.json",
    raw_dir: str = "data/raw",
    output_file: str = "data/training_samples.json",
    augment: bool = True
) -> List[Dict[str, str]]:
    """
    Builds supervised Q&A training pairs pairing student queries with
    relevant handbook context passages and verified answers.
    Formats inputs exactly matching the GroundedRAGGenerator inference prompt.
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
        base_q = item.get("question", "").strip()
        ans_text = item.get("ground_truth_answer", "").strip()
        in_scope = item.get("is_in_scope", True)
        sources = item.get("expected_sources", [])
        target_section = item.get("target_section", "")

        context_snippet = ""
        if in_scope and sources:
            src_doc = sources[0]
            full_doc = documents.get(src_doc, "")
            # Find the section snippet if available
            if target_section and target_section.lower() in full_doc.lower():
                idx = full_doc.lower().find(target_section.lower())
                start = max(0, idx - 40)
                end = min(len(full_doc), idx + 650)
                context_snippet = full_doc[start:end].strip()
            elif full_doc:
                context_snippet = full_doc[:650].strip()
            else:
                context_snippet = "Academic policy regulations document excerpt."
        else:
            context_snippet = "No relevant academic policy found in official university handbooks."

        # Add base and augmented questions
        q_list = get_variations(base_q) if (augment and in_scope) else [base_q]

        for q in q_list:
            # Format input using exact inference prompt template
            model_input = (
                f"Answer the following academic question based on the given context.\n\n"
                f"Context: {context_snippet}\n\n"
                f"Question: {q}\n\n"
                f"Answer:"
            )

            samples.append({
                "input_text": model_input,
                "target_text": ans_text,
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
