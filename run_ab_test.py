"""
A/B Prompt Test using Langfuse experiments + Mistral 8B.

Runs the same dataset against two prompt versions (labels "a" and "b"),
scores each response, and prints a comparison.

Usage:
    python run_ab_test.py
"""

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse
from mistralai import Mistral
import os

PROMPT_NAME = "mistral-ab-test"
DATASET_NAME = "ab-test-eval-set"

langfuse = Langfuse()
mistral = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


# ── Task factory: one per prompt label ───────────────────────────────
def make_task(prompt_label: str):
    """Return a task function bound to a specific prompt label."""

    def task(*, item, **kwargs):
        prompt = langfuse.get_prompt(PROMPT_NAME, label=prompt_label, type="chat")
        messages = prompt.compile(**item["input"])

        generation = langfuse.generation(
            name=f"mistral-{prompt_label}",
            input=messages,
            model=prompt.config.get("model", "ministral-14b-latest"),
            metadata={"prompt_label": prompt_label},
        )

        response = mistral.chat.complete(
            model=prompt.config.get("model", "ministral-14b-latest"),
            messages=messages,
            temperature=prompt.config.get("temperature", 0.5),
        )

        output = response.choices[0].message.content
        generation.end(output=output)

        return output

    return task


# ── Evaluators ───────────────────────────────────────────────────────
def keyword_overlap(*, output, expected_output, **kwargs):
    """Score based on keyword overlap between output and expected answer."""
    if not expected_output or not output:
        return {"name": "keyword_overlap", "value": 0.0}

    expected_words = set(expected_output.lower().split())
    output_words = set(output.lower().split())

    # Remove very short/common words
    stop_words = {"is", "a", "an", "the", "of", "in", "to", "and", "or", "for", "with", "that", "it", "by", "from", "on", "are", "was", "be", "has", "its"}
    expected_words -= stop_words
    output_words -= stop_words

    if not expected_words:
        return {"name": "keyword_overlap", "value": 0.0}

    overlap = expected_words & output_words
    score = len(overlap) / len(expected_words)

    return {"name": "keyword_overlap", "value": round(score, 2)}


def response_length(*, output, **kwargs):
    """Score based on response length (penalizes very short or very long)."""
    if not output:
        return {"name": "response_length", "value": 0.0}

    length = len(output)
    # Sweet spot: 50-500 chars
    if 50 <= length <= 500:
        score = 1.0
    elif length < 50:
        score = length / 50
    else:
        score = max(0.2, 500 / length)

    return {"name": "response_length", "value": round(score, 2)}


# ── Run experiments ──────────────────────────────────────────────────
def run():
    dataset = langfuse.get_dataset(DATASET_NAME)

    results = {}

    for label in ["a", "b"]:
        run_name = f"ab-test-prompt-{label}"
        print(f"\n{'='*50}")
        print(f"Running experiment: {run_name}")
        print(f"{'='*50}")

        scores_overlap = []
        scores_length = []

        for item in dataset.items:
            print(f"\n  Q: {item.input['user_question']}")

            with item.run(run_name=run_name) as root_span:
                # Get prompt and call Mistral
                prompt = langfuse.get_prompt(PROMPT_NAME, label=label, type="chat")
                messages = prompt.compile(**item.input)

                response = mistral.chat.complete(
                    model=prompt.config.get("model", "ministral-14b-latest"),
                    messages=messages,
                    temperature=prompt.config.get("temperature", 0.5),
                )

                output = response.choices[0].message.content
                print(f"  A: {output[:120]}...")

                # Evaluate
                s_overlap = keyword_overlap(
                    output=output, expected_output=item.expected_output
                )
                s_length = response_length(output=output)

                # Score the trace in Langfuse
                root_span.score_trace(
                    name="keyword_overlap", value=s_overlap["value"]
                )
                root_span.score_trace(
                    name="response_length", value=s_length["value"]
                )

                scores_overlap.append(s_overlap["value"])
                scores_length.append(s_length["value"])

        avg_overlap = sum(scores_overlap) / len(scores_overlap) if scores_overlap else 0
        avg_length = sum(scores_length) / len(scores_length) if scores_length else 0

        results[label] = {
            "avg_keyword_overlap": round(avg_overlap, 3),
            "avg_response_length": round(avg_length, 3),
            "n_items": len(scores_overlap),
        }

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Prompt A':>12} {'Prompt B':>12} {'Winner':>10}")
    print(f"{'-'*60}")

    for metric in ["avg_keyword_overlap", "avg_response_length"]:
        val_a = results["a"][metric]
        val_b = results["b"][metric]
        winner = "A" if val_a > val_b else ("B" if val_b > val_a else "Tie")
        print(f"{metric:<25} {val_a:>12.3f} {val_b:>12.3f} {winner:>10}")

    print(f"\nItems evaluated: {results['a']['n_items']}")
    print(f"\nPrompt A: Concise and direct (temp=0.3)")
    print(f"Prompt B: Chain-of-thought detailed (temp=0.7)")
    print(f"\nResults are also available in the Langfuse dashboard.")

    langfuse.flush()


if __name__ == "__main__":
    run()
