"""
Setup script: creates the two prompt versions (labels "a" and "b")
and the evaluation dataset in Langfuse.

Run once before running the A/B test:
    python setup_langfuse.py
"""

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse

PROMPT_NAME = "mistral-ab-test"
DATASET_NAME = "ab-test-eval-set"

langfuse = Langfuse()

def create_prompt(system_content: str, label: str, temperature: float):
    langfuse.create_prompt(
        name=PROMPT_NAME,
        type="chat",
        prompt=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": "{{user_question}}"},
        ],
        config={"model": "ministral-14b-latest", "temperature": temperature},
        labels=[label],
    )
    print(f"Prompt version {label.upper()} created.")


# ── Prompt version A: concise, direct ────────────────────────────────
create_prompt(
    system_content=(
        "You are a helpful and direct assistant. "
        "Answer concisely in at most 2 sentences."
    ),
    label="a",
    temperature=0.3,
)

# ── Prompt version B: chain-of-thought, detailed ────────────────────
create_prompt(
    system_content=(
        "You are an expert assistant. "
        "Think step by step before answering. "
        "Provide a detailed explanation with examples when possible."
    ),
    label="b",
    temperature=0.7,
)

# ── Evaluation dataset ───────────────────────────────────────────────
langfuse.create_dataset(
    name=DATASET_NAME,
    description="Dataset for A/B prompt testing with Mistral 8B",
)

items = [
    {
        "input": {"user_question": "What is recursion in programming?"},
        "expected_output": "Recursion is when a function calls itself to solve a problem by breaking it down into smaller subproblems.",
    },
    {
        "input": {"user_question": "Explain what a REST API is."},
        "expected_output": "A REST API is an interface that follows REST principles for communication between systems using HTTP.",
    },
    {
        "input": {"user_question": "What is the difference between a list and a tuple in Python?"},
        "expected_output": "Lists are mutable and tuples are immutable.",
    },
    {
        "input": {"user_question": "What is Big O notation?"},
        "expected_output": "Big O is a notation to describe the time or space complexity of an algorithm.",
    },
    {
        "input": {"user_question": "Explain the concept of closure in JavaScript."},
        "expected_output": "A closure is a function that has access to the outer function's scope even after it has returned.",
    },
    {
        "input": {"user_question": "What is a relational database?"},
        "expected_output": "A relational database organizes data into tables with rows and columns, using relations between them.",
    },
    {
        "input": {"user_question": "What is Docker used for?"},
        "expected_output": "Docker is used to create containers that package an application with all its dependencies.",
    },
    {
        "input": {"user_question": "What is machine learning?"},
        "expected_output": "Machine learning is a field of AI where algorithms learn patterns from data without being explicitly programmed.",
    },
]

for item in items:
    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME,
        input=item["input"],
        expected_output=item["expected_output"],
    )

print(f"Dataset '{DATASET_NAME}' created with {len(items)} items.")
langfuse.flush()
print("Done!")
