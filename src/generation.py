"""
generation.py
-------------
Hypothetical document generation strategies used in HypoSelectSimRAG.

Two strategies:
  - Few-shot prompting: provide example QA pairs to guide format and style
  - Question-oriented prompting: classify query type first, then use a type-specific template

Each strategy is called with two temperature values (0.1 and 0.9) to produce
conservative and exploratory variants, giving four paths total.
"""

# ---------------------------------------------------------------------------
# Few-shot examples (injected into the prompt as in-context demonstrations)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """Example 1:
Question: What caused the 2008 financial crisis?
Passage: The 2008 financial crisis stemmed from the collapse of the US housing bubble, \
driven by subprime mortgage lending, insufficient financial regulation, and complex \
instruments like mortgage-backed securities that spread risk across the global banking system.

Example 2:
Question: How does photosynthesis work?
Passage: Photosynthesis is the process by which plants convert sunlight, carbon dioxide, \
and water into glucose and oxygen. It occurs in two stages inside chloroplasts: the \
light-dependent reactions (which capture solar energy) and the Calvin cycle \
(which uses that energy to build sugar molecules)."""

# ---------------------------------------------------------------------------
# Question-type templates
# ---------------------------------------------------------------------------
QUESTION_TYPE_TEMPLATES = {
    "A": "Find a concise excerpt that directly answers:\nQuestion: {q}\nPassage:",
    "B": "Write a comprehensive multi-paragraph answer for:\nQuestion: {q}\nAnswer:",
    "C": "Write a factual statement confirming or refuting:\nClaim: {q}\nEvidence:",
    "D": "Write a helpful, conversational response to:\nUser: {q}\nBot:",
    "H": "Write a comparison paragraph for:\nQuestion: {q}\nComparison:",
    "J": "Provide numeric reasoning to answer:\nQuestion: {q}\nAnswer:",
    "X": "Write a helpful passage answering:\nQuestion: {q}\nPassage:",
}

CLASSIFICATION_PROMPT = """Classify this question into one category. Output only the letter.

A. ExtractiveQA   — factual, answer can be pulled directly from a passage
B. Complex/LongQA — requires multi-step reasoning or a long response
C. FactCheck      — verify whether a claim is true or false
D. Conversation   — casual chat or simple instruction
H. ComparativeQA  — compare two or more entities or options
J. NumericalQA    — requires numbers, calculations, or statistics
X. Other          — does not fit any of the above

Question: {q}
Category:"""


def generate_few_shot_doc(question, client, model="moonshot-v1-8k", temperature=0.1):
    """
    Generate a hypothetical document using few-shot prompting.

    Provides two QA examples so the LLM learns the expected format before
    generating a passage for the target question.
    """
    prompt = f"{FEW_SHOT_EXAMPLES}\n\nNow write a passage for:\nQuestion: {question}\nPassage:"

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def classify_question(question, client, model="moonshot-v1-8k"):
    """
    Classify a question into one of the predefined types (A–J, X).
    Returns a single uppercase letter.
    """
    prompt = CLASSIFICATION_PROMPT.format(q=question)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,    # always deterministic for classification
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    # Take the first character in case the model returns extra text
    label = raw[0].upper() if raw else "X"
    return label if label in QUESTION_TYPE_TEMPLATES else "X"


def generate_question_oriented_doc(question, client, model="moonshot-v1-8k", temperature=0.1):
    """
    Generate a hypothetical document using question-oriented prompting.

    First classifies the question type, then selects the appropriate
    prompt template for more targeted generation.
    """
    q_type = classify_question(question, client, model)
    template = QUESTION_TYPE_TEMPLATES[q_type]
    prompt = template.format(q=question)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def generate_four_paths(question, client, model="moonshot-v1-8k"):
    """
    Generate all four hypothetical documents.

    Returns a list of four strings:
        [few_shot_low_temp, few_shot_high_temp,
         question_oriented_low_temp, question_oriented_high_temp]

    Note: Temperature is capped at 0.9 for compatibility with Kimi and other
    models that do not support temperature > 1.0. The original paper used 1.1
    (GPT-3.5-turbo specific). See README for discussion of this difference.
    """
    return [
        generate_few_shot_doc(question, client, model, temperature=0.1),
        generate_few_shot_doc(question, client, model, temperature=0.9),
        generate_question_oriented_doc(question, client, model, temperature=0.1),
        generate_question_oriented_doc(question, client, model, temperature=0.9),
    ]
