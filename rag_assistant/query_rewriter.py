from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

REWRITE_PROMPT = """You are a query rewriter for a document retrieval system.
Rewrite the user's question to be more specific and retrieval-friendly.
Expand vague pronouns, add relevant context, and make implicit entities explicit.
Return ONLY the rewritten query, nothing else.

Examples:
- "tell me about sudheer" → "What is Sudheer Kumar Reddy Batthina's professional background, work experience, technical skills, and projects?"
- "what about hipaa" → "What are the key provisions and requirements of the HIPAA Privacy Rule?"
- "who is a business associate" → "How does HIPAA define a business associate and what are their obligations?"
"""


def rewrite_query(question: str) -> str:
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=150,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question  # fallback to original on any error


def rewrite_query_with_history(question: str, history: list[dict]) -> str:
    """Rewrite query using conversation history for follow-up resolution."""
    if not history:
        return rewrite_query(question)  # fallback to simple rewrite

    # Build a short history string (last 3 turns max)
    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200]  # truncate long answers
        history_text += f"{role}: {content}\n"

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a query rewriter for a document retrieval system.
Given a conversation history and a new user question, rewrite the question to be fully self-contained and retrieval-friendly.
Resolve pronouns, expand vague references, and make implicit context explicit.
Return ONLY the rewritten query, nothing else.

Examples:
- History: "User: What is HIPAA?" / "Assistant: HIPAA is..."
  Question: "tell me more about it" → "What are the detailed provisions and requirements of the HIPAA Privacy Rule?"
- History: "User: Who is Sudheer?" / "Assistant: Sudheer is an ML engineer..."
  Question: "what projects did he build?" → "What projects has Sudheer Kumar Reddy Batthina built in his career?"
"""},
                {"role": "user", "content": f"Conversation history:\n{history_text}\n\nNew question: {question}"}
            ],
            temperature=0,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip() or question
    except Exception:
        return rewrite_query(question)
