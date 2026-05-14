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
