import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(
    ROOT_DIR / ".env",
    override=True
)


class LLMClient:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing."
            )

        if not base_url:
            raise ValueError(
                "OPENAI_BASE_URL is missing."
            )

        if not model_name:
            raise ValueError(
                "OPENAI_MODEL is missing."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model_name = model_name

    def answer(
        self,
        query,
        results
    ):

        context_parts = []

        for i, r in enumerate(results, 1):

            context_parts.append(
                f"""
Source [{i}]
Paper: {r['source']}
Page: {r['page']}

Content:
{r['text']}
"""
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
You are a scientific research assistant specializing in
waste thermochemical conversion.

Answer the question based on the retrieved literature.

Rules:
1. Use only the provided context as evidence.
2. Cite the correct source number [1], [2], etc.
3. Do not invent references or experimental results.
4. Do not generalize conclusions from individual studies.
5. Distinguish hydrogen yield, hydrogen concentration,
   and total syngas yield.
6. Avoid incorrect scientific explanations.
   For example, the forward water-gas shift reaction
   is exothermic, not endothermic.
7. If evidence is insufficient, clearly state the limitation.
8. Keep the answer concise and scientifically accurate.


Context:
{context}

Question:
{query}
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
        print("实际调用模型:", response.model)
        print("本次 Token 用量:", response.usage)
        return response.choices[0].message.content