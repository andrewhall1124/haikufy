from typing import Dict, List

from huggingface_hub import InferenceClient

from src.haikufy.models.language_model import LanguageModel


class HuggingFaceModel(LanguageModel):
    def __init__(self, model_name: str, api_token: str) -> None:
        self.model_name = model_name
        self.client = InferenceClient(token=api_token)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        return (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            .choices[0]
            .message.content.strip()
        )
