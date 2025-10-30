from abc import ABC, abstractmethod
from typing import Dict, List


class LanguageModel(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        pass
