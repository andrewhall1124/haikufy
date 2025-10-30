from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.haikufy.models.language_model import LanguageModel


class LocalHuggingFaceModel(LanguageModel):
    def __init__(self, model_name: str, api_token: str | None = None) -> None:
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=api_token
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=api_token
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.model.to(self.device)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode only the new tokens (excluding the prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return result
