from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .language_model import LanguageModel


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
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
        num_beams: int = 1,
        do_sample: bool | None = None,
    ) -> str:
        """
        Generate text with configurable decoding methods.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-K sampling threshold (optional)
            num_beams: Number of beams for beam search (1 = no beam search)
            do_sample: Explicit sampling control (auto-determined if None)

        Decoding Methods:
            - Greedy: temperature=0 or do_sample=False, num_beams=1
            - Beam Search: num_beams > 1
            - Top-K: top_k=50, temperature > 0
            - Top-P: top_p=0.9, temperature > 0
            - Temperature Sampling: temperature > 0, no top_k/top_p
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Configure generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Auto-determine sampling if not explicitly set
        if do_sample is None:
            do_sample = temperature > 0 and num_beams == 1

        # Beam search mode (deterministic)
        if num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["do_sample"] = False
        # Sampling mode
        else:
            gen_kwargs["do_sample"] = do_sample
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                if top_k is not None:
                    gen_kwargs["top_k"] = top_k

        # Generate
        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens (excluding the prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return result
