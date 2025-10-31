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
        **kwargs
    ) -> str:
        """
        Generate text with configurable decoding methods.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            **kwargs: Generation parameters (temperature, top_p, top_k, num_beams, do_sample)

        Decoding Methods:
            - Greedy: temperature=0 or do_sample=False, num_beams=1
            - Beam Search: num_beams > 1
            - Top-K: top_k=50, temperature > 0
            - Top-P: top_p=0.9, temperature > 0
            - Temperature Sampling: temperature > 0, no top_k/top_p
        """
        # Extract parameters (only use defaults if not provided)
        temperature = kwargs.get('temperature')
        top_p = kwargs.get('top_p')
        top_k = kwargs.get('top_k')
        num_beams = kwargs.get('num_beams', 1)
        do_sample = kwargs.get('do_sample')

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
            do_sample = (temperature is None or temperature > 0) and num_beams == 1

        # Beam search mode (deterministic)
        if num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["do_sample"] = False
            # Don't include sampling parameters for beam search
        # Greedy decoding (deterministic)
        elif do_sample is False or temperature == 0:
            gen_kwargs["do_sample"] = False
            # Don't include sampling parameters for greedy decoding
        # Sampling mode
        else:
            gen_kwargs["do_sample"] = True
            # Apply defaults for sampling parameters if not provided
            gen_kwargs["temperature"] = temperature if temperature is not None else 0.7
            gen_kwargs["top_p"] = top_p if top_p is not None else 0.9
            if top_k is not None:
                gen_kwargs["top_k"] = top_k

        # Generate
        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens (excluding the prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return result
