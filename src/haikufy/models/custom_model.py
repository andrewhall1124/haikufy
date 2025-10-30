from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .language_model import LanguageModel

class CustomModel(LanguageModel):
    def __init__(self, model_name: str, api_token: str | None = None) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=api_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=api_token
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.model.to(self.device)

    def get_next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            return logits

    def apply_top_p_filtering(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Create a mask in the original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )

        # Zero out probabilities for removed tokens
        filtered_probs = probs.clone()
        filtered_probs[indices_to_remove] = 0.0

        # Renormalize (with safeguard against division by zero)
        prob_sum = filtered_probs.sum(dim=-1, keepdim=True)
        if prob_sum.item() > 0:
            filtered_probs = filtered_probs / prob_sum
        else:
            # If all probabilities were filtered out, return uniform distribution
            filtered_probs = torch.ones_like(probs) / probs.shape[-1]

        return filtered_probs

    def beam_search(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> tuple[torch.Tensor, int]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        beams = [(input_ids, 0.0)]

        beam_width = 5

        # Get stopping tokens (EOS and any chat template end tokens)
        stop_token_ids = [self.tokenizer.eos_token_id]
        # if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
        # stop_token_ids.append(self.tokenizer.eos_token_id)

        for _ in range(max_tokens):
            candidates = []

            for seq, score in beams:
                # Check if this sequence has already generated an EOS token
                if seq[0, -1].item() in stop_token_ids:
                    # Don't generate more tokens for completed sequences
                    candidates.append((seq, score))
                    continue

                # Get logits
                logits = self.get_next_token_logits(seq)

                # Apply temperature
                logits /= temperature

                # Get probs
                probs = torch.softmax(logits, dim=-1)

                # Apply top-p filtering
                if top_p is not None:
                    probs = self.apply_top_p_filtering(probs[0], top_p)
                    probs = probs.unsqueeze(0)  # Restore batch dimension

                # Get top k tokens from filtered probabilities
                top_probs, top_indices = torch.topk(
                    input=probs,
                    k=min(beam_width, (probs > 0).sum().item()),
                    dim=-1
                )

                # Generate new candidates
                for prob, index in zip(top_probs[0], top_indices[0]):
                    new_seq = torch.cat([seq, index.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + torch.log(prob).item()
                    candidates.append((new_seq, new_score))

            # Keep top beams (moved outside inner loop - this was the bug!)
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        # Get best beam
        best_seq, _ = beams[0]

        # Remove prompt tokens
        generated_token_ids = best_seq[0][prompt_length:]

        # Returns decoded result
        return self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        # Apply chat template to format the messages properly
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        result = self.beam_search(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        return result