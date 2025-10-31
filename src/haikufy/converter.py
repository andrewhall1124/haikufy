import os
import re
from typing import Dict, List, Optional, Tuple
from haikufy.models.hugging_face_model import HuggingFaceModel
from haikufy.models.local_hugging_face_model import LocalHuggingFaceModel
from haikufy.models.custom_model import CustomModel

import pyphen
from dotenv import load_dotenv

load_dotenv(override=True)

LOGGING = False


class HaikuConverter:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct", # "deepseek-ai/DeepSeek-V3-0324"
        model_wrapper: str = 'hf-inference',
        api_token: Optional[str] = None,
        generation_config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the haiku converter with HuggingFace Inference Client

        Args:
            model_name: HuggingFace model to use
            model_wrapper: Which model wrapper to use
            api_token: HuggingFace API token (or set HF_TOKEN environment variable)
            generation_config: Optional generation parameters (temperature, top_p, top_k, num_beams, etc.)
        """
        if LOGGING:
            print(f"Initializing with model: {model_name}...")

        self.model_name = model_name
        self.api_token = api_token or os.getenv("HF_TOKEN")
        # Only set defaults if no config provided at all
        if generation_config is None:
            self.generation_config = {"temperature": 1, "top_p": 0.9}
        else:
            self.generation_config = generation_config

        if not self.api_token:
            raise ValueError(
                "HuggingFace API token required. Either pass api_token parameter "
                "or set HF_TOKEN environment variable. "
                "Get your token at: https://huggingface.co/settings/tokens"
            )

        match model_wrapper:
            case 'hf-inference':
                self.model = HuggingFaceModel(
                    model_name=self.model_name, api_token=self.api_token
                )
            case 'hf-local':
                self.model = LocalHuggingFaceModel(
                    model_name=self.model_name, api_token=self.api_token
                )
            case 'hf-custom':
                self.model = CustomModel(
                    model_name=self.model_name, api_token=self.api_token
                )

            case _:
                raise ValueError("Invalid model_wrapper type!")
                
        self.dic = pyphen.Pyphen(lang="en_US")

    def create_line_messages(
        self,
        text: str,
        line_number: int,
        previous_lines: Optional[List[str]] = None,
        previous_attempt: Optional[str] = None,
        actual_syllables: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Create the chat messages for generating a single haiku line"""
        target_syllables = {1: 5, 2: 7, 3: 5}[line_number]

        context = f"""You are a haiku poet. You are creating line {line_number} of a haiku about this message:
"{text}"

"""

        if previous_lines:
            context += "Previous lines:\n"
            for i, line in enumerate(previous_lines, 1):
                context += f"Line {i}: {line}\n"
            context += "\n"

        instruction = f"""Create line {line_number} with EXACTLY {target_syllables} syllables.

Rules:
- Count syllables carefully
- Must be exactly {target_syllables} syllables, no more, no less
- Make it flow naturally with the haiku theme"""

        if previous_attempt and actual_syllables is not None:
            instruction += f"""

Your previous attempt was: "{previous_attempt}"
This had {actual_syllables} syllables, but we need EXACTLY {target_syllables}.
Try again with a different phrasing."""

        instruction += "\n\nRespond with ONLY the line, nothing else."

        return [{"role": "user", "content": context + instruction}]

    def count_syllables_in_word(self, word: str) -> int:
        """
        Count syllables in a single word using pyphen with fallback heuristic.

        First tries dictionary-based hyphenation (pyphen), then falls back to
        a vowel-counting heuristic for words not in the dictionary.
        """
        # Remove punctuation and convert to lowercase
        clean_word = re.sub(r"[^\w]", "", word).lower()
        if not clean_word:
            return 0

        # Skip pure numbers
        if clean_word.isdigit():
            return 0

        # Try pyphen first (dictionary-based)
        hyphenated = self.dic.inserted(clean_word)

        # If pyphen inserted hyphens, it knows the word
        if "-" in hyphenated:
            return hyphenated.count("-") + 1

        # Fallback heuristic for words not in dictionary
        # This handles proper nouns, technical terms, etc.
        return self._count_syllables_heuristic(clean_word)

    def _count_syllables_heuristic(self, word: str) -> int:
        """
        Heuristic syllable counter based on vowel patterns.
        Used as fallback when pyphen doesn't recognize a word.
        """
        word = word.lower()

        # Handle empty or very short words
        if len(word) <= 1:
            return 1 if word else 0

        # Count vowel groups
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for i, char in enumerate(word):
            is_vowel = char in vowels

            if is_vowel and not previous_was_vowel:
                syllable_count += 1

            previous_was_vowel = is_vowel

        # Adjust for common patterns

        # Silent 'e' at the end (like "make", "time")
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        # Words ending in 'le' with consonant before (like "table", "riddle")
        if len(word) >= 3 and word.endswith('le') and word[-3] not in vowels:
            # If we already counted it, don't add; if not, add one
            if syllable_count == 0:
                syllable_count = 1

        # Words ending in 'ed' (past tense)
        # "walked" = 1 syllable, "created" = 3 syllables
        if word.endswith('ed') and syllable_count > 1:
            # Check if the 'ed' should be silent
            if len(word) >= 3 and word[-3] not in 'td':
                syllable_count -= 1

        # Ensure at least 1 syllable
        return max(1, syllable_count)

    def count_syllables_in_line(self, line: str) -> int:
        """Count syllables in a line of text"""
        words = line.strip().split()
        total = 0
        for word in words:
            syllable_count = self.count_syllables_in_word(word)
            total += syllable_count

        return total

    def validate_haiku(self, haiku_text: str) -> Tuple[bool, List[int]]:
        """
        Check if the haiku follows 5-7-5 pattern
        Returns: (is_valid, syllable_counts)
        """
        lines = [
            line.strip() for line in haiku_text.strip().split("\n") if line.strip()
        ]

        if len(lines) != 3:
            return False, []

        counts = [self.count_syllables_in_line(line) for line in lines]
        is_valid = counts == [5, 7, 5]

        return is_valid, counts

    def generate_line(
        self,
        text: str,
        line_number: int,
        previous_lines: Optional[List[str]] = None,
        max_attempts: int = 5,
    ) -> Tuple[str, int, bool]:
        """
        Generate a single haiku line with the correct syllable count
        Uses a feedback loop to retry if syllable count is incorrect
        """
        target_syllables = {1: 5, 2: 7, 3: 5}[line_number]
        previous_attempt = None
        actual_syllables = None

        for attempt in range(max_attempts):
            messages = self.create_line_messages(
                text, line_number, previous_lines, previous_attempt, actual_syllables
            )

            try:
                completion = self.model.generate(
                    messages=messages,
                    max_tokens=50,
                    **self.generation_config
                )

                # Clean up any quotes or extra formatting
                generated_line = completion.strip('"').strip("'").strip()

            except Exception as e:
                print(f"  API request failed: {e}")
                if attempt < max_attempts - 1:
                    print("  Retrying...")
                    continue
                else:
                    raise

            # Count syllables
            actual_syllables = self.count_syllables_in_line(generated_line)

            if LOGGING:
                print(
                    f'  Attempt {attempt + 1}: "{generated_line}" ({actual_syllables} syllables)',
                    end="",
                )

            if actual_syllables == target_syllables:
                if LOGGING:
                    print(" ✓")
                return generated_line, actual_syllables, True
            else:
                if LOGGING:
                    print(f" ✗ (need {target_syllables})")
                previous_attempt = generated_line

        # Return best attempt even if not valid
        if LOGGING:
            print(
                f"Could not achieve {target_syllables} syllables after {max_attempts} attempts"
            )
        return generated_line, actual_syllables, False

    def generate_haiku(
        self, text: str, max_line_attempts: int = 5
    ) -> Tuple[str, List[int], bool]:
        """
        Generate a haiku line by line using feedback loops
        Each line is validated and regenerated until it has the correct syllable count
        """

        lines = []
        syllable_counts = []

        # Generate each line individually
        for line_num in [1, 2, 3]:
            target = {1: 5, 2: 7, 3: 5}[line_num]

            if LOGGING:
                print(f"\nLine {line_num} (target: {target} syllables):")

            line, count, _ = self.generate_line(
                text, line_num, previous_lines=lines, max_attempts=max_line_attempts
            )

            lines.append(line)
            syllable_counts.append(count)

        # Capitalize first word of each line
        capitalized_lines = []
        for line in lines:
            if line:
                # Capitalize the first character of the line
                capitalized_line = (
                    line[0].upper() + line[1:] if len(line) > 1 else line.upper()
                )
                capitalized_lines.append(capitalized_line)
            else:
                capitalized_lines.append(line)

        # Join lines and ensure no trailing newlines
        haiku = "\n".join(capitalized_lines).rstrip()
        is_valid = syllable_counts == [5, 7, 5]

        return haiku, syllable_counts, is_valid
