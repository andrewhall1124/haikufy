"""
Model Evaluation Script
Compares CustomModel (beam search) vs LocalHuggingFaceModel with different decoding methods.
Evaluates using multiple metrics: Perplexity, BLEU, ROUGE, BERTScore, and MAUVE.

Memory-efficient implementation:
- Models are loaded one at a time to avoid OOM errors
- Perplexity model is loaded/unloaded as needed
- GPU memory is cleared between model switches
"""

from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Evaluation metrics
from evaluate import load
import numpy as np

from haikufy.models.custom_model import CustomModel
from haikufy.models.local_hugging_face_model import LocalHuggingFaceModel


class ModelEvaluator:
    """Evaluates text generation models with various decoding methods and metrics."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.console = Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

        # Don't initialize models here - load them one at a time during evaluation
        self.current_model = None
        self.current_model_type = None
        self.model_to_reload = None

        # Initialize metrics - using lightweight metrics only
        self.console.print("[bold blue]Using BLEU and ROUGE metrics (lightweight)...[/bold blue]")

    def get_test_prompts(self) -> List[Tuple[str, str]]:
        """
        Returns a list of (prompt, reference) pairs for evaluation.
        Reference texts are human-written or high-quality examples.
        """
        return [
            (
                "The benefits of renewable energy include",
                "reducing carbon emissions, creating sustainable jobs, decreasing dependence on fossil fuels, and protecting the environment for future generations."
            ),
            (
                "Artificial intelligence is transforming",
                "healthcare, education, transportation, and communication by enabling faster decision-making, personalized experiences, and automation of complex tasks."
            ),
            (
                "The most important skills for the future are",
                "critical thinking, adaptability, digital literacy, emotional intelligence, and continuous learning to navigate rapid technological changes."
            ),
            (
                "Climate change impacts include",
                "rising sea levels, extreme weather events, ecosystem disruption, food security challenges, and displacement of communities worldwide."
            ),
            (
                "The key to effective communication is",
                "active listening, clear expression, empathy, understanding context, and adapting your message to your audience's needs and perspectives."
            ),
            (
                "Innovation in technology requires",
                "creativity, collaboration, risk-taking, continuous experimentation, and learning from both successes and failures to drive meaningful progress."
            ),
            (
                "Healthy living involves",
                "regular exercise, balanced nutrition, adequate sleep, stress management, social connections, and maintaining both physical and mental well-being."
            ),
            (
                "The future of work will emphasize",
                "flexibility, remote collaboration, lifelong learning, human-AI partnership, and skills that complement automation rather than compete with it."
            ),
        ]


    def _load_model(self, model_type: str):
        """Load model if not already loaded or if switching model types."""
        if self.current_model is None or self.current_model_type != model_type:
            # Cleanup previous model
            if self.current_model is not None:
                self._cleanup_current_model()

            # Load new model
            self.console.print(f"[bold blue]Loading {model_type} model...[/bold blue]")
            if model_type == "custom":
                self.current_model = CustomModel(model_name=self.model_name)
            elif model_type == "local":
                self.current_model = LocalHuggingFaceModel(model_name=self.model_name)

            self.current_model_type = model_type

    def _cleanup_current_model(self):
        """Cleanup current model to free memory."""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_with_method(
        self,
        model_type: str,
        decoding_method: str,
        prompt: str,
        max_tokens: int = 50,
        **kwargs
    ) -> str:
        """
        Generate text using specified model and decoding method.

        Args:
            model_type: "custom" or "local"
            decoding_method: "beam_search", "greedy", "top_k", "top_p", "sampling"
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for generation
        """
        self._load_model(model_type)
        messages = [{"role": "user", "content": prompt}]

        if model_type == "custom":
            # CustomModel uses beam search by default
            return self.current_model.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9)
            )

        elif model_type == "local":
            # LocalHuggingFaceModel - we can configure different decoding methods
            return self._generate_local_with_method(
                decoding_method=decoding_method,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs
            )

    def _generate_local_with_method(
        self,
        decoding_method: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate text with LocalHuggingFaceModel using different decoding methods."""
        prompt = self.current_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.current_model.tokenizer(prompt, return_tensors="pt").to(self.current_model.device)

        # Configure generation parameters based on decoding method
        # Only include parameters that are valid for each method
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.current_model.tokenizer.pad_token_id,
        }

        if decoding_method == "greedy":
            # Greedy: no sampling, single beam, no temperature/top_p
            gen_kwargs.update({
                "do_sample": False,
                "num_beams": 1,
            })

        elif decoding_method == "beam_search":
            # Beam search: no sampling, multiple beams, no temperature/top_p
            gen_kwargs.update({
                "do_sample": False,
                "num_beams": kwargs.get("num_beams", 5),
            })

        elif decoding_method == "top_k":
            # Top-K sampling: needs temperature
            gen_kwargs.update({
                "do_sample": True,
                "top_k": kwargs.get("top_k", 50),
                "temperature": kwargs.get("temperature", 0.7),
            })

        elif decoding_method == "top_p":
            # Top-P (nucleus) sampling: needs temperature
            gen_kwargs.update({
                "do_sample": True,
                "top_p": kwargs.get("top_p", 0.9),
                "temperature": kwargs.get("temperature", 0.7),
            })

        elif decoding_method == "sampling":
            # Temperature sampling only
            gen_kwargs.update({
                "do_sample": True,
                "temperature": kwargs.get("temperature", 1.0),
            })

        with torch.no_grad():
            # Pass input_ids and attention_mask separately, then generation kwargs
            outputs = self.current_model.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                **gen_kwargs
            )
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            result = self.current_model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up tensors immediately
        del inputs, outputs, generated_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def evaluate_generation(
        self,
        generated_text: str,
        reference_text: str
    ) -> Dict[str, float]:
        """Calculate BLEU and ROUGE metrics for a single generation."""
        metrics = {}

        # 1. BLEU Score (0-100, higher is better)
        try:
            from sacrebleu import corpus_bleu
            bleu_score = corpus_bleu([generated_text], [[reference_text]])
            metrics["bleu"] = bleu_score.score
        except Exception as e:
            print(f"[yellow]Warning: BLEU calculation failed: {e}[/yellow]")
            metrics["bleu"] = 0.0

        # 2. ROUGE Scores (0-100, higher is better)
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference_text, generated_text)
            metrics["rouge1"] = scores['rouge1'].fmeasure * 100
            metrics["rouge2"] = scores['rouge2'].fmeasure * 100
            metrics["rougeL"] = scores['rougeL'].fmeasure * 100
        except Exception as e:
            print(f"[yellow]Warning: ROUGE calculation failed: {e}[/yellow]")
            metrics["rouge1"] = metrics["rouge2"] = metrics["rougeL"] = 0.0

        return metrics

    def evaluate_all_methods(self) -> Dict:
        """Run comprehensive evaluation comparing all methods."""
        test_data = self.get_test_prompts()

        # Define configurations to test
        configs = [
            ("custom", "beam_search", "Custom Model (Beam Search)", {"temperature": 0.7, "top_p": 0.9}),
            ("local", "greedy", "Local Model (Greedy)", {}),
            ("local", "beam_search", "Local Model (Beam Search)", {"num_beams": 5}),
            ("local", "top_k", "Local Model (Top-K)", {"top_k": 50, "temperature": 0.7}),
            ("local", "top_p", "Local Model (Top-P)", {"top_p": 0.9, "temperature": 0.7}),
            ("local", "sampling", "Local Model (Sampling)", {"temperature": 1.0}),
        ]

        results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:

            for config_idx, (model_type, method, name, params) in enumerate(configs):
                task = progress.add_task(f"[cyan]Evaluating {name}...", total=len(test_data))

                method_results = {
                    "generations": [],
                    "metrics": []
                }

                for prompt_idx, (prompt, reference) in enumerate(test_data):
                    # Generate text
                    generated = self.generate_with_method(
                        model_type=model_type,
                        decoding_method=method,
                        prompt=prompt,
                        max_tokens=50,
                        **params
                    )

                    # Calculate metrics
                    metrics = self.evaluate_generation(generated, reference)

                    method_results["generations"].append({
                        "prompt": prompt,
                        "reference": reference,
                        "generated": generated,
                        "metrics": metrics
                    })
                    method_results["metrics"].append(metrics)

                    progress.update(task, advance=1)

                # Calculate average metrics
                avg_metrics = {}
                metric_keys = method_results["metrics"][0].keys()
                for key in metric_keys:
                    values = [m[key] for m in method_results["metrics"] if m[key] != float('inf')]
                    avg_metrics[key] = np.mean(values) if values else float('inf')

                method_results["average_metrics"] = avg_metrics
                results[name] = method_results

                progress.remove_task(task)

                # Cleanup current model before loading the next one
                # (except if the next config uses the same model type)
                if config_idx < len(configs) - 1:
                    next_model_type = configs[config_idx + 1][0]
                    if next_model_type != model_type:
                        self._cleanup_current_model()
                        self.console.print(f"[dim]Cleaned up {model_type} model to free memory[/dim]")

        # Final cleanup
        self._cleanup_current_model()

        return results


    def print_results(self, results: Dict):
        """Print formatted results table."""
        # Create summary table
        table = Table(title="Model Evaluation Results - Average Metrics", show_header=True, header_style="bold magenta")

        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("BLEU", justify="right", style="green")
        table.add_column("ROUGE-1", justify="right", style="green")
        table.add_column("ROUGE-2", justify="right", style="green")
        table.add_column("ROUGE-L", justify="right", style="green")

        for method_name, method_data in results.items():
            metrics = method_data["average_metrics"]
            table.add_row(
                method_name,
                f"{metrics.get('bleu', 0):.2f}",
                f"{metrics.get('rouge1', 0):.2f}",
                f"{metrics.get('rouge2', 0):.2f}",
                f"{metrics.get('rougeL', 0):.2f}",
            )

        self.console.print(table)

    def save_results(self, results: Dict, filename: str = None):
        """Save detailed results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"

        filepath = f"/home/amh/Projects/haikufy/research/{filename}"

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        self.console.print(f"\n[bold green]Results saved to: {filepath}[/bold green]")

    def print_sample_generations(self, results: Dict, num_samples: int = 2):
        """Print sample generations for inspection."""
        self.console.print("\n[bold magenta]Sample Generations[/bold magenta]\n")

        for method_name, method_data in results.items():
            self.console.print(f"[bold cyan]{method_name}[/bold cyan]")

            for i, gen_data in enumerate(method_data["generations"][:num_samples]):
                self.console.print(f"\n[yellow]Prompt {i+1}:[/yellow] {gen_data['prompt']}")
                self.console.print(f"[green]Generated:[/green] {gen_data['generated']}")
                self.console.print(f"[blue]Reference:[/blue] {gen_data['reference']}")

            self.console.print("\n" + "-" * 80 + "\n")


def main():
    """Run the model evaluation."""
    console = Console()

    console.print("[bold green]Starting Model Evaluation[/bold green]\n")
    console.print("This will compare:")
    console.print("  1. CustomModel with custom beam search implementation")
    console.print("  2. LocalHuggingFaceModel with multiple decoding methods:")
    console.print("     - Greedy decoding")
    console.print("     - Beam search")
    console.print("     - Top-K sampling")
    console.print("     - Top-P (nucleus) sampling")
    console.print("     - Temperature sampling")
    console.print("\nMetrics: BLEU, ROUGE (lightweight metrics only)\n")

    # Initialize evaluator
    evaluator = ModelEvaluator(model_name="meta-llama/Llama-3.2-1B-Instruct")

    # Run evaluation
    results = evaluator.evaluate_all_methods()

    # Display results
    evaluator.print_results(results)
    evaluator.print_sample_generations(results, num_samples=2)

    # Save results
    evaluator.save_results(results)

    console.print("\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
