"""
Model Evaluation Script
Compares CustomModel (beam search) vs LocalHuggingFaceModel with different decoding methods.
Evaluates using multiple metrics: Perplexity, BLEU, ROUGE, BERTScore, and MAUVE.
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

        # Initialize models
        self.console.print("[bold blue]Initializing models...[/bold blue]")
        self.custom_model = CustomModel(model_name=model_name)
        self.local_model = LocalHuggingFaceModel(model_name=model_name)

        # Initialize metrics
        self.console.print("[bold blue]Loading evaluation metrics...[/bold blue]")
        self.bleu = load("bleu")
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
        self.mauve = load("mauve")

        # For perplexity calculation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.perplexity_model.to(device)
        self.perplexity_model.eval()

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

    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for generated text."""
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.perplexity_model.device)

        with torch.no_grad():
            outputs = self.perplexity_model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity

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
        messages = [{"role": "user", "content": prompt}]

        if model_type == "custom":
            # CustomModel uses beam search by default
            return self.custom_model.generate(
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
        prompt = self.local_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.local_model.tokenizer(prompt, return_tensors="pt").to(self.local_model.device)

        # Configure generation parameters based on decoding method
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.local_model.tokenizer.pad_token_id,
        }

        if decoding_method == "greedy":
            gen_kwargs.update({
                "do_sample": False,
                "num_beams": 1,
            })

        elif decoding_method == "beam_search":
            gen_kwargs.update({
                "do_sample": False,
                "num_beams": kwargs.get("num_beams", 5),
            })

        elif decoding_method == "top_k":
            gen_kwargs.update({
                "do_sample": True,
                "top_k": kwargs.get("top_k", 50),
                "temperature": kwargs.get("temperature", 0.7),
            })

        elif decoding_method == "top_p":
            gen_kwargs.update({
                "do_sample": True,
                "top_p": kwargs.get("top_p", 0.9),
                "temperature": kwargs.get("temperature", 0.7),
            })

        elif decoding_method == "sampling":
            gen_kwargs.update({
                "do_sample": True,
                "temperature": kwargs.get("temperature", 1.0),
            })

        outputs = self.local_model.model.generate(**inputs, **gen_kwargs)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        result = self.local_model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return result

    def evaluate_generation(
        self,
        generated_text: str,
        reference_text: str
    ) -> Dict[str, float]:
        """Calculate multiple metrics for a single generation."""
        metrics = {}

        # 1. Perplexity (lower is better)
        try:
            metrics["perplexity"] = self.calculate_perplexity(generated_text)
        except Exception as e:
            print(f"[yellow]Warning: Perplexity calculation failed: {e}[/yellow]")
            metrics["perplexity"] = float('inf')

        # 2. BLEU Score (0-100, higher is better)
        try:
            bleu_result = self.bleu.compute(
                predictions=[generated_text],
                references=[[reference_text]]
            )
            metrics["bleu"] = bleu_result["bleu"] * 100
        except Exception as e:
            print(f"[yellow]Warning: BLEU calculation failed: {e}[/yellow]")
            metrics["bleu"] = 0.0

        # 3. ROUGE Scores (0-1, higher is better)
        try:
            rouge_result = self.rouge.compute(
                predictions=[generated_text],
                references=[reference_text]
            )
            metrics["rouge1"] = rouge_result["rouge1"] * 100
            metrics["rouge2"] = rouge_result["rouge2"] * 100
            metrics["rougeL"] = rouge_result["rougeL"] * 100
        except Exception as e:
            print(f"[yellow]Warning: ROUGE calculation failed: {e}[/yellow]")
            metrics["rouge1"] = metrics["rouge2"] = metrics["rougeL"] = 0.0

        # 4. BERTScore (0-1, higher is better)
        try:
            bertscore_result = self.bertscore.compute(
                predictions=[generated_text],
                references=[reference_text],
                lang="en"
            )
            metrics["bertscore_f1"] = bertscore_result["f1"][0] * 100
        except Exception as e:
            print(f"[yellow]Warning: BERTScore calculation failed: {e}[/yellow]")
            metrics["bertscore_f1"] = 0.0

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

            for model_type, method, name, params in configs:
                task = progress.add_task(f"[cyan]Evaluating {name}...", total=len(test_data))

                method_results = {
                    "generations": [],
                    "metrics": []
                }

                for prompt, reference in test_data:
                    # Generate text
                    generated = self.generate_with_method(
                        model_type=model_type,
                        decoding_method=method,
                        prompt=prompt,
                        max_tokens=50,
                        **params
                    )

                    # Evaluate
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

        return results

    def calculate_mauve_score(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> float:
        """
        Calculate MAUVE score comparing generated vs reference texts.
        MAUVE measures the gap between distributions (0-1, higher is better).
        """
        try:
            mauve_result = self.mauve.compute(
                predictions=generated_texts,
                references=reference_texts
            )
            return mauve_result.mauve * 100
        except Exception as e:
            print(f"[yellow]Warning: MAUVE calculation failed: {e}[/yellow]")
            return 0.0

    def print_results(self, results: Dict):
        """Print formatted results table."""
        # Create summary table
        table = Table(title="Model Evaluation Results - Average Metrics", show_header=True, header_style="bold magenta")

        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Perplexity", justify="right", style="yellow")
        table.add_column("BLEU", justify="right", style="green")
        table.add_column("ROUGE-1", justify="right", style="green")
        table.add_column("ROUGE-L", justify="right", style="green")
        table.add_column("BERTScore", justify="right", style="blue")

        for method_name, method_data in results.items():
            metrics = method_data["average_metrics"]
            table.add_row(
                method_name,
                f"{metrics.get('perplexity', 0):.2f}",
                f"{metrics.get('bleu', 0):.2f}",
                f"{metrics.get('rouge1', 0):.2f}",
                f"{metrics.get('rougeL', 0):.2f}",
                f"{metrics.get('bertscore_f1', 0):.2f}",
            )

        self.console.print(table)

        # Calculate MAUVE scores for each method
        self.console.print("\n[bold magenta]MAUVE Scores (Distribution Similarity)[/bold magenta]")
        test_data = self.get_test_prompts()
        reference_texts = [ref for _, ref in test_data]

        mauve_table = Table(show_header=True, header_style="bold magenta")
        mauve_table.add_column("Method", style="cyan")
        mauve_table.add_column("MAUVE Score", justify="right", style="blue")

        for method_name, method_data in results.items():
            generated_texts = [gen["generated"] for gen in method_data["generations"]]
            mauve_score = self.calculate_mauve_score(generated_texts, reference_texts)
            mauve_table.add_row(method_name, f"{mauve_score:.2f}")

        self.console.print(mauve_table)

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
    console.print("\nMetrics: Perplexity, BLEU, ROUGE, BERTScore, MAUVE\n")

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
