import gc
import torch
from rich.console import Console

from haikufy.converter import HaikuConverter

console = Console()
prompt = "Do you want to run today?"

# Define decoding methods to test
decoding_configs = [
    ("Greedy", {"temperature": 0, "do_sample": False}, 'hf-local'),
    ("Beam Search", {"num_beams": 5}, 'hf-local'),
    ("Top-K Sampling", {"top_k": 50, "temperature": 0.7}, 'hf-local'),
    ("Top-P Sampling", {"top_p": 0.9, "temperature": 0.7}, 'hf-local'),
    ("Temperature Sampling", {"temperature": 1.0}, 'hf-local'),
    ("Custom Beam Search (top_p=0.9)", {"temperature": 0.7, "top_p": 0.9}, 'hf-custom'),
]

console.print("=" * 80)
console.print(f"[bold]Prompt:[/bold] {prompt}")
console.print("=" * 80)

for method_name, config, wrapper in decoding_configs:
    console.print(f"\n[bold cyan]Decoding Method: {method_name}[/bold cyan]")
    console.print(f"[dim]Wrapper: {wrapper}[/dim]")
    console.print("-" * 80)

    try:
        # Create converter with specific decoding configuration
        hc = HaikuConverter(
            model_wrapper=wrapper,
            generation_config=config
        )

        haiku_text, syllable_counts, is_valid = hc.generate_haiku(text=prompt)

        console.print(f"\n[green]{haiku_text}[/green]\n")
        console.print(f"Syllable counts: {syllable_counts}")
        console.print(f"Valid 5-7-5: {'✓' if is_valid else '✗'}")
        console.print(f"Config: {config}")

        # Clean up model to prevent OOM
        del hc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    console.print("-" * 80)
