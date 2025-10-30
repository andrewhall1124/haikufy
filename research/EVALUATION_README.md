# Model Evaluation Guide

This guide explains how to use the model evaluation script to compare your CustomModel against LocalHuggingFaceModel with various decoding methods.

## Overview

The evaluation script (`model_evaluation.py`) performs a comprehensive comparison of:

1. **CustomModel** - Your custom beam search implementation
2. **LocalHuggingFaceModel** - HuggingFace's built-in generation with multiple decoding methods:
   - Greedy decoding
   - Beam search
   - Top-K sampling
   - Top-P (nucleus) sampling
   - Temperature sampling

## Metrics Used

The script evaluates using the following metrics:

### 1. Perplexity (Lower is Better)
- Measures how well the model predicts the text
- Lower values indicate more confident/natural predictions

### 2. BLEU Score (0-100, Higher is Better)
- Measures n-gram overlap between generated and reference text
- Commonly used for machine translation evaluation

### 3. ROUGE Scores (0-100, Higher is Better)
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence
- Measures recall-oriented overlap with reference text

### 4. BERTScore (0-100, Higher is Better)
- Uses contextual embeddings to compare semantic similarity
- More robust to paraphrasing than n-gram metrics

### 5. MAUVE (0-100, Higher is Better)
- Measures the gap between generated and reference text distributions
- Evaluates overall text quality and diversity

## Installation

1. Install the required dependencies:

```bash
# Using pip
pip install -e .

# Or using uv (if you're using it)
uv pip install -e .
```

This will install all the necessary evaluation libraries including:
- `evaluate` - HuggingFace's evaluation library
- `bert-score` - BERTScore metric
- `mauve-text` - MAUVE metric
- `rouge-score` - ROUGE metrics
- `sacrebleu` - BLEU metric

## Usage

### Basic Usage

Run the evaluation script:

```bash
python research/model_evaluation.py
```

This will:
1. Initialize both models
2. Load evaluation metrics
3. Generate text with all decoding methods
4. Calculate metrics for each method
5. Display results in formatted tables
6. Save detailed results to a JSON file

### Output

The script provides three types of output:

#### 1. Average Metrics Table
Shows average performance across all test prompts for each method.

#### 2. MAUVE Scores
Shows distribution similarity between generated and reference texts.

#### 3. Sample Generations
Displays example generations from each method for inspection.

#### 4. JSON Results File
Detailed results saved to `research/evaluation_results_TIMESTAMP.json` containing:
- All generated texts
- Per-prompt metrics
- Average metrics
- Prompts and references used

## Customization

### Change the Model

Edit the model name in `model_evaluation.py`:

```python
evaluator = ModelEvaluator(model_name="your-model-name-here")
```

### Add More Test Prompts

Modify the `get_test_prompts()` method to add your own test cases:

```python
def get_test_prompts(self) -> List[Tuple[str, str]]:
    return [
        ("Your prompt here", "Expected reference text here"),
        # Add more...
    ]
```

### Adjust Generation Parameters

Modify the `configs` list in `evaluate_all_methods()`:

```python
configs = [
    ("custom", "beam_search", "Custom Model", {"temperature": 0.8, "top_p": 0.95}),
    # Adjust temperature, top_p, top_k, num_beams, etc.
]
```

## Understanding Your Custom Decoding Method

Your `CustomModel` implements a custom beam search with top-p filtering:

### Key Features:
1. **Beam Search**: Maintains multiple candidate sequences (beam_width=5)
2. **Temperature Scaling**: Controls randomness in token selection
3. **Top-P Filtering**: Nucleus sampling - only considers tokens in top-p cumulative probability
4. **Score-based Selection**: Uses log probabilities to score and rank beams

### How It Differs from Standard Methods:

- **vs Greedy**: Explores multiple paths instead of just the best token
- **vs Standard Beam Search**: Adds top-p filtering for more diverse outputs
- **vs Top-K**: Uses cumulative probability instead of fixed number of tokens
- **vs Pure Sampling**: More structured with beam search framework

## Interpreting Results

### Good Performance Indicators:
- **Low Perplexity**: < 50 is generally good, < 20 is excellent
- **High BLEU**: > 20 is decent, > 40 is good
- **High ROUGE-L**: > 30 is decent, > 50 is good
- **High BERTScore**: > 80 is decent, > 90 is good
- **High MAUVE**: > 80 indicates good distribution match

### What to Look For:
1. **Custom Model vs Local Beam Search**: Compare your implementation to HuggingFace's
2. **Diversity vs Quality**: Sampling methods may have lower metrics but more diverse outputs
3. **Trade-offs**: Beam search typically scores higher but may be less creative
4. **Consistency**: Look at variance across different prompts

## Report Writing Tips

For your assignment report, consider discussing:

1. **Method Comparison**:
   - Which decoding methods performed best?
   - What trade-offs did you observe?

2. **Custom Implementation**:
   - How does your beam search work?
   - What makes it unique?
   - How does it compare to HuggingFace's implementation?

3. **Metric Analysis**:
   - Which metrics aligned with human judgment?
   - Any surprising results?
   - Metric correlations?

4. **Examples**:
   - Include sample generations that illustrate differences
   - Show both successes and failures

5. **Conclusions**:
   - Best method for your use case?
   - Future improvements?

## Troubleshooting

### CUDA Out of Memory
If you run out of GPU memory, you can:
1. Reduce batch size (generate one at a time)
2. Use CPU instead: Set `device = torch.device("cpu")`
3. Use a smaller model

### Metric Calculation Errors
Some metrics may fail on very short or very long texts. The script catches these errors and reports them.

### Slow Evaluation
MAUVE can be slow. You can comment it out if needed:
```python
# mauve_score = self.calculate_mauve_score(generated_texts, reference_texts)
```

## Example Output

```
Model Evaluation Results - Average Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Method                      ┃ Perplexity ┃  BLEU ┃ ROUGE-1 ┃ ROUGE-L ┃ BERTScore  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ Custom Model (Beam Search)  │      24.35 │ 28.45 │   42.31 │   38.67 │      85.23 │
│ Local Model (Greedy)        │      26.78 │ 25.12 │   38.92 │   35.44 │      83.45 │
│ Local Model (Beam Search)   │      23.91 │ 29.87 │   43.56 │   39.88 │      86.12 │
...
```

## Additional Resources

- [HuggingFace Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [BLEU Score Explanation](https://en.wikipedia.org/wiki/BLEU)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [MAUVE Paper](https://arxiv.org/abs/2102.01454)
