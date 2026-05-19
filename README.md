# Multilingual Hallucination Evaluation in Large Language Models

## Overview

This project evaluates hallucination behavior in Large Language Models (LLMs) across Indian languages using semantic evaluation and mechanistic interpretability techniques.

The study compares Phi-4, Qwen, and LLaMA-2 across Hindi, Bengali, Telugu, Tamil, and Malayalam using the TruthfulQA benchmark.

---

## Key Contributions

- Multilingual hallucination evaluation framework
- Translation noise validation
- Entity consistency analysis
- Mechanistic interpretability analysis
- Cross-language hallucination comparison

---

## Models Evaluated

- Phi-4
- Qwen
- LLaMA-2

---

## Languages

### Indo-Aryan
- Hindi
- Bengali

### Dravidian
- Telugu
- Tamil
- Malayalam

---

## Evaluation Metrics

### Surface-Level
- Semantic Similarity
- Drift Score
- Hallucination Rate

### Mechanistic
- Attention Entropy
- Self-Attention Ratio
- Layer-wise Confidence

### Additional Validation
- Translation Noise Analysis
- Answer vs Translation Gap
- Entity Consistency

---

## Repository Structure

```text
data/              -> datasets
notebooks/         -> experiments
src/               -> implementation
figures/           -> graphs and plots
paper/             -> IEEE paper
```

---

## Key Finding

Translation introduces a uniform noise floor, but multilingual hallucination is primarily amplified by model-specific and language-family-specific effects.
