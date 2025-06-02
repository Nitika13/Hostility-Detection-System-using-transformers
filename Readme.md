# Hostility Detection System (Hindi-English, Multi-Label NLP)

A multi-label classification system to detect **hostile content** such as `hate`, `offensive`, `defamation`, and `fake` in **Hindi-English (Hinglish)** social media text using transformer-based models like RoBERTa.

---

## Features

-  Supports multilingual Hindi-English inputs
-  Multi-label classification (`hate`, `offensive`, `defamation`, `fake`)
-  Trained using Hugging Face Transformers
-  Deployed with Gradio UI for interactive testing
-  Includes model export & reuse

---

## Dataset

- **Languages**: Hindi, English, Hinglish (code-mixed)
- **Splits**: `train.csv`, `validation.csv`, `test.csv`
- **Format**:
  - `text`: input string
  - `labels`: list of hostile tags (can be multiple per row)

---

## Model Architecture

- Base: [`RoBERTa-base`](https://huggingface.co/roberta-base) (or [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base) for multilingual support)
- Classification head: Sigmoid-activated dense layer for multi-label prediction

---

## Training

Trained using Hugging Face `Trainer`:

```python
from transformers import Trainer, TrainingArguments
