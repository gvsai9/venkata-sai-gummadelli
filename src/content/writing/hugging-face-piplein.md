---
title: "Hugging face pipline()"
description: "Some idea on pipeline in HF."
pubDate: 2026-09-09
category: "AI"
tags:
  - NLP
  - AI
  - Transformers
thumbnail: ../../assets/writing/hfp.png
thumbnailAlt: "Description of the image"
---
# Hugging Face Pipelines: Internal Working, Types, and Task Examples

## What is a Pipeline and Why Use It?

A **pipeline** in Hugging Face `transformers` is a high-level API that bundles together:

- Preprocessing (tokenization, feature extraction, image processing)
- Model inference
- Postprocessing (decoding, formatting, label mapping)

into a single, easy-to-use function.

**Use:**  
You call one function and get ready-to-use predictions for common tasks (classification, generation, QA, vision, audio, etc.) without manually handling tokenizers, tensors, or output parsing.

---

## Internal Working of a Pipeline

```text
                Pipeline
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   Preprocess    Model      Postprocess
       │           │           │
   Tokenizer       │       Formatting
   Processor       │       Labels
   Feature         │       Scores
   extraction      │       etc.
```

### 1. Preprocess

- **Text tasks:**
  - Tokenizer converts raw text to token IDs.
  - Adds special tokens (`[CLS]`, `[SEP]`, `<bos>`, `<eos>`, etc.).
  - Applies padding/truncation to a fixed length.
  - Builds attention masks and sometimes segment IDs.

- **Image tasks:**
  - Image processor resizes, normalizes, and converts images to tensors.
  - Applies model-specific transforms (e.g., mean/std normalization).

- **Audio tasks:**
  - Audio processor resamples audio, computes spectrograms or raw waveforms, etc.

Output: a dictionary of tensors ready for the model (`input_ids`, `attention_mask`, `pixel_values`, `input_features`, etc.).

### 2. Model

- The pipeline loads an appropriate pretrained model for the task:
  - `AutoModelForSequenceClassification`
  - `AutoModelForCausalLM`
  - `AutoModelForQuestionAnswering`
  - `AutoModelForImageClassification`, etc.
- Runs a forward pass:
  - Computes logits, hidden states, or other task-specific outputs.
- Handles device placement (CPU/GPU) and batching.

### 3. Postprocess

- Converts raw model outputs into human-friendly results:
  - Maps label IDs to label names (`LABEL_0` → `POSITIVE`, etc.).
  - Applies softmax/sigmoid to get probabilities.
  - Decodes token IDs back to text (for generation, translation, summarization).
  - Formats bounding boxes, masks, segments for vision tasks.
  - Aggregates scores, picks top-k results, etc.

Output: simple Python structures (dicts, lists) with labels, scores, text, boxes, etc.

---

## Different Ways to Create a Pipeline

### 1. Normal Way (Task String Only)

Hugging Face automatically picks a default model for that task.

```python
from transformers import pipeline

pipe = pipeline("text-classification")
result = pipe("I love this movie!")
```

- **Use case in one line:** Quick experiments and demos with sensible defaults.

---

### 2. With a Specific Model

You specify the model; the pipeline infers the tokenizer/processor.

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = pipe("I love this movie!")
```

- **Use case in one line:** Use a particular pretrained model for a known task.

---

### 3. With Model + Tokenizer (or Processor)

You explicitly load model and tokenizer/processor, then pass them to the pipeline.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = pipe("I love this movie!")
```

- **Use case in one line:** Fine-grained control (custom tokenizer config, modified model, special preprocessing).

---

## Task-wise Pipeline Examples (with One-liner Use Cases)

Below are minimal “glue code” snippets for each task using `pipeline`. Adjust model names as needed.

---

### 1. Text Classification

**Use case:** Predict sentiment or topic of a sentence.

```python
from transformers import pipeline

pipe = pipeline("text-classification")
out = pipe("I love this movie!")
# out: [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

### 2. Zero-Shot Classification (and Few-Shot)

**Use case:** Classify text into arbitrary labels without task-specific training.

```python
from transformers import pipeline

pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
out = pipe(
    "This movie is fantastic and well-acted.",
    candidate_labels=["positive", "negative", "neutral"]
)
# out: {'labels': [...], 'scores': [...]}

# Few-shot: include examples in the text or as part of the hypothesis design.
```

---

### 3. Text Generation

**Use case:** Continue a prompt, chat, or generate creative text.

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="gpt2")
out = pipe("Once upon a time", max_new_tokens=30)
# out: [{'generated_text': 'Once upon a time ...'}]
```

---

### 4. Fill-Mask

**Use case:** Predict missing tokens in a sentence (masked language modeling).

```python
from transformers import pipeline

pipe = pipeline("fill-mask", model="distilbert-base-uncased")
out = pipe("The cat sat on the [MASK].")
# out: list of {'token_str': 'mat', 'score': ...}, etc.
```

---

### 5. Named Entity Recognition (NER)

**Use case:** Detect entities (person, organization, location, etc.) in text.

```python
from transformers import pipeline

pipe = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
out = pipe("Apple is looking at buying U.K. startup for $1 billion.")
# out: list of {'entity': 'B-ORG', 'word': 'Apple', ...}
```

---

### 6. Question Answering

**Use case:** Extract answer span from a context given a question.

```python
from transformers import pipeline

pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
out = pipe(
    question="Who is the CEO of Microsoft?",
    context="Satya Nadella is the CEO of Microsoft."
)
# out: {'answer': 'Satya Nadella', 'score': ...}
```

---

### 7. Summarization

**Use case:** Produce a short summary of a long document.

```python
from transformers import pipeline

pipe = pipeline("summarization", model="facebook/bart-large-cnn")
out = pipe("Long article text goes here...", max_new_tokens=50)
# out: [{'summary_text': '...'}]
```

---

### 8. Translation

**Use case:** Translate text from one language to another.

```python
from transformers import pipeline

pipe = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
out = pipe("Hello, how are you?")
# out: [{'translation_text': 'Bonjour, comment ça va ?'}]
```

(Use other task names like `translation_en_to_de`, etc., for other language pairs.)

---

### 9. Feature Extraction

**Use case:** Get contextual embeddings for text (for similarity, clustering, etc.).

```python
from transformers import pipeline

pipe = pipeline("feature-extraction", model="distilbert-base-uncased")
out = pipe("This is a sample sentence.")
# out: tensor of shape [1, seq_len, hidden_size]
```

---

### 10. Image Classification

**Use case:** Predict image label (e.g., cat, dog, car).

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
img = Image.open("cat.jpg").convert("RGB")
out = pipe(img)
# out: [{'label': 'tabby', 'score': ...}, ...]
```

---

### 11. Object Detection

**Use case:** Detect and localize objects with bounding boxes.

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("object-detection", model="facebook/detr-resnet-50")
img = Image.open("street.jpg").convert("RGB")
out = pipe(img)
# out: list of {'label': 'car', 'score': ..., 'box': {...}}
```

---

### 12. Image Segmentation

**Use case:** Pixel-level segmentation (semantic or instance).

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
img = Image.open("street.jpg").convert("RGB")
out = pipe(img)
# out: list of {'label': 'road', 'score': ..., 'mask': PIL.Image}
```

---

### 13. Image-to-Text (Captioning / VQA-style)

**Use case:** Generate a caption or description for an image.

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
img = Image.open("cat.jpg").convert("RGB")
out = pipe(img)
# out: [{'generated_text': 'a cat sitting on a mat'}]
```

---

### 14. Automatic Speech Recognition (ASR)

**Use case:** Transcribe speech audio to text.

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
out = pipe("audio.wav")
# out: {'text': 'recognized transcript'}
```

(You can also pass raw audio arrays with appropriate sampling rate.)

---

### 15. Text-to-Speech (TTS)

**Use case:** Synthesize speech audio from text.

```python
from transformers import pipeline

pipe = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")
out = pipe("Hello, this is a test.")
# out: {'audio': array, 'sampling_rate': 16000}
```

(You can save the audio array using `scipy.io.wavfile.write` or `soundfile`.)

---

### 16. Text-to-Image

**Use case:** Generate images from text prompts.

```python
from transformers import pipeline

pipe = pipeline("text-to-image", model="stabilityai/stable-diffusion-2-1")
img = pipe("a futuristic city at sunset, digital art")
# img: PIL.Image
img.save("generated.png")
```

(Depending on your setup, you may need `diffusers` for some models, but HF now exposes some T2I via pipelines.)

---

## Summary

- **Pipeline** = Preprocess + Model + Postprocess in one call.
- Creation styles:
  - `pipeline("task")` → default model.
  - `pipeline("task", model="...")` → specific model.
  - `pipeline("task", model=..., tokenizer=...)` → full control.
- Covers NLP, vision, and audio tasks with minimal code, ideal for prototyping, demos, and baseline systems.