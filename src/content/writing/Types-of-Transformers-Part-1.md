---
title: "Types of Transformer Part 1"
description: "Types of transformers."
pubDate: 2026-09-06
category: "AI"
tags:
  - NLP
  - AI
  - Transformers
thumbnail: ../../assets/writing/tp1.png
thumbnailAlt: "Description of the image"
---
# Transformers – Part 1: Encoder-only, Decoder-only, and Their Differences

## 1) Encoder-only Transformers

### Use cases

Encoder-only models are best when you need rich contextual representations of input text for tasks like:

- Text classification (sentiment, spam, topic)
- Token-level tasks (NER, POS tagging)
- Sentence similarity / semantic search
- Question answering (extractive)
- Reranking and retrieval scoring

They are not designed for autoregressive generation; they are designed to “understand” input sequences.

### Representative models

- BERT (Bidirectional Encoder Representations from Transformers)
- RoBERTa, ALBERT, DeBERTa
- DistilBERT, ELECTRA (efficient / training-trick variants)
- Domain-specific: BioBERT, SciBERT, LegalBERT

### Full process: how an encoder-only transformer works

Given an input sentence, the pipeline is:

1. **Tokenization**
   - Text → tokens using a subword tokenizer (e.g., WordPiece, BPE).
   - Special tokens added: `[CLS]` at start, `[SEP]` between sentences, etc.
   - Each token mapped to an integer ID.

2. **Embedding layer**
   - Token IDs → token embeddings (learned vectors).
   - Add positional embeddings (sinusoidal or learned) to inject order information.
   - Optionally add segment embeddings (for sentence-pair tasks).

3. **Stack of encoder layers**

   Each encoder block has:
   - Multi-Head Self-Attention (MHSA)
   - Feed-Forward Network (FFN)
   - LayerNorm and residual connections around each.

   For a sequence of token embeddings \(X \in \mathbb{R}^{n \times d}\):

   - Self-attention computes:
     \[
     \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
     \]
     where \(Q = XW_Q\), \(K = XW_K\), \(V = XW_V\).

   - Multi-head: split into \(h\) heads, each with its own \(W_Q^i, W_K^i, W_V^i\), compute attention in parallel, then concatenate and project:
     \[
     \text{MultiHead}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W_O
     \]

   - FFN: position-wise two-layer MLP:
     \[
     \text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
     \]

   - Residual + LayerNorm:
     \[
     X' = \text{LayerNorm}(X + \text{MHSA}(X))
     \]
     \[
     X'' = \text{LayerNorm}(X' + \text{FFN}(X'))
     \]

   This repeats for \(L\) layers (e.g., 12 for BERT-base).

4. **Task-specific head**
   - For classification: take `[CLS]` token representation → dense → softmax.
   - For token classification: per-token output → CRF or softmax.
   - For sentence pairs: concatenate `[CLS]` or use pair embeddings.

5. **Training objectives (pretraining)**
   - Masked Language Modeling (MLM): randomly mask tokens, predict them.
   - Next Sentence Prediction (NSP) or variants (sentence-order prediction).
   - Then fine-tune on downstream tasks with labeled data.

### MHSA intuition (light math + example)

Consider the sentence: “The cat sat on the mat”.

- Each token can “attend” to every other token.
- For token “cat”, attention scores might be high for “The”, “sat”, “the”, “mat” because they are contextually relevant.
- Multi-head lets different heads focus on different relations:
  - Head 1: syntactic dependencies (subject–verb)
  - Head 2: coreference (cat ↔ it)
  - Head 3: local n-gram patterns

In BERT, all tokens see all others (bidirectional), which is why it is powerful for understanding but not for left-to-right generation.

### Minimal Hugging Face snippet (encoder-only)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "I love this movie!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = probs.argmax().item()

print("Sentiment:", "positive" if pred == 1 else "negative")
```

---

## 2) Decoder-only Transformers

### Use cases

Decoder-only models are designed for **autoregressive generation**:

- Text completion and continuation
- Chatbots and assistants
- Code generation
- Story / poem / email drafting
- Conditional generation when you prepend instructions or context as part of the input sequence

They generate one token at a time, conditioning on all previous tokens.

### Representative models

- GPT family: GPT-2, GPT-3, GPT-3.5, GPT-4 (architecture class: decoder-only)
- LLaMA, Mistral, Falcon, Phi
- Code models: CodeLlama, StarCoder
- Instruction-tuned variants: GPT-3.5/4, LLaMA-2/3-Chat, etc.

### Full process: how a decoder-only transformer works

1. **Tokenization & embeddings**
   - Similar to encoder: text → tokens → token + positional embeddings.
   - Often no segment embeddings; position encodings may be learned or rotary (RoPE).

2. **Stack of decoder layers**

   Each decoder block typically has:
   - Masked Multi-Head Self-Attention
   - (Optionally) cross-attention in encoder-decoder setups; pure decoder-only models usually skip this.
   - Feed-Forward Network
   - LayerNorm + residuals.

   Key difference: **causal masking** in self-attention.

3. **Masked Multi-Head Self-Attention**

   Standard attention:
   \[
   \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
   \]

   In decoder-only:
   - For position \(i\), token can attend only to positions \(\le i\).
   - Implement via a causal mask \(M\) where:
     \[
     M_{ij} = 
     \begin{cases}
       0 & \text{if } j \le i \\
       -\infty & \text{if } j > i
     \end{cases}
     \]
   - Then:
     \[
     \text{CausalAttention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
     \]

   This ensures generation is strictly left-to-right.

   Multi-head works as in encoders, but each head respects the causal mask.

4. **Autoregressive generation loop**

   At inference:
   - Start with prompt tokens.
   - Run transformer, get logits for next token at last position.
   - Sample or take argmax → new token.
   - Append new token to sequence, repeat.

   Training uses teacher forcing: feed full sequence, but mask ensures each position predicts the next token given only past tokens.

5. **Training objective**
   - Next-token prediction (language modeling):
     \[
     \mathcal{L} = -\sum_t \log p(x_t \mid x_{<t})
     \]
   - Pretrain on massive corpora; then optionally instruction-tune with supervised + RLHF.

### Masked multi-head self-attention intuition

Take prompt: “Once upon a”.

- When predicting “a”, the model sees “Once upon”.
- When predicting next token after “a”, it sees “Once upon a”.
- Causal mask prevents seeing future tokens, so no leakage.

Different heads capture:
- Local syntax (next word patterns)
- Long-range dependencies (subject–verb across sentences)
- Style and topic consistency

### Minimal Hugging Face snippet (decoder-only)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"  # or "meta-llama/Llama-2-7b-hf" if you have access
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        top_p=0.95
    )

print(tokenizer.decode(outputs, skip_special_tokens=True))
```

---

## 3) Encoder-only vs Decoder-only: Component-level Differences

Both use the same core building blocks (embeddings, self-attention, FFN, LayerNorm), but differ in **attention pattern**, **training objective**, and **typical usage**.

### Core component comparison

| Component                | Encoder-only (e.g., BERT)                          | Decoder-only (e.g., GPT)                            |
|------------------------|------------------------------------------------------|-----------------------------------------------------|
| Self-attention type    | Bidirectional (full) self-attention                 | Causal (masked) self-attention                     |
| Masking                | No causal mask; may use attention mask for padding  | Causal mask to prevent seeing future tokens        |
| Primary objective      | Masked LM (predict masked tokens), NSP, etc.        | Next-token prediction (autoregressive LM)          |
| Typical tasks          | Classification, NER, QA, similarity, reranking      | Text generation, chat, code gen, completion        |
| Context usage          | Sees entire sequence at once                        | Sees only prefix when generating                   |
| Output head            | Often `[CLS]` or token-level heads                  | Language modeling head over vocab at each position |
| Fine-tuning pattern    | Add task-specific head on top                       | Often used as-is or with prompt/instruction tuning |
| Example models         | BERT, RoBERTa, ALBERT, DeBERTa                      | GPT-2/3, LLaMA, Mistral, Falcon                    |

### How attention differs in practice

- **Encoder-only**:
  - For token \(i\), attention scores over all \(j \in \{1,\dots,n\}\).
  - Great for tasks needing full context (e.g., “Does pronoun ‘it’ refer to ‘cat’ or ‘mat’?”).

- **Decoder-only**:
  - For token \(i\), attention scores only over \(j \le i\).
  - Necessary for coherent generation; otherwise the model would “cheat” by seeing the future.

### When to choose which?

- Need **understanding / representation** of fixed text → encoder-only.
- Need **generation** of new text token-by-token → decoder-only.
- For tasks like QA:
  - Extractive QA (span selection) → encoder-only.
  - Generative QA (free-form answers) → decoder-only or encoder-decoder.