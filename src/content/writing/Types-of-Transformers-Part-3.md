---
title: "Types of Transformer Part 3"
description: "Types of transformers."
pubDate: 2026-09-08
category: "AI"
tags:
  - NLP
  - AI
  - Transformers
thumbnail: ../../assets/writing/tp3.png
thumbnailAlt: "Description of the image"
---
# Transformers – Part 3: Diffusion, MoE, Long-Context, and Retrieval-Augmented Transformers

## 7 Diffusion Transformers

### Use cases

Diffusion transformers combine transformer backbones with diffusion models to generate high-quality data, especially images and sometimes audio or video:

- Text-to-image generation
- Image inpainting and outpainting
- Image editing and style transfer
- Super-resolution
- Conditional generation (e.g., layout-to-image, sketch-to-image)

Instead of generating tokens autoregressively, diffusion models generate data by iteratively denoising a noisy signal.

### Representative models

- DiT (Diffusion Transformer)
- Stable Diffusion variants that use transformer-based denoisers (e.g., some SDXL components)
- Latent diffusion models with transformer backbones
- Video diffusion models using temporal transformers

### Core idea: diffusion process

Diffusion models work with two processes:

1. **Forward (noising) process**
   - Start from clean data \(x_0\) (e.g., an image).
   - Gradually add Gaussian noise over \(T\) steps:
     \[
     x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,I)
     \]
   - After many steps, \(x_T\) is approximately pure noise.

2. **Reverse (denoising) process**
   - Learn a model \(p_\theta(x_{t-1} \mid x_t, t)\) that predicts how to remove noise.
   - Typically parameterized as a neural network that predicts:
     - The noise \(\epsilon_\theta(x_t, t)\), or
     - The clean image \(x_0\), or
     - The mean/variance of \(x_{t-1}\).

   - Sampling starts from random noise \(x_T \sim \mathcal{N}(0,I)\) and iteratively denoises to get \(x_0\).

### Where the transformer fits in

In diffusion transformers:

- The **denoiser network** is a transformer instead of a U-Net CNN.
- Input:
  - Noisy latent/image \(x_t\)
  - Time step \(t\) (embedded)
  - Optional condition (e.g., text prompt, class label)
- The transformer processes patches (for images) or tokens (for latents) using self-attention and possibly cross-attention to conditions.

Typical setup (e.g., DiT):

1. **Patchify image/latent**
   - Split \(x_t\) into patches, embed to dimension \(d\).
2. **Add time and condition embeddings**
   - Time \(t\) → embedding via MLP or sinusoidal.
   - Text condition → encoded with a text transformer (e.g., CLIP text encoder).
3. **Transformer blocks**
   - Self-attention over patches.
   - Cross-attention to text embeddings (for text-conditioned generation).
   - FFN, LayerNorm, residuals.
4. **Output head**
   - Predict noise (or clean signal) for each patch.
   - Loss: MSE between predicted and true noise.

### Denoising vs noising: intuitive view

- **Noising**: systematically corrupt the data until it becomes random.
- **Denoising**: train a model to reverse this corruption step-by-step.
- The transformer learns how global structure (shapes, objects, layout) should look at each noise level and gradually “sculpts” coherent images from noise.

### Minimal Hugging Face–style sketch (conceptual)

Actual diffusion models are heavy and often use custom pipelines, but conceptually:

```python
# Pseudo-code style (not directly runnable as-is)
from diffusers import DDPMScheduler, UNet2DModel  # or transformer-based denoiser
import torch

# Assume we have a transformer-based denoiser `model`
# and a scheduler that handles noise schedule
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Start from random noise
x_t = torch.randn(1, 4, 64, 64)  # e.g., latent shape

for t in reversed(range(scheduler.num_train_timesteps)):
    # Prepare input
    t_tensor = torch.full((1,), t, dtype=torch.long)

    # Model predicts noise
    with torch.no_grad():
        noise_pred = model(x_t, t_tensor)  # transformer denoiser

    # Scheduler step: compute x_{t-1}
    x_t = scheduler.step(noise_pred, t, x_t).prev_sample

# x_t is now the generated sample (latent), decode to image
```

For real usage, libraries like `diffusers` provide ready pipelines with transformer backbones.

---

## 8) Mixture of Experts (MoE) Transformers

### How it works

Mixture of Experts (MoE) transformers scale model capacity without proportionally increasing compute per token by:

- Having many **expert** sub-networks (usually FFNs).
- Using a **gating network** to route each token to a small subset of experts (often 1–2).

Only the selected experts are activated for each token, so total FLOPs per token stay manageable even if the model has billions or trillions of parameters.

### Basic architecture

For each transformer layer:

1. **Self-attention block** (shared across all tokens).
2. **MoE feed-forward block**:
   - Suppose there are \(E\) experts: \(\text{FFN}_1, \dots, \text{FFN}_E\).
   - A gating network \(g(x)\) takes token representation \(x\) and produces routing weights.

   Common scheme (top-k gating):
   - Compute gate scores:
     \[
     s = \text{softmax}(W_g x)
     \]
   - Select top-\(k\) experts (e.g., \(k=1\) or \(2\)).
   - Output:
     \[
     \text{MoE}(x) = \sum_{i \in \text{TopK}(s)} s_i \cdot \text{FFN}_i(x)
     \]

3. **Load balancing**
   - Auxiliary losses or constraints ensure tokens are distributed reasonably across experts, avoiding “dead” experts.

### Why many experts?

- **Capacity**: More parameters → better function approximation, especially for diverse tasks and domains.
- **Efficiency**: Only a fraction of parameters are used per token, so inference/training cost grows slower than parameter count.
- **Specialization**: Different experts can specialize in different patterns (syntax, facts, domains, languages).

### Representative models

- Switch Transformer (Google)
- GShard / MoE variants of T5
- GPT-style MoE models (e.g., some large LLMs from OpenAI, Anthropic, Meta)
- Mixtral (open MoE LLM)

### Trade-offs

- Pros:
  - Very large effective model size with manageable compute.
  - Good for multi-task, multi-domain settings.
- Cons:
  - More complex training (routing, load balancing).
  - Memory footprint is large (many experts stored in GPU/TPU memory).
  - Routing can complicate distributed training and inference.

### Minimal Hugging Face–style note

Many MoE models are not directly available as simple `AutoModel` yet, but conceptually:

```python
# Conceptual pseudo-code for one MoE layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        b, s, d = x.shape
        flat = x.view(-1, d)  # [B*seq, d]

        gate_logits = self.gate(flat)  # [B*seq, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)

        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i).any(dim=-1)
            if not mask.any():
                continue
            tokens = flat[mask]
            # sum contributions from experts where selected
            for k in range(self.top_k):
                sel = (topk_idx[:, k] == i) & mask
                if not sel.any():
                    continue
                out[sel] += topk_probs[sel, k: k+1] * expert(tokens[sel])

        return out.view(b, s, d)
```

---

## 9) Long-Context & Sparse-Attention Transformers

### The problem with standard attention

Standard self-attention has:

- Time and memory complexity: \(O(n^2)\) for sequence length \(n\).
- For long documents, code files, books, or long conversations, this becomes prohibitive.

Long-context transformers aim to handle sequences of tens or hundreds of thousands of tokens efficiently.

### Sparse attention: definition

Sparse attention restricts which tokens each token can attend to, reducing the number of attention scores computed.

Common patterns:

1. **Local attention**
   - Each token attends only to a window of nearby tokens (e.g., ±128 or ±512 positions).
   - Captures local dependencies efficiently.

2. **Global token attention**
   - A small set of “global” tokens (e.g., CLS, section headers, special tokens) attend to and are attended by all tokens.
   - Provides long-range connectivity.

3. **Structured patterns**
   - Strided attention (attend to every k-th token).
   - Block-sparse patterns (blocks of tokens attend to specific other blocks).
   - Combinations: local + global + strided.

The attention matrix becomes sparse, so complexity can drop to roughly \(O(n \log n)\) or \(O(n)\) depending on pattern.

### Representative models

- Longformer (local + global attention)
- BigBird (random + local + global patterns)
- Reformer (attention via hashing, approximate attention)
- Sparse Transformers (Child et al.)
- Models using block-sparse kernels for efficiency

### Flash Attention

Flash Attention is an algorithmic + systems optimization for attention that:

- Keeps the mathematical definition of full attention intact.
- Reduces memory traffic and improves cache usage on GPUs.
- Computes attention in tiles/blocks, avoiding materializing the full \(n \times n\) attention matrix in HBM.

Key properties:

- Same output as exact softmax attention.
- Much lower memory usage → enables longer sequences on the same hardware.
- Significant speedups, especially for long contexts.

Flash Attention is now widely used in modern LLM implementations (e.g., LLaMA, Mistral) to support long-context training and inference.

### Minimal Hugging Face–style note

You don’t usually configure sparse patterns manually; instead, you use models/architectures designed for long context:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "allenai/longformer-base-4096"  # example long-context model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Very long document..." * 1000
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)

outputs = model(**inputs)
# outputs.last_hidden_state has contextual representations for long input
```

For Flash Attention, it’s typically enabled via:

- Model implementation (e.g., `attn_implementation="flash_attention_2"` in some libraries).
- Hardware and library support (CUDA, Triton-based kernels).

---

## 10) Retrieval-Augmented Transformers

### What they are

Retrieval-augmented transformers combine:

- A **transformer language model** (for reasoning and generation).
- An external **retrieval system** (for fetching relevant documents, passages, or knowledge snippets).

Instead of relying solely on parametric memory (weights), they dynamically pull in information from a large corpus at inference time.

### How they differ from “RAG” as commonly used

“RAG” (Retrieval-Augmented Generation) is actually the standard term for this family of models. If you meant “how retrieval-augmented transformers differ from plain LLMs”:

- **Plain LLM**:
  - All knowledge is stored in model weights.
  - Cannot directly access external documents at inference (unless via tool use outside the model).

- **Retrieval-augmented transformer**:
  - Has an explicit retrieval step:
    - Given a query, retrieve top-k documents/passages from a corpus.
    - Feed these as additional context to the transformer.
  - Can be trained end-to-end (retriever + generator) or used in a pipelined way.

If you literally meant “how they differ from RAG”, then: they don’t; “RAG” *is* the common name for retrieval-augmented generation systems.

### Typical architecture and process

1. **Retriever**
   - Often a dense retriever:
     - Encoder (e.g., BERT-style) encodes queries and documents into embeddings.
     - Use approximate nearest neighbor search (e.g., FAISS) to find top-k documents for a query.
   - Can also be sparse (BM25) or hybrid.

2. **Generator (transformer)**
   - Takes as input:
     - Original query/prompt.
     - Retrieved documents (concatenated or structured).
   - Generates answer or continuation conditioned on this context.

3. **Training variants**
   - **Frozen retriever + fine-tuned generator**:
     - Retriever fixed (e.g., pre-trained DPR), generator fine-tuned on QA or generation tasks.
   - **End-to-end training**:
     - Jointly optimize retriever and generator (e.g., original RAG model).
   - **Tool-use style**:
     - LLM calls a retrieval API; retrieved text is inserted into the prompt.

### Examples

- RAG (Facebook/Meta): retriever (BART-style encoder) + BART generator.
- RETRO (DeepMind): retrieves chunks of text and conditions a transformer on them.
- Atlas, KNN-LM, and many open-domain QA systems.
- Modern LLMs using “search” tools are effectively retrieval-augmented at inference time.

### Benefits

- Access to up-to-date information (news, docs, codebases).
- Reduced hallucination when grounded in retrieved evidence.
- Smaller models can perform well by leaning on external knowledge.

### Minimal Hugging Face–style sketch

Conceptual pipeline:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Assume we have a retriever (e.g., DPR) and a document index

# 1. Retrieve
query = "Who invented the transformer architecture?"
top_docs = retriever.search(query, top_k=3)  # list of text passages

context = "\n\n".join(top_docs)
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

# 2. Generate
model_name = "facebook/rag-token-nq"  # example RAG model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
outputs = model.generate(**inputs, max_new_tokens=50)
answer = tokenizer.decode(outputs, skip_special_tokens=True)

print(answer)
```

In practice, many systems separate retrieval (e.g., using Elasticsearch/FAISS + embedding model) and generation (LLM API) but follow the same principle.

---

This completes Parts 7–10.  
If you want, next we can:

- Merge all parts into one cohesive long article with a unified intro and conclusion.
- Add more diagrams/explanations for specific sections (e.g., deeper MoE math, more diffusion details).