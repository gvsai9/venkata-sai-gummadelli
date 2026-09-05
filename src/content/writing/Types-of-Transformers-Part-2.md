---
title: "Types of Transformer Part 2"
description: "Types of transformers."
pubDate: 2026-09-07
category: "AI"
tags:
  - NLP
  - AI
  - Transformers
thumbnail: ../../assets/writing/tp2.png
thumbnailAlt: "Description of the image"
---
# Transformers – Part 2: Encoder–Decoder, Vision, and Multi-modal Transformers

## 4) Encoder–Decoder Transformers

### Use cases

Encoder–decoder transformers are designed for **sequence-to-sequence** tasks where the input and output are both sequences, potentially of different lengths and modalities:

- Machine translation (e.g., English → French)
- Summarization (long document → short summary)
- Text-to-text generation with strong conditioning (e.g., question → answer)
- Speech-to-text (audio tokens → text)
- Structured generation (code generation from description, data-to-text)

They combine the “understanding” strength of encoders with the “generation” strength of decoders.

### Representative models

- T5 (Text-To-Text Transfer Transformer)
- BART (Bidirectional and Auto-Regressive Transformers)
- Original Transformer (Vaswani et al.) for translation
- mT5, ByT5 (multilingual / byte-level variants)
- Domain-specific seq2seq models (e.g., PEGASUS for summarization)

### Full process: how an encoder–decoder transformer works

The architecture has two main parts:

1. **Encoder**
   - Similar to encoder-only models (e.g., BERT-style).
   - Takes the input sequence and produces contextualized representations.
   - Uses **bidirectional self-attention** (each token can attend to all tokens in the input).

2. **Decoder**
   - Similar to decoder-only models but with an extra component: **cross-attention**.
   - Generates the output sequence token-by-token in an autoregressive way.
   - Uses:
     - Masked self-attention over already-generated output tokens.
     - Cross-attention over encoder outputs.

#### Step-by-step flow

Given input sequence \(X = (x_1, \dots, x_n)\) and target sequence \(Y = (y_1, \dots, y_m)\):

1. **Encoding**
   - Tokenize and embed \(X\).
   - Pass through L encoder layers with bidirectional self-attention.
   - Output: encoder representations \(H = (h_1, \dots, h_n)\).

2. **Decoding (training)**
   - Tokenize and embed \(Y\), shifted right by one (teacher forcing).
   - For each decoder layer:
     - **Masked self-attention** over decoder inputs (so position \(t\) sees only \(< t\)).
     - **Cross-attention**:
       - Queries come from decoder states.
       - Keys and values come from encoder outputs \(H\).
     - FFN, LayerNorm, residuals.

3. **Cross-attention (key idea)**

   In cross-attention:
   - Query: from decoder state \(S_t\) at time \(t\).
   - Key, Value: from encoder outputs \(H\).

   \[
   \text{CrossAttention}(S_t, H) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
   \]
   where:
   - \(Q = S_t W_Q\)
   - \(K = H W_K\), \(V = H W_V\)

   This lets the decoder “look at” relevant parts of the input sequence when generating each output token. In translation, this is what enables alignment (e.g., focusing on the subject noun phrase when generating its translation).

4. **Output head**
   - Final decoder hidden states → linear projection → vocabulary logits.
   - Train with next-token loss over target sequence:
     \[
     \mathcal{L} = -\sum_{t=1}^{m} \log p(y_t \mid y_{<t}, X)
     \]

5. **Inference (generation)**
   - Start with start-of-sequence token.
   - At each step:
     - Run encoder once on \(X\) (cached).
     - Run decoder with current output prefix.
     - Sample or take argmax next token.
     - Append and repeat until end-of-sequence.

### Why cross-attention is important

- It explicitly connects input and output sequences.
- Allows dynamic, content-based focusing on different input tokens for each output token.
- Critical for tasks where output depends heavily on specific parts of the input (translation, summarization, QA with long context).

### Minimal Hugging Face snippet (encoder–decoder)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_text = "translate English to French: The cat sits on the mat."
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=20,
    num_beams=4,
    early_stopping=True
)

print(tokenizer.decode(outputs, skip_special_tokens=True))
```

---

## 5) Vision Transformers (ViT)

### Use cases

Vision Transformers apply the transformer architecture to images, treating them as sequences of patches:

- Image classification
- Object detection (with adaptations like DETR)
- Semantic segmentation
- Self-supervised representation learning for vision
- Multi-modal models that combine vision and text (as the vision backbone)

They compete with CNNs by relying more on global attention and less on local convolutional inductive bias.

### Representative models

- ViT (Vision Transformer) – base, large, huge variants
- DeiT (Data-efficient Image Transformers)
- Swin Transformer (hierarchical, shifted windows)
- BEiT, MAE (self-supervised ViT variants)
- DETR (for detection), Segmenter (for segmentation)

### End-to-end working of a basic ViT for classification

1. **Input image**
   - Image of size \(H \times W \times C\) (e.g., 224×224×3).

2. **Patch embedding**
   - Split image into fixed-size patches, e.g., \(P \times P\) (16×16).
   - Number of patches: \(N = \frac{H \times W}{P^2}\).
   - Flatten each patch and project with a linear layer to dimension \(d\):
     \[
     \text{PatchEmbed}: \mathbb{R}^{P^2 C} \to \mathbb{R}^d
     \]
   - Add a learned **[CLS]** token embedding at the start.
   - Add learned positional embeddings for each patch + CLS.

3. **Transformer encoder**
   - Stack of L encoder layers (similar to BERT):
     - Multi-head self-attention over all patch tokens (including CLS).
     - FFN, LayerNorm, residuals.
   - Attention lets each patch attend to all other patches → global context.

4. **Classification head**
   - Take the final representation of the **[CLS]** token.
   - Pass through MLP (often just a linear layer) → class logits.
   - Train with cross-entropy loss over image labels.

5. **Training strategies**
   - Supervised training on large datasets (ImageNet-1k, ImageNet-21k).
   - Self-supervised pretraining (e.g., MAE: mask patches, reconstruct pixels).
   - Fine-tuning on downstream tasks (detection, segmentation) with task-specific heads.

### Intuition: why transformers work for images

- CNNs use local filters and build up receptive fields layer-by-layer.
- ViT uses self-attention from the first layer: every patch can directly interact with every other patch.
- This gives strong global modeling capacity, especially when trained on large data.

### Minimal Hugging Face snippet (ViT classification)

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

img = Image.open("cat.jpg").convert("RGB")
inputs = processor(img, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = probs.argmax().item()

print("Predicted class ID:", pred)
print("Top probabilities:", probs[pred].item())
```

---

## 6) Multi-modal Transformers (e.g., CLIP)

### Use cases

Multi-modal transformers process and align multiple modalities (typically text and image) in a shared representation space:

- Zero-shot image classification (using text labels as prompts)
- Image–text retrieval (find images matching a caption, or vice versa)
- Visual question answering (with additional heads)
- Guiding generative models with text (e.g., text-to-image scoring)
- Embedding-based search over images and text together

### Representative models

- CLIP (Contrastive Language–Image Pre-training)
- ALIGN, LiT
- Multi-modal extensions: Flamingo, BLIP, LLaVA (for VQA / chat)

### CLIP: end-to-end process

CLIP learns a joint embedding space for images and text using contrastive learning.

#### Architecture overview

1. **Image encoder**
   - Typically a ViT or ResNet.
   - Input: image.
   - Output: fixed-dimensional image embedding \(v\).

2. **Text encoder**
   - Typically a transformer (similar to GPT/BERT-style).
   - Input: tokenized text (e.g., class name, caption).
   - Output: text embedding \(t\).

3. **Projection to shared space**
   - Both image and text embeddings are projected to the same dimension \(d\) via learned linear layers:
     - \(v' = W_v v\)
     - \(t' = W_t t\)
   - Often followed by L2 normalization.

4. **Contrastive training objective**
   - Given a batch of \(N\) image–text pairs \((I_i, T_i)\):
     - Compute similarity matrix \(S \in \mathbb{R}^{N \times N}\):
       \[
       S_{ij} = \frac{v'_i \cdot t'_j}{\tau}
       \]
       where \(\tau\) is a learnable temperature parameter.
     - Apply softmax row-wise and column-wise.
     - Maximize likelihood that each image matches its correct text and vice versa (symmetric cross-entropy loss).
   - This pulls matching pairs together and pushes non-matching pairs apart in the embedding space.

#### Inference: zero-shot classification

To classify an image into classes \(\{c_1, \dots, c_K\}\):

1. Create text prompts, e.g.:  
   - “a photo of a {class}” → “a photo of a cat”, “a photo of a dog”, etc.
2. Encode all prompt texts → embeddings \(\{t'_1, \dots, t'_K\}\).
3. Encode the image → embedding \(v'\).
4. Compute cosine similarities:
   \[
   s_k = \frac{v' \cdot t'_k}{\|v'\|\|t'_k\|}
   \]
5. Choose class with highest similarity:
   \[
   \hat{c} = \arg\max_k s_k
   \]

No task-specific training is needed; the model relies on the aligned space learned during pretraining.

### Connectors and projectors in multi-modal models

In more advanced multi-modal architectures (e.g., Flamingo, LLaVA):

- **Connector / projector**: a small network (often MLP) that maps embeddings from one modality into the input space of a language model.
  - Example: image features from ViT → projector → tokens/embeddings that a LLM can attend to.
- This allows:
  - The LLM to “see” images as additional context tokens.
  - Unified reasoning over text and images using a single decoder.

Typical pattern:
- Vision encoder: ViT → feature map.
- Projector: linear/MLP → dimension compatible with LLM.
- LLM: decoder-only transformer that attends to both text tokens and projected visual tokens via cross-attention or concatenated sequences.

### Minimal Hugging Face snippet (CLIP zero-shot classification)

```python
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

model_name = "openai/clip-vit-base-patch32"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load image
img = Image.open("cat.jpg").convert("RGB")

# Define classes
classes = ["cat", "dog", "bird", "car"]
texts = [f"a photo of a {c}" for c in classes]

# Process inputs
inputs_img = processor(images=img, return_tensors="pt")
inputs_txt = processor(text=texts, return_tensors="pt", padding=True)

# Get embeddings
with torch.no_grad():
    img_emb = model.get_image_features(**inputs_img)
    txt_emb = model.get_text_features(**inputs_txt)

# Normalize
img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

# Compute similarities
similarities = (img_emb @ txt_emb.T).softmax(dim=-1)
pred_idx = similarities.argmax().item()

print("Predicted class:", classes[pred_idx])
print("Probabilities:", similarities.tolist())
```

---

This completes Parts 4–6.  
Next part will cover:

- Diffusion transformers
- Mixture of Experts (MoE)
- Long-context & sparse-attention transformers (including Flash Attention)