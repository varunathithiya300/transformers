### **🚀 Transformer Architecture (Vaswani et al., 2017)**

The **Transformer** is a deep learning model introduced in the paper **"Attention Is All You Need"** by Vaswani et al. in 2017. It **replaces RNNs and CNNs** for sequence modeling tasks by relying entirely on **self-attention** and **feedforward layers**.

**✨ Why Transformers?**
✅ **Faster** than RNNs (Processes all tokens at once)  
✅ **Better at long-range dependencies** (Uses self-attention)  
✅ **Scalable & parallelizable** (Easier to train on GPUs/TPUs)  
✅ **Foundation for modern NLP models** (e.g., BERT, GPT, T5, LLaMA)

---

## **⚙️ High-Level Overview**

The Transformer consists of **two main components**:

1️⃣ **Encoder** (Processes input text and generates embeddings)  
2️⃣ **Decoder** (Generates output text based on encoded input)

🔹 **BERT uses only the encoder** (for understanding text).  
🔹 **GPT uses only the decoder** (for generating text).  
🔹 **T5 uses both encoder & decoder** (for tasks like translation & summarization).

---

## **🛠️ Transformer Architecture Components**

### **1️⃣ Input Representation: Token + Positional Encoding**

Since transformers **do not process text sequentially**, they need a way to understand word order.

📌 **Steps:**

- Text is **tokenized** into word pieces (e.g., `"I love NLP"` → `["I", "love", "NLP"]`).
- Each token is converted into an **embedding vector**.
- **Positional encodings** (sinusoidal functions) are added to capture word order.

🔸 **Example:**  
| Token | Word Embedding | Positional Encoding | Final Input Vector |
|--------|--------------|--------------------|------------------|
| "I" | [0.2, 0.5] | [0.1, 0.3] | [0.3, 0.8] |
| "love" | [0.6, 0.9] | [0.2, 0.4] | [0.8, 1.3] |

---

### **2️⃣ Encoder Block (Used in BERT)**

Each encoder block consists of:  
1️⃣ **Multi-Head Self-Attention**  
2️⃣ **Feedforward Neural Network**  
3️⃣ **Layer Normalization & Residual Connections**

📌 **How Self-Attention Works:**

- Each token is converted into **Query (Q), Key (K), and Value (V) matrices**.
- Attention scores are computed as:

\[
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]

- This gives a weighted sum of the word relationships.

🔸 **Example:** In the sentence `"The cat sat on the mat"`, the word `"cat"` may have strong attention to `"sat"` but weak attention to `"the"`.

---

### **3️⃣ Multi-Head Attention**

Instead of using a single attention mechanism, **multiple heads** learn different relationships.

🔹 **Example:**

- **Head 1:** Focuses on **subject-verb** relationships.
- **Head 2:** Focuses on **semantic similarity** (e.g., `"dog"` and `"pet"`).
- **Head 3:** Focuses on **syntactic structure**.

📌 **Formula:**
\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}\_1, \text{head}\_2, ..., \text{head}\_h) W^O
\]

---

### **4️⃣ Feedforward Network (FFN)**

After self-attention, each token goes through a **fully connected feedforward network** with ReLU activation:

\[
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
\]

This helps the model **learn non-linear transformations** of the input.

---

### **5️⃣ Decoder Block (Used in GPT)**

The **decoder** is similar to the encoder but has an extra **Masked Self-Attention** layer.

- This ensures the model **only sees past tokens** (important for text generation).
- The **encoder-decoder attention** layer helps the decoder focus on relevant input tokens.

🔸 **Example:**  
If given `"The cat sat"`, the decoder predicts `"on"` next but doesn’t see `"mat"` yet.

---

## **🚀 How Transformers Work in NLP Models**

1️⃣ **BERT (Encoder-Only Model)**

- Pretrained using **Masked Language Modeling (MLM)**.
- Great for **text understanding** (e.g., classification, question answering).

2️⃣ **GPT (Decoder-Only Model)**

- Pretrained using **Causal Language Modeling (CLM)**.
- Great for **text generation** (e.g., ChatGPT, code generation).

3️⃣ **T5 (Encoder-Decoder Model)**

- Converts every NLP task into a **text-to-text problem**.
- Great for **translation, summarization, and question answering**.

---

## **📊 Summary Table**

| Component                 | Encoder (BERT) | Decoder (GPT) |
| ------------------------- | -------------- | ------------- |
| Self-Attention            | ✅             | ✅ (Masked)   |
| Multi-Head Attention      | ✅             | ✅            |
| Feedforward Network       | ✅             | ✅            |
| Layer Normalization       | ✅             | ✅            |
| Positional Encoding       | ✅             | ✅            |
| Encoder-Decoder Attention | ❌             | ✅            |

---

## **📌 Final Thoughts**

🔹 Transformers are **faster & more scalable** than RNNs.  
🔹 **Self-attention helps capture long-range dependencies**.  
🔹 **Multi-head attention** lets the model focus on different aspects of language.  
🔹 **Encoder models (BERT)** → Text understanding.  
🔹 **Decoder models (GPT)** → Text generation.  
🔹 **Encoder-Decoder models (T5)** → Translation & summarization.

---
