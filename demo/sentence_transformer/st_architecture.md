### **🔍 Transformer Architecture Explained (Vaswani et al., 2017)**

The **Transformer** is a deep learning model that **replaces recurrent networks (RNNs & LSTMs) with self-attention**. It is the foundation of models like **BERT, GPT, T5, and LLaMA**.

🔹 **Why Transformers?**

- Faster than RNNs (parallel processing).
- Handles long-range dependencies better.
- Used in NLP, vision, and multi-modal tasks.

---

## **⚙️ Transformer Architecture Overview**

The transformer consists of **an Encoder-Decoder** architecture:

- **Encoder** (Processes input text)
- **Decoder** (Generates output text)

However, models like **BERT use only the encoder**, while **GPT uses only the decoder**.

### **📌 Transformer Block (Inside Encoder & Decoder)**

Each block contains:  
1️⃣ **Multi-Head Self-Attention** (Focuses on different parts of the input)  
2️⃣ **Feedforward Network** (Enhances feature representation)  
3️⃣ **Layer Normalization & Residual Connections** (For stability & better learning)

---

## **🔹 1. Encoder (Used in BERT)**

The **encoder** processes input text **all at once** (unlike RNNs, which process sequentially).

🔹 **Steps in Encoder:**  
✅ **Token + Positional Encoding** (Adds order information)  
✅ **Self-Attention** (Finds relationships between words)  
✅ **Feedforward Network** (Processes learned representations)  
✅ **Layer Normalization & Skip Connections** (Stabilizes learning)

🔸 **Example:**  
For `"The cat sat on the mat."`, the **self-attention mechanism** allows the word `"cat"` to focus on `"sat"` rather than just adjacent words.

---

## **🔹 2. Decoder (Used in GPT & T5)**

The **decoder** is responsible for text generation (e.g., GPT models). It is similar to the encoder but with **Masked Self-Attention** to prevent "peeking" at future tokens.

🔹 **Steps in Decoder:**  
✅ **Masked Self-Attention** (Ensures autoregressive text generation)  
✅ **Encoder-Decoder Attention** (Links encoder & decoder representations)  
✅ **Feedforward Network** (Enhances representation)  
✅ **Layer Normalization & Skip Connections**

🔸 **Example:**  
If given `"The cat sat"`, the decoder predicts `"on"` next by attending to past words.

---

## **🔹 3. Self-Attention (Key Mechanism)**

Instead of processing words one by one (like RNNs), self-attention allows **each word to focus on other words**.

📌 **How Self-Attention Works:**  
Each word in the sentence gets converted into **Query (Q), Key (K), and Value (V)** matrices.

- Words compare their **Q-K similarity** using a **dot product**.
- The result is a **weighted sum** using **V** values.
- This creates a **context-aware representation** of the input.

🔸 **Example:**  
In `"The cat sat on the mat"`, the word `"cat"` might focus on `"sat"` strongly while giving less attention to `"the"`.

---

## **🔹 4. Multi-Head Attention**

Instead of a single self-attention mechanism, **multi-head attention** allows different "heads" to learn different aspects of relationships.

🔸 **Example:**

- One head might focus on **syntactic** relationships (e.g., subject-verb agreement).
- Another head might focus on **semantic** relationships (e.g., `"dog"` and `"pet"`).

---

## **🔹 5. Positional Encoding**

Since transformers don’t have sequential processing like RNNs, they need **positional encodings** to track word order.  
This is done using **sin & cos functions**:

\[
PE*{(pos, 2i)} = \sin(pos / 10000^{2i/d*{\text{model}}})
\]

\[
PE*{(pos, 2i+1)} = \cos(pos / 10000^{2i/d*{\text{model}}})
\]

---

## **🚀 Applications of Transformer Models**

- **BERT (Bidirectional Encoder)** → Text Understanding
- **GPT (Decoder-Only Model)** → Text Generation
- **T5 (Encoder-Decoder)** → Translation, Summarization
- **ViTs (Vision Transformers)** → Image Recognition

---
