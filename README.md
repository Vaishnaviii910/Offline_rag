# 📚 Precision-RAG: Advanced Textbook Analysis System

### **Overview**
This project implements a high-precision **Retrieval-Augmented Generation (RAG)** pipeline designed to eliminate "Vagueness" and "Hallucinations" in academic document sets. Unlike traditional RAG systems that rely on simple keyword matching, this system utilizes a **Two-Stage Retrieval** architecture with a **Cross-Encoder Reranker**.

### **🛠️ The Dataset**
* **Source:** *Project Management: A Managerial Approach* by Meredith & Mantel.
* **Format:** Digital PDF (approx. 600 pages).
* **Content:** Complex project management theories, case studies, and the "Triple Constraint" (Time, Cost, Scope) framework.

---

### **🚀 Technical Architecture (Step-by-Step)**

#### **Phase 1: Ingestion & Chunking**
* The PDF is parsed and split into overlapping text chunks.
* **Strategy:** Overlapping ensures that definitions spanning across two pages are not cut in half.

#### **Phase 2: Hybrid Retrieval (Recall)**
* **Dense Search:** Uses Vector Embeddings to find "Meaning."
* **Sparse Search:** Uses BM25 algorithm to find "Exact Keywords."
* **Action:** The system pulls the **Top 40** most likely candidates.

#### **Phase 2.5: Cross-Encoder Reranking (Precision)**
* **The Filter:** Uses the `ms-marco-MiniLM-L-6-v2` model to "read" the query and chunks together.
* **Scoring:** Each chunk receives a **Relevance Score**. 
    * *High Scores (+3.35):* Exact match.
    * *Low Scores (-1.58):* Vague or irrelevant context.
* **Outcome:** Only the top 5 highest-scoring chunks are sent to the AI.

#### **Phase 3: Grounded Generation**
* **Model:** `DeepSeek-R1:1.5b` (Running locally via Ollama).
* **Reasoning:** The R1 model performs step-by-step thinking to ensure its answer is strictly grounded in the provided textbook excerpts.

---

### **📊 Performance Highlights**
* **Anti-Hallucination:** By using a "Truth Filter" (Rerank Score), the system identifies when a query is too vague to be answered accurately.
* **Latency:** Retrieval (~0.002s) + Reranking (~0.4s) + Generation (~19s).
* **Accuracy:** Successfully distinguishes between general mentions of authors and specific definitions (e.g., distinguishing between "Project Success" and the "Definition of a Project").

---

### **💻 How to Run**
1.  **Install Dependencies:**
    ```bash
    pip install langchain sentence-transformers chromadb ollama
    ```
2.  **Start Ollama:**
    ```bash
    ollama run deepseek-r1:1.5b
    ```
3.  **Run the Notebook:**
    Open `offline_rag.ipynb` and execute all cells.
