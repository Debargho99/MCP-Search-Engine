# 🔍 MCP Search Engine

A modern GUI-based platform for searching and extending Model Context Protocols (MCPs) using open-source LLMs like Sarvam-0.5 🇮🇳, Mistral 🇪🇺, and LLaMA 🌍.

---

## 📌 Architecture

![MCP Schema](mcp_schema_diagram_final.png)

---

## ✨ Features
- Upload your own MCP dataset (JSON)
- Find similar MCPs using vector search (FAISS)
- Get relevant tool suggestions using open-source LLMs
- Interactive UI built with Streamlit

---

## 💬 Example MCPs
- Invoice table extractor using OCR + LayoutLM
- Summarizer for academic PDFs with mT5
- Toxic content classifier for social media
- Recommender system using embeddings + Faiss
- Chatbot orchestration with fallback tools

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run mcp_search_gui.py
```

> Requires GPU for fast LLM inference or use quantized models.

---

## 🧠 LLMs Supported

| Model            | Description                                    |
|------------------|------------------------------------------------|
| Sarvam-0.5 🇮🇳    | Indian instruction-tuned, compact & fast       |
| Mistral-7B 🇪🇺    | High-quality, multilingual, performant         |
| LLaMA-3 🌍       | Versatile general-purpose inference            |

---

## 📁 Dataset Format

```json
[
  {
    "id": "mcp_001",
    "description": "An MCP to extract tabular data from scanned invoices..."
  }
]
```
