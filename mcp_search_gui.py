import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
import os

MODEL_OPTIONS = {
    "Sarvam-0.5 (India)": "sarvamai/sarvam-0.5",
    "Mistral-7B (Europe)": "mistralai/Mistral-7B-Instruct-v0.1",
    "LLaMA-3 (Meta)": "meta-llama/Meta-Llama-3-8B"
}

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DATA_PATH = "mcp_dataset.json"

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_mcp_data():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data

def build_faiss_index(embedder, mcp_data):
    if not mcp_data:
        return None, None, []
    texts = [item["description"] for item in mcp_data]
    vectors = embedder.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors, texts

@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return gen

st.set_page_config(page_title="MCP Search Engine", layout="wide")
st.title("🔍 MCP (Model Context Protocol) Search Engine")

st.markdown("""
Enter a Model Context Protocol (MCP) and choose an open-source LLM to:
- Find similar MCPs
- Suggest relevant tools or completions
""")

mcp_input = st.text_area("Enter your MCP here:", height=200)
model_choice = st.selectbox("Choose LLM for Suggestions", list(MODEL_OPTIONS.keys()))
search_btn = st.button("🔍 Search & Suggest")

if search_btn and mcp_input:
    with st.spinner("Loading components and searching..."):
        embedder = load_embedder()
        mcp_data = load_mcp_data()
        index, vectors, texts = build_faiss_index(embedder, mcp_data)

        if index is None:
            st.warning("No MCP data found. Please upload a dataset.")
        else:
            query_vec = embedder.encode([mcp_input], convert_to_numpy=True)
            D, I = index.search(query_vec, k=5)

            st.subheader("🔁 Top 5 Similar MCPs")
            for i in I[0]:
                st.markdown(f"**•** {texts[i]}")

            st.subheader("🤖 LLM Suggestions")
            model_id = MODEL_OPTIONS[model_choice]
            generator = load_model_and_tokenizer(model_id)
            prompt = f"Given this MCP: {mcp_input}\n\nSuggest relevant tools, APIs or extensions to implement it."
            output = generator(prompt, max_new_tokens=200, do_sample=True)[0]['generated_text']
            st.code(output.split(prompt)[-1].strip(), language='markdown')

st.sidebar.header("📂 MCP Dataset Uploader")
uploaded_file = st.sidebar.file_uploader("Upload JSON file with MCPs", type=["json"])
if uploaded_file:
    try:
        data = json.load(uploaded_file)
        with open(DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
        st.sidebar.success("✅ Uploaded and replaced MCP dataset!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
