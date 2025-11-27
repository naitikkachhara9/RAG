# import os
# from typing import List, Dict
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage

# import dotenv
# dotenv.load_dotenv()

# CHROMA_PERSIST_DIR = "chroma_db"
# TOP_K = 5

# def load_db():
#     embeddings = OpenAIEmbeddings()
#     db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
#     return db

# def retrieve(question: str, top_k: int = TOP_K):
#     db = load_db()
#     results = db.similarity_search_with_score(question, k=top_k)
#     chunks = []
#     for doc, score in results:
#         chunks.append({
#             "text": doc.page_content,
#             "metadata": doc.metadata,
#             "score": score
#         })
#     return chunks

# def build_system_prompt(question: str, chunks: List[Dict]) -> str:
#     context_texts = []
#     for i, c in enumerate(chunks):
#         md = c.get("metadata", {})
#         hdr = f"Source: {md.get('contract_id','unknown')} | section: {md.get('section_title','')} | chunk: {md.get('chunk_id','')}"
#         context_texts.append(f"=== CHUNK {i+1} ===\n{hdr}\n{c['text']}\n")
#     context_block = "\n\n".join(context_texts)
#     system = f"""You are a helpful contract Q&A assistant. Use ONLY the provided context chunks below to answer the user's question.
# If the answer is not in the provided chunks, respond with: "I don't know based on the provided documents."
# Do not hallucinate new facts.

# Context:
# {context_block}

# User question: {question}

# Provide a concise answer and, if applicable, cite the chunk source (contract_id and chunk_id)."""
#     return system

# def call_gemini(system_prompt: str, question: str) -> str:
#     """Call Gemini model with system prompt and question."""
#     model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
#     # Create messages: system message + user question
#     messages = [
#         HumanMessage(content=system_prompt)  # System prompt as first message
#     ]
    
#     resp = model.invoke(messages)
#     return resp.content

# def call_openai_chat(system_prompt: str, question: str) -> str:
#     """Fallback to OpenAI."""
#     from langchain_openai import ChatOpenAI
    
#     model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
#     messages = [
#         HumanMessage(content=system_prompt)
#     ]
#     resp = model.invoke(messages)
#     return resp.content

# def get_answer(question: str):
#     chunks = retrieve(question, top_k=TOP_K)
#     prompt = build_system_prompt(question, chunks)
    
#     try:
#         # Check which API key is configured
#         if os.getenv("GOOGLE_API_KEY"):
#             answer = call_gemini(prompt, question)
#         elif os.getenv("OPENAI_API_KEY"):
#             answer = call_openai_chat(prompt, question)
#         else:
#             raise RuntimeError("No LLM credentials configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
#     except Exception as e:
#         raise RuntimeError(f"LLM call failed: {e}")
    
#     return {
#         "question": question,
#         "retrieved_chunks": chunks,
#         "answer": answer
#     }

# if __name__ == "__main__":
#     q = input("Question: ")
#     out = get_answer(q)
#     print("ANSWER:\n", out["answer"])
#     print("\nRETRIEVED CHUNKS:")
#     for c in out["retrieved_chunks"]:
#         print("-", c["metadata"].get("contract_id"), c["metadata"].get("chunk_id"))


import os
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

import dotenv
dotenv.load_dotenv()

CHROMA_PERSIST_DIR = "chroma_db"
TOP_K = 5

def _get_keys():
    # start with environment (including .env)
    google_key = os.getenv("GOOGLE_API_KEY")

    # if running inside Streamlit, prefer st.secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            # nested tables like [openai] / [google] are supported
            google_key = st.secrets.get("google", {}).get("api_key", st.secrets.get("GOOGLE_API_KEY", google_key))
            # also allow top-level keys
            google_key = st.secrets.get("GOOGLE_API_KEY", google_key)
    except Exception:
        # not running under Streamlit (or import failed) — keep env values
        pass

    return google_key

def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    return db

def retrieve(question: str, top_k: int = TOP_K):
    db = load_db()
    results = db.similarity_search_with_score(question, k=top_k)
    chunks = []
    for doc, score in results:
        chunks.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
    return chunks

def build_system_prompt(question: str, chunks: List[Dict]) -> str:
    context_texts = []
    for i, c in enumerate(chunks):
        md = c.get("metadata", {})
        hdr = f"Source: {md.get('contract_id','unknown')} | section: {md.get('section_title','')} | chunk: {md.get('chunk_id','')}"
        context_texts.append(f"=== CHUNK {i+1} ===\n{hdr}\n{c['text']}\n")
    context_block = "\n\n".join(context_texts)
    system = f"""You are a helpful contract Q&A assistant. Use ONLY the provided context chunks below to answer the user's question.
If the answer is not in the provided chunks, respond with: \"I don't know based on the provided documents.\"
Do not hallucinate new facts.

Context:
{context_block}

User question: {question}

Provide a concise answer and, if applicable, cite the chunk source (contract_id and chunk_id)."""
    return system

def call_gemini(system_prompt: str, question: str) -> str:
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    messages = [HumanMessage(content=system_prompt)]
    resp = model.invoke(messages)
    return resp.content



def get_answer(question: str):
    chunks = retrieve(question, top_k=TOP_K)
    prompt = build_system_prompt(question, chunks)

    try:
        google_key = _get_keys()
        if google_key:
            # If using provider that needs env var, ensure the library can find it.
            os.environ["GOOGLE_API_KEY"] = google_key
            answer = call_gemini(prompt, question)
        else:
            raise RuntimeError("No LLM credentials configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    return {
        "question": question,
        "retrieved_chunks": chunks,
        "answer": answer
    }

if __name__ == "__main__":
    q = input("Question: ")
    out = get_answer(q)
    print("ANSWER:\n", out["answer"])
    print("\nRETRIEVED CHUNKS:")
    for c in out["retrieved_chunks"]:
        print("-", c["metadata"].get("contract_id"), c["metadata"].get("chunk_id"))