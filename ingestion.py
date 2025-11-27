# import os
# from pathlib import Path
# import json
# # import pdfplumber

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.documents import Document
# from dotenv import load_dotenv
# load_dotenv()

# # CONFIG
# DATA_DIR = Path("data")            # put your docs here (txt, md, pdf)
# CHROMA_PERSIST_DIR = "chroma_db"  # persisted DB dir
# # EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI embedding model (adjust if needed)

# # Helper: simple PDF -> text (pdfplumber)
# # def load_pdf_text(path: Path) -> str:
# #     text_parts = []
# #     with pdfplumber.open(path) as pdf:
# #         for p in pdf.pages:
# #             text_parts.append(p.extract_text() or "")
# #     return "\n\n".join(text_parts)

# def load_file(path: Path) -> str:
#     if path.suffix.lower() in [".txt", ".md"]:
#         return path.read_text(encoding="utf-8", errors="ignore")
#     # elif path.suffix.lower() == ".pdf":
#     #     return load_pdf_text(path)
#     else:
#         raise ValueError("Unsupported file type: " + str(path))

# def make_documents():
#     docs = []
#     for file in sorted(DATA_DIR.glob("*")):
#         try:
#             text = load_file(file)
#         except Exception as e:
#             print(f"skip {file}: {e}")
#             continue

#         # naive section split: try to split by headings '##' or 'Section' or fallback
#         sections = []
#         if "\n## " in text or "\n# " in text:
#             # split on top-level headings
#             parts = [p.strip() for p in text.split("\n## ") if p.strip()]
#             # ensure file name included
#             sections = [(f"section_{i+1}", p) for i,p in enumerate(parts)]
#         else:
#             sections = [("full", text)]

#         for sec_title, sec_text in sections:
#             docs.append({
#                 "source_path": str(file),
#                 "contract_id": file.stem,
#                 "section_title": sec_title,
#                 "text": sec_text
#             })
#     return docs

# def chunk_and_index():
#     documents = make_documents()
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1024,
#         chunk_overlap=128,
#         separators=["\n\n", "\n", " ", ""]
#     )

#     lc_docs = []
#     for doc in documents:
#         chunks = splitter.split_text(doc["text"])
#         for i, chunk in enumerate(chunks):
#             metadata = {
#                 "contract_id": doc["contract_id"],
#                 "section_title": doc["section_title"],
#                 "chunk_id": f"{doc['contract_id']}_chunk_{i}",
#                 "source_path": doc["source_path"],
#                 "position": i
#             }
#             lc_docs.append(Document(page_content=chunk, metadata=metadata))

#     embeddings = OpenAIEmbeddings()  # reads OPENAI_API_KEY from env
    
#     # create Chroma DB (automatically persists to persist_directory)
#     db = Chroma.from_documents(
#         documents=lc_docs,
#         embedding=embeddings,
#         persist_directory=CHROMA_PERSIST_DIR
#     )
#     # No need to call db.persist() - it's automatic now
#     print(f"Indexed {len(lc_docs)} chunks into Chroma at {CHROMA_PERSIST_DIR}")

# if __name__ == "__main__":
#     chunk_and_index()


import os
from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# CONFIG
DATA_DIR = Path("data")            # put your docs here (txt, md, pdf)
CHROMA_PERSIST_DIR = "chroma_db"  # persisted DB dir

def _prefer_streamlit_secrets():
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            hf_key = st.secrets.get("huggingface", {}).get("api_key", st.secrets.get("HUGGINGFACEHUB_API_TOKEN"))
            if hf_key:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key
    except Exception:
        pass

_prefer_streamlit_secrets()

def load_file(path: Path) -> str:
    if path.suffix.lower() in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type: " + str(path))

def make_documents():
    docs = []
    for file in sorted(DATA_DIR.glob("*")):
        try:
            text = load_file(file)
        except Exception as e:
            print(f"skip {file}: {e}")
            continue

        sections = []
        if "\n## " in text or "\n# " in text:
            parts = [p.strip() for p in text.split("\n## ") if p.strip()]
            sections = [(f"section_{i+1}", p) for i,p in enumerate(parts)]
        else:
            sections = [("full", text)]

        for sec_title, sec_text in sections:
            docs.append({
                "source_path": str(file),
                "contract_id": file.stem,
                "section_title": sec_title,
                "text": sec_text
            })
    return docs

def chunk_and_index():
    documents = make_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        separators=["\n\n", "\n", " ", ""]
    )

    lc_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            metadata = {
                "contract_id": doc["contract_id"],
                "section_title": doc["section_title"],
                "chunk_id": f"{doc['contract_id']}_chunk_{i}",
                "source_path": doc["source_path"],
                "position": i
            }
            lc_docs.append(Document(page_content=chunk, metadata=metadata))

    # Use HuggingFace embeddings (runs locally, no API key needed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print(f"Indexed {len(lc_docs)} chunks into Chroma at {CHROMA_PERSIST_DIR}")

if __name__ == "__main__":
    chunk_and_index()