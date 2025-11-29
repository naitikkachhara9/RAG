# Mini RAG — Contract Q&A Application

A Retrieval-Augmented Generation (RAG) application that enables users to ask questions about indexed documents (Service Agreements, NDAs, SLAs, etc.) and receive accurate answers based on the document content.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Keys & Environment Variables](#api-keys--environment-variables)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **Document Ingestion**: Load and process `.txt` and `.md` files from a designated directory
- **Smart Chunking**: Automatically splits documents into meaningful sections with proper metadata
- **Vector Embeddings**: Uses HuggingFace embeddings for efficient similarity search
- **Intelligent Retrieval**: Retrieves top 5 most relevant document chunks based on similarity scores
- **LLM Integration**: Uses Google's Gemini 2.5 Flash model for generating answers
- **Streamlit UI**: Interactive web interface for seamless user experience
- **Conversation History**: Maintains chat history with retrieved chunks for reference
- **Persistent Storage**: Chroma database for persisting indexed documents

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                         │
│                     (Streamlit - app.py)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴──────────────┐
        │                              │
┌───────▼──────────────┐      ┌────────▼──────────────┐
│  INGESTION PIPELINE  │      │  RETRIEVAL & GENERATION│
│   (ingestion.py)     │      │  (retrieval.py)       │
│                      │      │                       │
│ • Load Documents     │      │ • Query Embeddings   │
│ • Split into Chunks  │      │ • Vector Search      │
│ • Generate Embeddings│      │ • Prompt Building    │
│ • Index in Chroma    │      │ • LLM Response       │
└───────┬──────────────┘      └────────┬──────────────┘
        │                              │
        └───────────────┬──────────────┘
                        │
        ┌───────────────▼──────────────┐
        │    CHROMA VECTOR DATABASE    │
        │   (chroma_db/)               │
        │                              │
        │ • Embeddings (HuggingFace)  │
        │ • Document Metadata          │
        │ • Persistent Storage         │
        └──────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Package installer for Python
- **Git** (optional): For cloning the repository

### Step 1: Clone the Repository

```bash
git clone https://github.com/naitikkachhara9/RAG.git
cd RAG
```

### Step 2: Create Virtual Environment

```bash
# On Windows
python -m venv rag_venv
rag_venv\Scripts\activate

# On macOS/Linux
python3 -m venv rag_venv
source rag_venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `streamlit`: Web UI framework
- `langchain`: LLM orchestration
- `langchain-chroma`: Vector database integration
- `langchain-huggingface`: Embedding model
- `langchain-google-genai`: Google Gemini integration
- `python-dotenv`: Environment variable management

---

## ⚙️ Configuration

### Create Environment File

Create a `.env` file in the project root:

```plaintext
# .env file
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional fallback
HUGGINGFACEHUB_API_TOKEN=your_hf_token  # Optional
```

**Important**: Add `.env` to `.gitignore` to prevent accidentally pushing API keys to GitHub.

### Add Documents

Place your documents in the `data/` directory:

```
data/
├── NDA_DataNova_PartnerX_2025.txt
├── Service_Agreement_AlphaTech_BrightRetail_2024.txt
└── SLA_NimbusCloud_EduStream_2024.txt
```

Supported formats: `.txt`, `.md`

### Index Documents

Before running the app for the first time, index your documents:

```bash
python ingestion.py
```

This will create/update the `chroma_db/` directory with indexed embeddings.

---

## 💻 Usage

### Running Locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Using the Interface

1. **Enter your question** in the input field
2. **Click "Ask"** or press Enter
3. **View the answer** below along with:
   - Generated response from Gemini
   - Top 5 retrieved document chunks
   - Metadata for each chunk (contract ID, section title, chunk ID)
   - Similarity scores for each chunk
   - Expandable previews of chunk content

### Example Queries

- "What is the termination notice period in the service agreement?"
- "What are the uptime guarantees mentioned in the SLA?"
- "What confidentiality obligations are mentioned in the NDA?"
- "What are the payment terms?"
- "What is the service availability commitment?"

---

## 📁 Project Structure

```
Simple_RAG/
├── app.py                          # Streamlit UI application
├── ingestion.py                    # Document processing & indexing
├── retrieval.py                    # QA & retrieval logic
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (not committed)
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── data/                           # Input documents directory
│   ├── NDA_DataNova_PartnerX_2025.txt
│   ├── Service_Agreement_AlphaTech_BrightRetail_2024.txt
│   └── SLA_NimbusCloud_EduStream_2024.txt
├── chroma_db/                      # Vector database (auto-created)
│   ├── chroma.sqlite3
│   └── ...
└── rag_venv/                       # Virtual environment (not committed)
```

---

## 🔑 API Keys & Environment Variables

### Google Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Add to `.env`:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

### Streamlit Cloud Secrets

For production deployment on Streamlit Cloud:

1. Go to your app dashboard on Streamlit Cloud
2. Navigate to **Settings** → **Secrets**
3. Add your secrets in TOML format:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   
   [huggingface]
   api_key = "your_hf_token"
   ```

The application automatically checks for Streamlit secrets first, then falls back to environment variables.

---

## 🌐 Deployment

### Deploy on Streamlit Cloud

1. **Push code to GitHub** (ensure `.env` is in `.gitignore`):
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

3. **Create new app**:
   - Select your GitHub repository
   - Select `main` branch
   - Set main file to `app.py`

4. **Configure secrets**:
   - Click "Settings" in the app dashboard
   - Add your API keys in the Secrets section

5. **Deploy** - Streamlit will automatically build and serve your app

### Deployment Requirements

- GitHub repository (public or private)
- Valid Google API key for Gemini
- Sufficient Chroma database size for your documents

---

## 🔧 How It Works

### 1. Document Ingestion (`ingestion.py`)

- **Load**: Reads `.txt` and `.md` files from `data/` directory
- **Parse**: Splits documents by section headers (marked with `---`)
- **Chunk**: Uses `RecursiveCharacterTextSplitter`:
  - Chunk size: 1024 characters
  - Overlap: 128 characters
  - Preserves natural breaks (paragraphs, sentences)
- **Embed**: Generates embeddings using HuggingFace's `all-MiniLM-L6-v2` model
- **Index**: Stores in Chroma vector database with metadata

### 2. Query Processing (`retrieval.py`)

- **Embed Query**: Converts user question to embedding
- **Search**: Finds top 5 most similar chunks using cosine similarity
- **Rank**: Returns results with similarity scores
- **Build Prompt**: Constructs system prompt with context chunks
- **Generate**: Sends to Google Gemini for answer generation
- **Return**: Provides answer + retrieved chunks + metadata

### 3. User Interface (`app.py`)

- Built with Streamlit for simplicity
- Displays conversation history
- Shows retrieved chunks with previews
- Allows downloading answers
- Manages session state for conversation tracking

---

## 🐛 Troubleshooting

### Error: "Collection expecting embedding with dimension of 384, got 768"

**Cause**: Mismatch between embedding models used during ingestion and retrieval.

**Solution**: Ensure both `ingestion.py` and `retrieval.py` use the same model:
```python
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Dimension: 384
```

### Error: "No LLM credentials configured"

**Cause**: `GOOGLE_API_KEY` not set.

**Solution**:
- Local: Create `.env` file with `GOOGLE_API_KEY=...`
- Cloud: Add to Streamlit secrets

### Error: "Module not found"

**Cause**: Dependencies not installed.

**Solution**:
```bash
pip install -r requirements.txt
```

### Empty or Poor Quality Answers

**Cause**: Retrieved chunks don't contain relevant information.

**Solution**:
- Ensure documents are in `data/` folder
- Re-run `python ingestion.py` after adding documents
- Use more specific queries
- Check chunk previews to see retrieved content

### Slow Response Time

**Cause**: Large number of chunks or slow embedding model.

**Solution**:
- Reduce `TOP_K` in `retrieval.py` (default: 5)
- Optimize chunk size in `ingestion.py`
- Use faster embedding model (if needed)

---

## 📊 Performance Notes

- **Embedding Time**: ~100-200ms per query (HuggingFace model runs locally)
- **LLM Response Time**: ~2-5 seconds (depends on Google Gemini API)
- **Database Size**: ~10KB per indexed chunk
- **Optimal Chunk Size**: 512-1024 characters for contracts

---

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the application.

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Naitik Kachhara**

- GitHub: [@naitikkachhara9](https://github.com/naitikkachhara9)
- Project: [RAG](https://github.com/naitikkachhara9/RAG)

---

## 📚 Additional Resources

- [LangChain Documentation](https://docs.langchain.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Chroma Documentation](https://docs.trychroma.com)
- [Google Gemini API](https://ai.google.dev)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

**Last Updated**: November 29, 2025
