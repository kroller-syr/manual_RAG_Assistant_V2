# Maintenance Manual RAG Assistant v2

A Streamlit app that lets you **ask questions about maintenance manuals in PDF format**, using a Retrieval-Augmented Generation (RAG) pipeline — now with:

- ✅ OCR support for scanned/image-only PDFs  
- ✅ Persistent, named FAISS indexes saved to disk  
- ✅ UI for loading previously indexed manuals  

You upload a PDF, give it a manual name, the app builds an embedding index, and then an OpenAI model (`gpt-4.1-mini`) answers your questions using the most relevant sections of the document.

---

## Features

- 📝 **Upload a PDF** maintenance/service manual
- 🔍 **Text extraction**:
  - Native text layer via `pypdf`
  - Fallback to **OCR** (Tesseract + `pdf2image`) for scanned/image-only PDFs
- 📐 **Embeddings** via `sentence-transformers` (`all-MiniLM-L6-v2`)
- 📚 **Vector search** using FAISS over chunked manual text
- 🤖 **Answer generation** using OpenAI’s `gpt-4.1-mini`
- 💾 **Persistent indexes**:
  - Indexes + chunks saved under a user-defined manual name
  - Stored in an `indexes/` directory on disk
- 📂 **Multiple manuals**:
  - Name each manual when indexing
  - Load any previously indexed manual from a dropdown
- 🖥 **Streamlit web UI**:
  - Upload + name manual
  - “Index manual” button (with OCR-aware progress)
  - “Load an existing indexed manual” section
  - Question box + clear, step-by-step answers
  - Display of which manual is currently loaded

---

## How it works (high level)

1. **Upload** a PDF through the Streamlit UI and enter a **manual name**.
2. The app tries to **extract text**:
   - First via `pypdf` (fast; uses the PDF text layer).
   - If no text is found, it falls back to **OCR**:
     - Renders each page to an image (`pdf2image`)
     - Runs Tesseract via `pytesseract` to extract text.
3. The combined text is **split into overlapping chunks**.
4. Each chunk is **embedded** with a SentenceTransformer model.
5. The embeddings are stored in a **FAISS index**, alongside the chunk list.
6. The index + chunks are **saved to disk** under the given manual name:
   - `indexes/<manual_name>.faiss`
   - `indexes/<manual_name>_chunks.pkl`
7. When you ask a question:
   - The question is embedded.
   - FAISS returns the **top-k most similar chunks**.
   - Those chunks + your question are passed to `gpt-4.1-mini`.
   - The model returns a **step-by-step answer** grounded in the manual.

On later runs, you can **load any previously indexed manual** from a dropdown and start asking questions immediately, without re-indexing the PDF.

---

## Prerequisites

You’ll need:

- **Python 3.10+**
- **pip** (Python package manager)
- An **OpenAI API key** with billing/quota enabled
- **Tesseract OCR** installed on your system
- **Poppler** for Windows (for `pdf2image` to render PDFs as images)
- (Optional but recommended) **virtual environment** (`venv`)

The instructions below assume Windows/PowerShell, but can be adapted to macOS/Linux.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/manual_RAG_Assistant_V2.git
cd manual_RAG_Assistant_V2

### 2. Create and activate a virtual environment (instructions for windows)
pyhton -m venv .venv
.venv\Scripts\Activate.ps1

### For use in Linux use the following instead
python -m venv .venv
source .venv/bin/activate

### 3. Install the required dependencies
pip install -r requirements.txt

### 4. Install Tesseract OCR
### Will need to set to PATH in windows

###5. Install Poppler (for pdf2image)
### Also needs to be set to PATH in windows

### 6. Set up your OpenAI key
### This will require you to create a file name ".env" in the project in the same folder as "app.py"
### NOTE: this step does require that you have billing/quotos enabled through OpenAI for API usauge. 
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

### 7. Run the app
### With the virtual environment active and ".env" set up run:
streamlight run app.py


###---------Limitations and Future Improvements---------####
#1. OCR quality is dependent on the scan quality of the pdf being used and the Tesseract language/data configuration
#2. Large scanned PDFs (100mb+) can be VERY SLOW on the initial index, though loading from the saved index is almost instant
#3. Indexes are stored per manual name, for this iteration there is no rename/delete functionality built into the app (though the index can be manually deleted from the disc)
#4. Currently no cross-manual queries are possible, as each manual is indexed seperately.
###----Future improvements for the project----###
#1. Clean up OCR text to normalize whitespace and remove weird characters
#2. Extract and display relevant images from the scanned PDF
#3. Create better UI for the app side of the project to allow in-app re-naming and deletion of previously saved indexes.
#4. Create a way to view index stats such as size and chunk count
#5. Allow for cross-index queries to prevent user from having to change indexes when asking questions 
