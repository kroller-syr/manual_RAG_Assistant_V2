"""
Docstring for src.rag_manual
Core RAG engine for maintenance manuals.

This module defines the ManualRAG class, which is responsible for:
-Extracting text from a PDF
-Splitting that text into overlapping chunks
-Encoding chunks as vectors using a sentence-transformer embedding model
-Building a FAISS index over those vectors 
-Retrieving the most relevant chunks for a given user query
"""
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pytesseract
from pdf2image import convert_from_bytes
import pickle


class ManualRAG:
    """
    ManualRAG encapsulates the Retrieval part of a RAG system

    It DOES NOT call the LLM directly. Instead, it:
    -Reads a PDF
    -Prepares an embedding index from the text
    -Retrieves top-k chunks for a query

    The LLM call happens in src/llm_answer.py, which takes these chunks
    and generates a natural language answer. 
    """
    #Create a folder to store FAISS index + chunks on disc so the user
    # wont need to re-index the same manual/PDF everytime the app is closed
    # and then reopened
    INDEX_DIR = Path("indexes")

        #Create a list of saved indexes that can be recalled later without
    #having to rerun the indexer on each subsequent launching of the app

    def list_indexes(self) -> list[str]:
        """List the name of all saved manual indexes on disk
        """
        if not self.INDEX_DIR.exists():
            return[]
        
        names: list[str] = []
        for path in self.INDEX_DIR.glob("*.faiss"):
            #strip off the .faiss extension to get the manual name
            names.append(path.stem)

        return sorted(names)

    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model and empty index."""
        #Load the embedding model once. This can be relatively expensive,
        #which is why we create the ManualRAG instance once and cache it in app. 
        self.embed_model = SentenceTransformer(embed_model_name)
        #Will hold the FAISS index once built.
        self.index: faiss.Index | None = None
        #Will store the list of text chunks corresponding to the index vectors.
        self.chunks: list[str] | None = None
        #Check for previously stored indexes on the disc
        self.current_name: str | None = None

    def extract_text_from_pdf(self, file_obj_or_path) -> str:
        """
        Extract text from a PDF.
        Accepts either a filesystem path or a file-like object (e.g. Streamlit UploadedFile).
        """
        #Handle both "path-like" and "file-like" inputs.
        if isinstance(file_obj_or_path, (str, Path)):
            reader = PdfReader(str(file_obj_or_path))
        else:
            reader = PdfReader(file_obj_or_path)

        pages_text: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        #Join all pages with blank lines between them. 
        return "\n\n".join(pages_text)
    
    #Add OCR functionality for extracting text from PDFs that 
    #are considered images instead of pure text
    def _ocr_pdf(self, file_obj_or_path) -> str:
        """Run OCR on each page and and return combined text."""
        #Point pytesseract to the Tesseract executable in Windows
        #Update the path if you are anothe user trying to use the code locally
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        #Get the raw bytes of the PDF
        if isinstance(file_obj_or_path, (str, Path)):
            #Path-like: open and read bytes
            with open(file_obj_or_path, "rb") as f:
                pdf_bytes= f.read()
        else:
            #Streamlit UploadFile or other file-like:
            #make sure to rewind to the start before reading
            try:
                file_obj_or_path.seek(0)
            except Exception:
                #if seek fails, we just assume we're at the start
                pass
            pdf_bytes = file_obj_or_path.read()

        #Convert PDF pages to images (300dpi for sake of speed)
        images = convert_from_bytes(pdf_bytes, dpi=300)

        ocr_texts: list[str]= []
        for i, img in enumerate(images, start=1):
            page_text = pytesseract.image_to_string(img)
            if page_text.strip():
                ocr_texts.append(page_text)

        #Join all page texts with blank lines between them.
        return "\n\n".join(ocr_texts)        

    @staticmethod
    def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
        """
        Split the full manual text into overlapping charcater based chunks.

        This is a simple first pass, better chunking methods could be used,
        such as by paragraphs, headings, etc. if required/to acheive better results

        :param text: the raw text of the manual
        :param max_chars: maximum number of characters per chunk.
        :param overlap: Number of characters to overlap between consecutive chunks,
                        to avoid cutting important sentences in half.
        :return: List of chunk strings.                 
        """
        chunks: list[str] = []
        start = 0
        n = len(text)
        #Slide a window of max_chars through the text, with overlap
        while start < n:
            end = start + max_chars
            chunk = text[start:end]
            chunks.append(chunk)
            #Move the window forward, but step back by 'overlap' so chunks
            # share some context boundry
            start = end - overlap

        return chunks

    def build_index_from_pdf(self, file_obj_or_path) -> None:
        """
        Build the FAISS index from a PDF.

        Steps:
        1. Extract text from a PDF
        2. Chunk the text
        3. Compute embeddings for each chunk
        4. Build a FAISS index over those embeddings

        :param file_obj_or_path: PDF path or file-like object
        :raises ValueError: if no text can be extracted
        """

        #Step 1. Attempt to run pure text extraction first before 
        # attempting to use OCR if the text extaction if fails
        text = self.extract_text_from_pdf(file_obj_or_path)

        
        #Step 2. If the raw text extraction method fails then fall back to
        # the OCR method to try and extract the text from the image of the PDF
        if not text or not text.strip():
            text = self._ocr_pdf(file_obj_or_path)

        #Step 3. If still no text is able to be extracted throw an exception error
        if not text or not text.strip():
            raise ValueError(
                "No extractable text found in the PDF, even after OCR"
                "The document may be very low quality or unsupported"
            )    

        #Step 4. Chunk the text
        chunks = self.chunk_text(text)
        if not chunks:
            # Fallback in case chunk_text returns an empty list for some reason
            chunks = [text]
        #Save chunks so they can be returned during retrieval
        self.chunks = chunks

        #Step 5. Compute the embeddings for each chunk using the sentence-transformer model
        embeddings = self.embed_model.encode(chunks, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")

        # If we got a single 1-D vector (dim,), make it (1, dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve_chunks(self, query: str, k: int = 5) -> list[str]:
        """
        Retrieve the top-K most relevant text chunks for a given query.

        :param query: User's natural-language question
        :param k: Number of chunks to retrieve
        :return: List of chunk strings, ordered by similarity
        "raises RuntimeError: if build_index_from_pdf hasnt been called yet.
        """

        #Make sure something exist to be indexed
        if self.index is None or self.chunks is None:
            raise RuntimeError("Index not built. Call build_index_from_pdf first.")
        #Embed the query into the same vector as the chunks.
        q_emb = self.embed_model.encode([query]).astype("float32")
        #Search the FAISS index for the k nearest chunk vectors.
        distances, indices = self.index.search(q_emb, k)
        #indices is a 2D array, guard against any out of range indices
        return [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]
    
    def save_index(self, name: str = "current manual" ) -> None:
        """
        Save the FAISS index and chunks list to disk so that they can 
        be reused without re-indexing the same manual everytime
        
        """
        if self.index is None or self.chunks is None:
            raise RuntimeError("No index/chunks to save. Build the index first.")
        
        #Ensure the index directory exists
        self.INDEX_DIR.mkdir(exist_ok=True)

        index_path = self.INDEX_DIR / f"{name}.faiss"
        chunks_path = self.INDEX_DIR / f"{name}_chunks.pk1"

        #Save FAISS index
        faiss.write_index(self.index, str(index_path))

        #Save chunks list using pickle
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks,f)

        #Remember which manual we just saved
        #Allows for custom naming of manuals for easier retrieval
        self.current_name = name

    def load_index(self, name: str)-> bool:
        """
        Load a previously saved FAISS index and chunks list from disk.
        
    
        :param name: Logical name for this manual/index. Must watch what was used in save_index
        :return: True if index was loaded succesfully, False if files were not found.
        
        """
        index_path = self.INDEX_DIR / f"{name}.faiss"
        chunks_path = self.INDEX_DIR / f"{name}_chunks.pk1"

        if not index_path.exists() or not chunks_path.exists():
            return False
        
        #Load FAISS Index
        self.index = faiss.read_index(str(index_path))

        #Load chunks list
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        #Remember which manual is currently active
        self.current_name = name

        return True
    
