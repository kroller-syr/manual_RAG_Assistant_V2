"""
Streamlit web app for querying maintenance manuals in PDF form
using RAG. First iteration works only with text format PDFs, not
PDFs that are scanned as images. 

High-level Overview:
1. User uploads PDF
2. Embedding index is built from its text via ManualRAG
3. User is apply to ask questions via input box
4. We retrieve the most relevant chunks and send them + question 
   to the LLM. 
5. The LLM returns an answer grounded in those chunks.    
"""

import re
from pathlib import Path
import streamlit as st

#ManualRAG handles:
#Extracting the text from a pdf
#Chunking the extracted text
#Building a FAISS index over the embeddings
from src.rag_manual import ManualRAG

#answer_from_chunks handles:
#Building a prompt from the retrieved chunks + user request
#Calls the OpenAI chat model, in this version using gpt-4.1-mini, 
# to generate the answer.
from src.llm_answer import answer_from_chunks

def sanitize_name(name: str) -> str:
    """Turn a user provided manual name into a safe filename string.
    Keeps only letters, numbers, underscores, and dashes"""
    name= name.strip().lower()
    #Replace any sequence of non alphanumeric characters with a single underscore
    name= re.sub(r"[^a-z0-9_-]+","_", name)
    if not name:
        name="manual"
    return name

#Title for the Streamlit app UI, can be changed to anything
st.title("Maintenance Manual Assistant")

"""
Create and cache a single ManualRAG instance for the life of the app.
"""
#Streamlit reruns this script on every interaction, so without 
#caching we would recreate the embedding model  each time. 
#@st.cache_resource  ensres we reuse the same ManualRAG object instead.

@st.cache_resource
def get_rag() -> ManualRAG:
    return ManualRAG()


#Shared RAG engine used for all user interactions
rag = get_rag()
available_manuals=rag.list_indexes()

#Short descriptor for the applet that appears under the title
#Like the title this can be changed to whatever fits the use case best
st.markdown("Upload a maintenance manual PDF and ask questions about it.")

#File uploader widget for the user to provide a PDF.
#Returns a Streamlit UploadedFile object (file-like)
uploaded_pdf = st.file_uploader("Upload a maintenance manual (PDF) or load a previously saved manual", type=["pdf"])

#let the user declare a logical name for the manual being uploaded
#This name will be used for saving/loading the index on future use
manual_name_input = st.text_input(
    "Manual name (used to save/load the index)",
    placeholder="e.g. forklift_model_x_service_manual"
)

#----PDF indexing section------
#When the user has uploaded a PDF and clicks the "Index Manual" button,
# we read the file, build chunks + embeddings, and store them in the
# FAISS index. 
if uploaded_pdf is not None and st.button("Index manual"):
    if not manual_name_input.strip():
        st.error("Please enter a manual name before indexing.")
    else:
        safe_name = sanitize_name(manual_name_input)
        try:
        #Show spinner icon while processing the pdf
            with st.spinner("Reading and indexing manual...This may take longer with scanned PDFs"):
                rag.build_index_from_pdf(uploaded_pdf)
            #after building the index save it to disc
                rag.save_index(safe_name)
        except ValueError as e:
        #ValueError is used by build_index_from_pdf for known issues,
        #e.g. no extractable text (scanned pdf with no OCR)
        #Will implement OCR in a second version of app
            st.error(str(e))
        except Exception as e:
        #Catch all for any unexpected errors during indexing
            st.error(f"Failed to index manual: {e}")
        else:
        #Only show success if no exception was raised.
            st.success("Manual indexed and saved as '{safe_name}'! You can now ask questions.")


st.subheader("Load an existing indexed manual")

available_manuals= rag.list_indexes()

if available_manuals:
    selected_manual = st.selectbox(
        "Choose a saved manual to load",
        options=available_manuals,
        index=available_manuals.index(rag.current_name) if rag.current_name in available_manuals else 0,

    )

    if st.button("Load Selected Manual"):
        try:
            with st.spinner("Loading manual index...."):
                loaded = rag.load_index(selected_manual)
            if loaded:
                st.success(f"Loaded manual: {selected_manual}")
            else:
                st.error("Saved index files not found for this manual")
        except Exception as e:
            st.error(f"Failed to load manual index: {e}")
    else:
        st.info("No saved manual indexes found yet. Index a manual first")

#Show which manual is currently loaded if any loaded from disk
if rag.index is not None and rag.current_name:
    st.markdown(f"**Current loaded manual:** '{rag.current_name}'")

#----Question & Answer Section-------
#Text input where the user types their question about the manual
#Note: this could be any PDF does not need to be only a maintenance 
# manual, could be industry specific/organization material
question = st.text_input("Ask a question about the manual:")

#Q&A is only allowed if the following are true:
# 1. The user has typed a non empty quesiton 
# (input box cant be blank)
# 2. The RAG index has actually been built (rag.index is not NONE)
if question and rag.index is not None:
    #Seperate button to trigger the answer generation
    if st.button("Get answer"):
        with st.spinner("Thinking..."):
            #Retrieve the most relevant chunks for the question
            chunks = rag.retrieve_chunks(question, k=5)
            #Ask the LLM to answer using those chunks as context
            answer = answer_from_chunks(question, chunks)
        #Display the model's answer
        st.markdown("### Answer")
        st.write(answer)

        #Show the underlying retrieved chunks so the user can see
        # what parts of the manual the answer is based on
        with st.expander("Show retrieved context"):
            for i, c in enumerate(chunks, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(c)
 #If the user typed a quesiton but we dont have an index yyet, 
 # remind them that they need to upload + index a PDF first               
elif question and rag.index is None:
    st.info("Please upload and index a manual first.")