import os
import fitz  
import pytesseract
from PIL import Image
from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import streamlit as st


os.environ["OPENAI_API_KEY"] = "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"

# Explicitly set the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# --- Streamlit setup ---
st.set_page_config(page_title="Citi OCR Chatbot", layout="wide")
st.title("🧾 Citi Financial OCR RAG Bot")

# --- PDF OCR + Extraction ---
@st.cache_data(show_spinner="🔍 Scanning and extracting PDFs...")
def extract_docs_from_all_pdfs(folder_path):
    docs = []
    for pdf_file in Path(folder_path).glob("*.pdf"):
        with fitz.open(pdf_file) as doc:
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if not text:  # If text is empty, likely a table/image
                    pix = page.get_pixmap(dpi=300)
                    image_path = f"page_{i+1}.png"
                    pix.save(image_path)
                    text = pytesseract.image_to_string(Image.open(image_path))
                    os.remove(image_path)
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": pdf_file.name, "page": i + 1}
                        )
                    )
    return docs

# --- Chunking ---
@st.cache_data(show_spinner="🔗 Chunking documents...")
def chunk_documents(_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(_docs)

# --- Embedding model ---
@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings()

# --- VectorDB ---
@st.cache_resource
def setup_vectorstore(_chunks,_embeddings):
    return Chroma.from_documents(
        documents=_chunks,
        embedding=_embeddings,
        persist_directory="./ocr_chroma_store"
    )

# --- Prompts ---
prompt = PromptTemplate.from_template("""
Use the following context to answer the question as accurately as possible.

{context}

Question: {question}
Answer:
""")

# --- Load all components (cached) ---
pdf_folder_path = "C:/Users/shashank/Downloads/chatbotpdf"
documents = extract_docs_from_all_pdfs(pdf_folder_path)
chunks = chunk_documents(documents)
embeddings = get_embedding_model()
vectordb = setup_vectorstore(chunks, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 5}, search_type="mmr")

# --- Query interface ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

query = st.text_input("Ask your financial question:")
if query:
    with st.spinner("Fetching"):
        result = qa_chain.invoke({"query": query})
        st.markdown("###  Answer")
        st.write(result["result"])
        st.markdown("###  Sources")
        for doc in result["source_documents"]:
            st.markdown(f"- **{doc.metadata['source']}**, Page {doc.metadata['page']}")