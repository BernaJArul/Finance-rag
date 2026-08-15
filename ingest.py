import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Project folders
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

# Chroma collection name
COLLECTION_NAME = "finance_rag"

# Chunking settings required for the assignment
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_pdfs():
    """Read all PDF files from the data folder."""

    documents = []

    pdf_files = sorted(
        file for file in os.listdir(DATA_DIR)
        if file.lower().endswith(".pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in the data folder."
        )

    for filename in pdf_files:
        filepath = os.path.join(DATA_DIR, filename)

        reader = PdfReader(filepath)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_number
                        }
                    )
                )

    return documents


def split_documents(documents):
    """Split PDF text into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)


def create_vector_database(chunks):
    """Create and persist the ChromaDB vector database."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create a fresh collection when indexing
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    # Clear previously indexed documents
    existing = vectorstore.get()

    if existing and existing.get("ids"):
        vectorstore.delete(ids=existing["ids"])

    # Add the new chunks
    vectorstore.add_documents(chunks)

    return vectorstore


def main():
    print("\nFinanceRAG - Document Ingestion")
    print("=" * 40)

    print("Reading PDF files...")
    documents = load_pdfs()

    print(f"Pages loaded: {len(documents)}")

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    print("Creating embeddings and storing in ChromaDB...")
    create_vector_database(chunks)

    print("\nIndexing completed successfully!")
    print(f"PDF files processed: {len(set(d.metadata['source'] for d in documents))}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"ChromaDB folder: {CHROMA_DIR}")


if __name__ == "__main__":
    main()