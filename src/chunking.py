import os
import io
import pdfplumber
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance
from utils.model_config import get_embedding_model, get_qdrant_client

def split_text_into_chunks(text, file_name, page_number, max_chars=1200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "page_content": chunk,
                "metadata": {
                    "file": file_name,
                    "page": page_number,
                }
            })
        start = end
    return chunks

def extract_chunks_from_pdf(file_bytes, file_name):
    chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                chunks.extend(split_text_into_chunks(text.strip(), file_name, i))

            tables = page.extract_tables()
            for idx, table in enumerate(tables, start=1):
                if table:
                    chunks.append({
                        "page_content": table,  
                        "metadata": {
                            "file": file_name,
                            "page": i,
                        }
                    })
    return chunks

def normalize_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        try:
            return "\n".join(
                ["\t".join([str(cell) if cell else "" for cell in row]) for row in content]
            )
        except Exception as e:
            print(f"Error normalizing list content: {e}")
            return str(content)
    else:
        return str(content)

def build_documents(chunks):
    return [
        Document(
            page_content=normalize_content(chunk["page_content"]),
            metadata=chunk.get("metadata", {})
        )
        for chunk in chunks
    ]

async def embed_chunks_from_file(pdf_file, collection_name="docs"):
    try:
        file_name = pdf_file.filename
        file_bytes = await pdf_file.read()

        chunks = extract_chunks_from_pdf(file_bytes, file_name)

        if not chunks:
            raise ValueError(f"No extractable content found in '{file_name}'")

        documents = build_documents(chunks)

        client = get_qdrant_client()
        if client is None:
            raise RuntimeError("Qdrant client failed to initialize")

        embedding_model = get_embedding_model()

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
            content_payload_key="page_content",
            metadata_payload_key="metadata"
        )
        vectorstore.add_documents(documents)

        count = client.count(collection_name=collection_name).count
        print(f"Embedded {count} chunks from '{file_name}' into collection '{collection_name}'.")
        return chunks

    except Exception as e:
        print(f"Error embedding chunks from {pdf_file.filename}: {e}")
        raise e
