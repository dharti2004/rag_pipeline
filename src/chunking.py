import os
import io
import pdfplumber
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders.parsers.pdf import PDFPlumberParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance
from utils.model_config import get_embedding_model, get_qdrant_client


class CustomPDFPlumberLoader:
    def __init__(self, file_path: str, extract_images: bool = True):
        self.file_path = file_path
        self.extract_images = extract_images

    def extract_text_chunks(self, text: str, file_name: str, page_number: int, max_chars: int = 1200) -> List[Dict[str, Any]]:
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

    def format_table(self, table: List[List]) -> str:
        try:
            return "\n".join(
                ["\t".join([str(cell) if cell else "" for cell in row]) for row in table]
            )
        except Exception as e:
            print(f"[Table Format Error] {e}")
            return str(table)

    def load_with_custom_extraction(self, file_bytes: bytes, file_name: str, max_chars: int = 1200) -> List[Dict[str, Any]]:
        chunks = []
        blob = Blob.from_path(self.file_path)
        parser = PDFPlumberParser(extract_images=self.extract_images)
        docs = parser.parse(blob)

        for doc in docs:
            page_num = doc.metadata.get("page", 1)
            content = doc.page_content.strip()
            if content:
                chunks.extend(self.extract_text_chunks(content, file_name, page_num, max_chars))

            images = doc.metadata.get("images", [])
            for image_data in enumerate(images):
                chunks.append({
                    "page_content": image_data,
                    "metadata": {
                        "file": file_name,
                        "page": page_num,
                    }
                })

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for tbl_idx, table in enumerate(tables):
                    if table:
                        formatted_table = self.format_table(table)
                        chunks.append({
                            "page_content": formatted_table,
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
            return "\n".join(["\t".join([str(cell) if cell else "" for cell in row]) for row in content])
        except Exception as e:
            print(f"[Normalize Error] {e}")
            return str(content)
    else:
        return str(content)


def build_documents_from_chunks(chunks_by_type: Dict[str, List[Dict[str, Any]]]) -> List[Document]:
    """Convert each chunk dict into a LangChain Document with minimal metadata."""
    documents = []
    for chunk in chunks_by_type.get("chunks", []):
        metadata = chunk.get("metadata", {})
        minimal_metadata = {
            "file": metadata.get("file"),
            "page": metadata.get("page")
        }
        documents.append(Document(
            page_content=normalize_content(chunk["page_content"]),
            metadata=minimal_metadata
        ))
    return documents


def extract_chunks_from_pdf(file_bytes: bytes, file_name: str) -> Dict[str, List[Dict[str, Any]]]:
    temp_file_path = f"temp_{os.path.basename(file_name)}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_bytes)
    try:
        loader = CustomPDFPlumberLoader(temp_file_path, extract_images=True)
        combined_chunks = loader.load_with_custom_extraction(file_bytes, file_name)
        return {"chunks": combined_chunks}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def embed_chunks_from_file(pdf_file, collection_name="docs"):
    try:
        file_name = pdf_file.filename
        file_bytes = await pdf_file.read()

        chunks_by_type = extract_chunks_from_pdf(file_bytes, file_name)

        if not chunks_by_type.get("chunks"):
            raise ValueError(f"No extractable content found in '{file_name}'")

        documents = build_documents_from_chunks(chunks_by_type)

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
        return chunks_by_type

    except Exception as e:
        print(f"Error embedding chunks from {pdf_file.filename}: {e}")
        raise e
