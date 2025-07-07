from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from utils.model_config import get_qdrant_client

def delete_vectors_by_file(file_name: str, collection_name: str = "docs"):
    try:
        client = get_qdrant_client()
        if not client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return False

        condition = Filter(must=[
            FieldCondition(
                key="metadata.file",
                match=MatchValue(value=file_name)
            )
        ])
        client.delete(collection_name=collection_name, points_selector=condition)
        print(f"Deleted vectors for file: {file_name}")
        return True

    except Exception as e:
        print(f"Error deleting vectors for file {file_name}: {e}")
        return False
