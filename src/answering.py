from langchain_core.runnables import RunnableSequence
from langchain_qdrant import QdrantVectorStore
from utils.model_config import get_llm, get_embedding_model, get_qdrant_client
from utils.prompt import get_answer_prompt, get_rephrase_prompt
from langchain_core.documents import Document
from qdrant_client.models import VectorParams, Distance

class QAEngine:
    def __init__(self, collection_name="docs"):
        try:
            self.collection_name = collection_name
            self.llm = get_llm()
            self.embedding = get_embedding_model()
            self.client = get_qdrant_client()

            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )

            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embedding,
                content_payload_key="page_content",
                metadata_payload_key="metadata"
            )

            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
            self.answer_chain = get_answer_prompt() | self.llm
            self.rephrase_chain = get_rephrase_prompt() | self.llm
            self.history = []

        except Exception as e:
            print(f"Error initializing QAEngine: {e}")

    def format_history(self):
        return "\n".join(f"{h['role']}: {h['text']}" for h in self.history[-20:])

    def rephrase(self, question):
        try:
            prompt_input = {
                "question": question,
                "chat_history": self.format_history()
            }
            output = self.rephrase_chain.invoke(prompt_input)

            if hasattr(output, 'content'):
                result = output.content
            elif isinstance(output, str):
                result = output
            elif isinstance(output, dict) and "rephrased" in output:
                result = output["rephrased"]
            else:
                result = str(output)

            if result.startswith("REPHRASED:"):
                rephrased_question = result.replace("REPHRASED:", "").strip()
                print(f"Rephrased question: '{rephrased_question}'")
                return rephrased_question
            elif result.startswith("UNCHANGED:"):
                unchanged_question = result.replace("UNCHANGED:", "").strip()
                print(f"Question unchanged: '{unchanged_question}'")
                return unchanged_question
            else:
                print(f"Unexpected format: '{result}'. Returning original question.")
                return question

        except Exception as e:
            print(f"Error rephrasing question: {e}")
            return question

    def ask(self, user_question, conversation_history=None):
        try:

            if conversation_history is not None:
                self.history = conversation_history.copy()

            self.history.append({"text": user_question, "role": "USER"})

            final_q = self.rephrase(user_question)

            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding,
                content_payload_key="page_content",
                metadata_payload_key="metadata"
            )
            docs = self.vectorstore.similarity_search(final_q, k=6)

            context = "\n\n".join(doc.page_content for doc in docs)

            response = self.answer_chain.invoke({
                "context": context,
                "question": final_q
            })
            answer = response.content if hasattr(response, "content") else str(response)
            self.history.append({"text": answer, "role": "AI"})
            sources = []
            for doc in docs:
                file = doc.metadata.get("file")
                page = doc.metadata.get("page")
                sources.append({"file": file, "page": page})
            sources = [dict(t) for t in {tuple(d.items()) for d in sources}]

            return {
                "answer": answer,
                "sources": sources,
                "conversation_history": self.history
            }

        except Exception as e:
            print(f"Error answering question: {e}")
            return {
                "answer": "Sorry, an error occurred.",
                "sources": [],
                "conversation_history": self.history
            }

    def ingest_documents(self, documents):
        try:
            self.vectorstore.add_documents(documents)
        except Exception as e:
            print(f"Error ingesting documents: {e}")
