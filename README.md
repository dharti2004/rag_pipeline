# RAG Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **FastAPI**, **LangChain**, and **Qdrant** to enable question-answering over uploaded PDF documents. It supports document ingestion, vector storage, and conversational query processing with context-aware responses.

## Features
- **PDF Ingestion and Chunking**: Upload PDF files, extract text and tables, and split them into manageable chunks for embedding.
- **Vector Storage**: Uses Qdrant as a vector database to store document embeddings for efficient retrieval.
- **Question Answering**: Leverages a Google Gemini language model to provide contextually relevant answers based on retrieved document chunks.
- **Conversation History**: Maintains conversation context for multi-turn interactions with question rephrasing capabilities.
- **Document Deletion**: Allows deletion of document vectors from the Qdrant database by filename.

## Prerequisites
- **Python**: Version 3.10 or higher.
- **Qdrant**: Local Qdrant instance for vector storage.
- **Google API Key**: Required for the Gemini language model.
- **Dependencies**: Install required packages listed in `requirements.txt`.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dharti2004/rag_pipeline.git
   cd rag_pipeline
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your Google API key:
   ```bash
   API_KEY=your_google_api_key
   ```

5. **Run Qdrant Locally**:
   Ensure Qdrant is running locally. You can use Docker:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Usage
1. **Run the FastAPI Server**:
   ```bash
   uvicorn main:app --reload
   ```
   The server will be available at `http://localhost:8000`.

2. **API Endpoints**:
   - **Upload PDFs**:
     ```
     POST /upload
     Content-Type: multipart/form-data
     Body: files (List of PDF files)
     ```
     Uploads PDF files, extracts text and tables, and stores embeddings in Qdrant.

   - **Ask Questions**:
     ```
     POST /ask
     Content-Type: multipart/form-data
     Body: question (str), conversation_history (optional JSON string)
     ```
     Submits a question and optionally includes conversation history. Returns an answer, sources, and updated history.

   - **Delete Vectors**:
     ```
     POST /delete
     Content-Type: multipart/form-data
     Body: filename (str)
     ```
     Deletes vectors associated with a specific file from the Qdrant collection.

3. **Example Workflow**:
   - Upload a PDF document using the `/upload` endpoint.
   - Ask a question about the document using the `/ask` endpoint.
   - Delete the document's vectors using the `/delete` endpoint if no longer needed.

## Project Structure
```
rag_pipeline/
├── src/
│   ├── chunking.py        # PDF parsing and chunking logic
│   ├── answering.py       # Question-answering engine with conversation handling
│   ├── utils/
│   │   ├── model_config.py # LLM, embedding model, and Qdrant client setup
│   │   ├── prompt.py       # Prompt templates for answering and rephrasing
│   │   ├── delete.py       # Logic for deleting vectors from Qdrant
├── main.py                # FastAPI application
├── requirements.txt       # Project dependencies
```

## Dependencies
Key dependencies include:
- `fastapi`: For the API server.
- `langchain`: For document processing, embeddings, and LLM integration.
- `qdrant_client`: For vector storage and retrieval.
- `pdfplumber`: For PDF text and table extraction.
- See `requirements.txt` for the full list.

## Notes
- Ensure only one process accesses the Qdrant storage at a time to avoid lock errors.
- The pipeline uses the `sentence-transformers/all-MiniLM-L6-v2` model for embeddings and `gemini-2.5-flash` for language generation.
- Conversation history is maintained for up to 20 messages to provide context for rephrasing and answering.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License.
```