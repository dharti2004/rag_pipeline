from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from src.chunking import embed_chunks_from_file  
from src.answering import QAEngine 
from utils.delete import delete_vectors_by_file
import json

app = FastAPI()
_qa_engine = None

def get_qa_engine():
    global _qa_engine
    if _qa_engine is None:
        _qa_engine = QAEngine()
    return _qa_engine

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if not file.filename:
            continue
        try:
            chunks = await embed_chunks_from_file(file)
            results.append({
                "message": f"Embedded chunks from '{file.filename}' successfully.",
                "file": file.filename,
                "chunks": chunks
            })

        except Exception as e:
            results.append({
                "message": f"Failed to embed chunks from '{file.filename}'",
                "error": str(e)
            })

    return JSONResponse(content=results)


@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    conversation_history: Optional[str] = Form(None)
):
    try:
        qa = get_qa_engine()
        history = None
        if conversation_history:
            try:
                parsed = json.loads(conversation_history)
                if not isinstance(parsed, list):
                    raise ValueError("History should be a list of messages")
                history = parsed
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid conversation_history format: {e}")

        result = qa.ask(question, history)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "conversation_history": result.get("conversation_history", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/delete")
async def delete_vectors(filename: str = Form(...)):
    try:
        delete_vectors_by_file(filename)
        return {"message": f"Deleted vectors for file '{filename}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))