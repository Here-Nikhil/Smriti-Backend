"""
main.py
-------
The FastAPI backend. Run locally with: uvicorn main:app --reload

This replaces what app.py did in Streamlit, except now it's just an API --
no UI rendering happens here at all. The frontend (separate project) calls
these endpoints and decides how to display the results.

In-memory storage note:
Like the Streamlit version, each user's uploaded documents and chat history
live in memory on the server, keyed by username. This resets if the server
restarts (e.g. Render's free tier sleeping after inactivity). That's an
acceptable tradeoff for a personal/demo project -- a "v2" upgrade would be
persisting this to a real database.
"""

import os
import tempfile
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()  # reads .env in this folder, if present -- no-op on Render, which uses real env vars

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

from auth import verify_credentials, create_access_token, get_current_user
from rag_engine import VectorStore, build_chunks_from_pdfs, format_context_with_citations, get_embedding_model
from quiz_engine import generate_questions, grade_answer, generate_mcq_questions, grade_mcq_answer

app = FastAPI(title="Smriti API", docs_url=None)


@app.on_event("startup")
def preload_embedding_model():
    """Downloads/loads the embedding model once, at boot, before the server
    starts accepting traffic -- avoids the race where a slow first download
    could overlap with an incoming request trying to use a half-written file."""
    get_embedding_model()

from fastapi.openapi.docs import get_swagger_ui_html

# Small inline SVG brain icon as a data URI -- avoids needing a static file
# host just for a favicon, and replaces FastAPI's default black-circle icon.
FAVICON_DATA_URI = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 48'%3E"
    "%3Ccircle cx='32' cy='16' r='3' fill='%232BBE8C'/%3E"
    "%3Ccircle cx='32' cy='28' r='3' fill='%232BBE8C'/%3E"
    "%3Ccircle cx='14' cy='10' r='2.5' fill='%232BBE8C'/%3E"
    "%3Ccircle cx='50' cy='10' r='2.5' fill='%232BBE8C'/%3E"
    "%3C/svg%3E"
)

@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Smriti API",
        swagger_favicon_url=FAVICON_DATA_URI,
    )

# ---------------------------------------------------------------------------
# CORS: by default, browsers block JS on one domain from calling a server on
# a different domain. Since the frontend (Vercel) and backend (Render) will
# live on different domains, we have to explicitly allow it here.
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
LLM_MODEL = "llama-3.3-70b-versatile"

# In-memory per-user state. user_data[username] = {"store": VectorStore, "messages": [...]}
user_data = {}


def get_user_state(username: str):
    if username not in user_data:
        user_data[username] = {"store": VectorStore(), "messages": [], "quiz": None}
    return user_data[username]


# ---------------------------------------------------------------------------
# Request/response shapes -- FastAPI uses these to validate incoming JSON
# automatically and to auto-generate API docs at /docs
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str

class QuizGenerateRequest(BaseModel):
    num_questions: int
    mode: str  # "mcq" or "interactive"

class QuizGradeRequest(BaseModel):
    question_index: int
    answer: Optional[str] = None          # for interactive mode
    selected_index: Optional[int] = None  # for MCQ mode


# ---------------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------------
@app.post("/login")
def login(req: LoginRequest):
    if not verify_credentials(req.username, req.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(req.username)
    return {"access_token": token, "token_type": "bearer"}


# ---------------------------------------------------------------------------
# DOCUMENTS
# ---------------------------------------------------------------------------
@app.post("/documents/upload")
def upload_documents(files: List[UploadFile] = File(...), user: str = Depends(get_current_user)):
    state = get_user_state(user)
    tmp_paths = []
    for f in files:
        tmp_path = os.path.join(tempfile.gettempdir(), f.filename)
        with open(tmp_path, "wb") as out:
            out.write(f.file.read())
        tmp_paths.append((tmp_path, f.filename))

    chunks = build_chunks_from_pdfs(tmp_paths)
    state["store"].add(chunks)
    return {"indexed_files": [name for _, name in tmp_paths], "total_chunks": len(state["store"].chunks)}


@app.get("/documents")
def list_documents(user: str = Depends(get_current_user)):
    state = get_user_state(user)
    sources = sorted(set(c.source for c in state["store"].chunks))
    return {"documents": sources}


@app.delete("/documents")
def clear_documents(user: str = Depends(get_current_user)):
    """Resets this user's vector store and chat history -- lets you start
    fresh without needing to restart the whole backend."""
    user_data[user] = {"store": VectorStore(), "messages": [], "quiz": None}
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# CHAT MODE
# ---------------------------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest, user: str = Depends(get_current_user)):
    state = get_user_state(user)
    if not state["store"].chunks:
        raise HTTPException(status_code=400, detail="Upload a PDF first")

    results = state["store"].search(req.question, k=4)
    context, citations = format_context_with_citations(results)

    history = state["messages"][-6:]
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided document excerpts. If the answer isn't in the excerpts, "
        "say you don't know. When you use information from a source, mention "
        "which source number it came from, like (Source 2)."
    )
    user_prompt = f"Conversation so far:\n{history_text}\n\nDocument excerpts:\n{context}\n\nQuestion: {req.question}"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    state["messages"].append({"role": "user", "content": req.question})
    state["messages"].append({"role": "assistant", "content": answer})

    return {"answer": answer, "citations": citations}


@app.get("/chat/history")
def chat_history(user: str = Depends(get_current_user)):
    return {"messages": get_user_state(user)["messages"]}


# ---------------------------------------------------------------------------
# PREP MODE (MCQ + interactive)
# ---------------------------------------------------------------------------
@app.post("/quiz/generate")
def quiz_generate(req: QuizGenerateRequest, user: str = Depends(get_current_user)):
    state = get_user_state(user)
    if not state["store"].chunks:
        raise HTTPException(status_code=400, detail="Upload a PDF first")

    if req.mode == "mcq":
        questions = generate_mcq_questions(client, LLM_MODEL, state["store"].chunks, req.num_questions)
    else:
        questions = generate_questions(client, LLM_MODEL, state["store"].chunks, req.num_questions)

    state["quiz"] = {"mode": req.mode, "questions": questions, "results": {}}
    # Don't leak the correct answer index to the client for MCQs
    safe_questions = []
    for q in questions:
        safe_q = {k: v for k, v in q.items() if k != "correct_index"}
        safe_questions.append(safe_q)

    return {"questions": safe_questions}


@app.post("/quiz/grade")
def quiz_grade(req: QuizGradeRequest, user: str = Depends(get_current_user)):
    state = get_user_state(user)
    quiz = state.get("quiz")
    if not quiz:
        raise HTTPException(status_code=400, detail="No active quiz -- call /quiz/generate first")

    question = quiz["questions"][req.question_index]

    if quiz["mode"] == "mcq":
        if req.selected_index is None:
            raise HTTPException(status_code=400, detail="selected_index is required for MCQ grading")
        result = grade_mcq_answer(question, req.selected_index)
    else:
        if not req.answer:
            raise HTTPException(status_code=400, detail="answer is required for interactive grading")
        result = grade_answer(client, LLM_MODEL, question["question"], question["source_text"], req.answer)

    quiz["results"][req.question_index] = result
    return result


@app.get("/quiz/summary")
def quiz_summary(user: str = Depends(get_current_user)):
    state = get_user_state(user)
    quiz = state.get("quiz")
    if not quiz:
        raise HTTPException(status_code=400, detail="No active quiz")

    total = len(quiz["questions"])
    results = quiz["results"]
    scores = [results[i]["score"] for i in range(total) if i in results]
    correct_count = sum(1 for s in scores if s >= 7)
    percentage = round(sum(scores) / (total * 10) * 100) if total and scores else 0

    weak_topics = [
        quiz["questions"][i]["question"]
        for i in range(total)
        if i in results and results[i]["score"] < 7
    ]

    return {
        "correct_count": correct_count,
        "total": total,
        "percentage": percentage,
        "weak_topics": weak_topics,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    """
    Universal voice input: works the same on Chrome, Safari, Firefox, mobile --
    because the browser only needs to record audio (MediaRecorder, supported
    everywhere) instead of doing speech recognition itself (which isn't).
    Transcription happens here, server-side, using Groq's free Whisper model.
    """
    audio_bytes = await file.read()
    transcription = client.audio.transcriptions.create(
        file=(file.filename or "audio.webm", audio_bytes),
        model="whisper-large-v3",
    )
    return {"text": transcription.text}
