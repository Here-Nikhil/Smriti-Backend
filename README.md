# Smriti Backend (FastAPI)

This is the API server. There's no UI here -- it's pure backend, meant to
be called by the (separate) frontend project.

## Local setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Generate your login credentials:
   ```
   python generate_password_hash.py
   ```
   This prints values to paste into your `.env` file.

3. Copy `.env.example` to `.env` and fill in:
   - `GROQ_API_KEY` (from console.groq.com)
   - `APP_USERNAME` / `APP_PASSWORD_HASH` (from step 2)
   - `JWT_SECRET` (any random string)
   - `ALLOWED_ORIGINS` (leave as `http://localhost:3000` for now -- this is where the frontend will run locally)

4. Run the server:
   ```
   uvicorn main:app --reload
   ```

5. Visit `http://localhost:8000/docs` in your browser -- FastAPI auto-generates
   an interactive page where you can test every endpoint (login, upload, chat,
   quiz) without needing the frontend built yet. This is how we'll verify the
   backend works before building any UI on top of it.

## Endpoints

| Method | Path | Auth required | Purpose |
|---|---|---|---|
| POST | /login | no | Get a JWT token |
| POST | /documents/upload | yes | Upload PDF(s) |
| GET | /documents | yes | List indexed documents |
| POST | /chat | yes | Ask a question |
| GET | /chat/history | yes | Get past messages |
| POST | /quiz/generate | yes | Generate MCQ or interactive questions |
| POST | /quiz/grade | yes | Grade one answer |
| GET | /quiz/summary | yes | Final score + weak topics |
| GET | /health | no | Check server is alive |
