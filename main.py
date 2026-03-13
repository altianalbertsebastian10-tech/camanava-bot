import os
import json
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Knowledge Base from separate JSON file
def load_knowledge():
    with open("knowledge.json", "r", encoding="utf-8") as f:
        return json.load(f)

KNOWLEDGE = load_knowledge()

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

@app.get("/health")
async def health_check():
    return {"status": "alive", "mode": "JSON_RAG"}

@app.post("/chat")
async def chat(request: ChatRequest):
    # Pass the whole dictionary to ensure the AI sees all city data
    context = json.dumps(KNOWLEDGE, indent=2)
    
    system_prompt = f"""
    You are NaviGo, a polite and proud CAMANAVA heritage expert. 
    DATA: {context}

    STRICT OUTPUT FORMAT:
    - Line 1: TITLE (Use double asterisks: **Title Name**)
    - Line 2: A single brief introductory sentence (NO DASHES OR BULLETS HERE).
    - Line 3+: Use a simple dash (-) followed by a space ONLY for the actual list items.
    - Final Line: A polite closing question or 'Related Spots:' list.

    CRITICAL UI RULES:
    1. FORMATTING: Use double asterisks (**) for bolding key names and titles.
    2. NO ITALICS: Do not use single asterisks or underscores.
    3. HERITAGE FOCUS: If asked for hotels/malls, politely decline and pivot to heritage.
    4. ONE CITY ONLY: If vague, ask which city they want to explore.
    5. FIRST LINE = TITLE: Must be a Bold Title.
    6. BULLETS: Use a simple dash (-) ONLY for list items. Never use them for intro or closing sentences.
    7. NO PLACEHOLDERS: Use 'your location' instead of {{user_loc}}.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                *request.history, # Simplifies passing the history
                {"role": "user", "content": request.message}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        response = completion.choices[0].message.content
        updated_history = request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ]
        
        return {"response": response, "history": updated_history}
    except Exception as e:
        print(f"Server Error: {e}") # This helps you see the error in Render Logs
        return {"response": f"System Error: {str(e)}", "history": request.history}
    
@app.get("/", response_class=HTMLResponse)
async def get_gui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()