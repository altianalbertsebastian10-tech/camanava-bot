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
    You are NaviGo, a CAMANAVA heritage expert. 
    DATA: {context}

    STRICT OUTPUT FORMAT:
    - Line 1: TITLE (Use double asterisks: **Title Name**)
    - Line 2: A single brief introductory sentence.
    - Line 3+: Use a simple dash (-) for bullet points.
    - Final Line: Always start with 'Related Spots:' followed by a comma-separated list.

    STRICT RULES:
    1. NO ITALICS: Never use single asterisks or underscores.
    2. NO PLACEHOLDERS: Never say '{{user_loc}}'. Use 'your location'.
    3. ONE CITY ONLY: If the user is vague, ask which city they want to explore.
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
            max_tokens=600
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