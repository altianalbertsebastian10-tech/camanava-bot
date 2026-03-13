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
    # Filter out commercial categories
    heritage_only = [item for item in KNOWLEDGE if item.get("category") not in ["mall", "hotel", "accommodation"]]
    context = json.dumps(heritage_only, indent=2)
    
    system_prompt = f"""
    You are NaviGo, a CAMANAVA heritage and culture expert. 
    DATA: {context}

    CRITICAL UI RULES:
    1. FORMATTING: Use double asterisks (**) for bolding key names and titles. 
    2. NO ITALICS: Do not use single asterisks or underscores.
    3. HERITAGE FOCUS: If a user asks for malls, hotels, or commercial areas, politely explain that NaviGo focuses on history and culture. Suggest a nearby heritage site instead.
    4. ONE CITY ONLY: If the user is vague, ask which specific city they want to explore first.
    5. FIRST LINE = TITLE: The very first line must be a Bold Title (e.g., **Valenzuela Heritage**).
    6. BULLETS: Use only a simple dash (-) for lists.
    """

    try:
        messages = [{"role": "system", "content": system_prompt}]
        for entry in request.history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": request.message})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2
        )
        
        response = completion.choices[0].message.content
        updated_history = request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ]
        
        return {"response": response, "history": updated_history}
    except Exception as e:
        return {"response": f"System Error: {str(e)}", "history": request.history}
    
@app.get("/", response_class=HTMLResponse)
async def get_gui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()