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
    # Pass the JSON context to the AI
    context = json.dumps(KNOWLEDGE, indent=2)
    
    system_prompt = f"""
    You are NaviGo, a CAMANAVA local expert. 
    DATA: {context}

    RULES:
    1. FOCUS: Only talk about Heritage Spots, Info, Tips, and Directions.
    2. SMART DIRECTIONS & MEMORY: 
       - Check history FIRST. If the user already said their location (e.g., Wawang Pulo), use it.
       - If user is in the SAME city (e.g., Wawang Pulo to Polo Museum), give local landmarks/jeep signboards.
    3. NO REPETITION: Do not repeat historical info if already mentioned in history.
    4. ADAPTIVE RESPONSES: Provide ONE title per response followed by info. Move extra spots to the footer.
    5. FORMAT: Always provide a clear TITLE on the first line, then the content. 
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