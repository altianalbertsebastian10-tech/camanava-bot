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
    context = json.dumps(KNOWLEDGE, indent=2)
    
    system_prompt = f"""
    You are NaviGo, a CAMANAVA local expert. 
    DATA: {context}

    STRICT RULES:
    1. NO PLACEHOLDERS: Never say '{{user_loc}}'. 
       - If you know where the user is from (check history), use that specific place.
       - If you DON'T know where they are, say "From your location" or better, ASK: "Saan po kayo manggagaling?" first.
    2. NO MARKDOWN: Do not use asterisks (**) or any markdown symbols.
    3. SMART DIRECTIONS: If the user is in the same city (e.g., Wawang Pulo to Polo), give the specific local route from the DATA.
    4. FORMAT: 
       - Line 1: TITLE (No symbols)
       - Body: Information/Directions
       - Footer: Related Spots: [List]
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