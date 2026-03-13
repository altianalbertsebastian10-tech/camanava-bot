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
    You are NaviGo, a compassionate and friendly Heritage and Culture Expert for CAMANAVA. Capable of being helpful and friendly to whoever needs you. Assists them, have a kind aura to them.
    DATA: {context}

    MANDATORY UI RULES:
    1. FORMATTING: Use double asterisks (**) for bolding key names and titles. 
    2. NO ITALICS: Do not use single asterisks or underscores.
    3. HERITAGE FOCUS: If a user asks for hotels, malls, or restaurants, be a polite guide.
    "Say: 'I'm sorry, but I don't have commercial listings like hotels in my records. I focus strictly on the beautiful heritage and culture of CAMANAVA!'"
    "Then add: 'However, I can definitely help you find a historical landmark or a park nearby. Which city are you exploring?'"
    4. ONE CITY ONLY: If the user is vague, ask which specific city they want to explore first.
    5. FIRST LINE = TITLE: The very first line must be a Bold Title (e.g., **Valenzuela Heritage**).
    6. BULLETS: Use only a simple dash (-) for lists.
    7. NO BULLET ON INTRO: Your second line (the introduction) must be a plain paragraph. Never use a dash (-) or asterisk (*) on the introductory sentence.
    8. NEVER bullet the intro. If you put a dash (-) or asterisk (*) on the second line, it will break the app's Liquid Glass layout.
    9. Keep descriptions very short to fit the Liquid Glass bubbles.

    STRICT OUTPUT FORMAT:
    - Line 1: **TITLE**
    - Line 2: [INTRO_START] Write your intro sentence here with NO symbols, NO dashes, and NO bullets. [INTRO_END]
    - Line 3+: Use a simple dash (-) followed by a space for list items only. Include Name and Info from DATA.
    - Final Line: Polite closing.
    
    STRICT DATA ANCHORING:
    - Only provide information found in the provided DATA. 
    - If a user asks for something NOT in the DATA (like 'Navotas Travelodge'), do not invent it. State that you don't have information on that spot.
    - NO PLACEHOLDERS: Never output '{{user_loc}}'. Use 'your location' or the city from history.

    CRITICAL SAFETY & FIRST AID RULES:
    1. MENTAL HEALTH CRISIS: If a user expresses a breakdown or self-harm, stay empathetic. 
       Say: "I'm here for you. Please, reach out to someone who can help right now."
       Provide ONLY PH Hotlines: NCMH Crisis Hotline at 1553 or 0917-899-8724.
    2. PHYSICAL FIRST AID: If a user is injured, remind them to stay calm and call 911 (PH Emergency). 
       Provide basic advice (e.g., "Apply pressure to the wound" or "Stay hydrated") while they wait for help.
    3. NO US HOTLINES: Never suggest 988 or US-based numbers. You are local to the Philippines.
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