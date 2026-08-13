import os
import json
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import threading
import time
import requests

def self_ping():
    time.sleep(20)
    while True:
        try:
            requests.get("https://camanava-bot.onrender.com/health")
        except:
            pass
        time.sleep(600)

threading.Thread(target=self_ping, daemon=True).start()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_knowledge():
    with open("knowledge.json", "r", encoding="utf-8") as f:
        return json.load(f)

KNOWLEDGE = load_knowledge()

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

@app.get("/health")
@app.head("/health")
@app.post("/health")
async def health_check():
    return {"status": "alive", "mode": "JSON_RAG"}

@app.post("/chat")
def chat(request: ChatRequest):
    user_msg = request.message.lower()

    relevant_data = {}
    target_city = None

    for city in ["caloocan", "malabon", "navotas", "valenzuela"]:
        if city in user_msg:
            target_city = city
            break

    if not target_city and request.history:
        for entry in reversed(request.history):
            content = entry['content'].lower()
            for city in ["caloocan", "malabon", "navotas", "valenzuela"]:
                if city in content:
                    target_city = city
                    break
            if target_city:
                break

    if target_city:
        relevant_data = {target_city: KNOWLEDGE.get(target_city, [])}
        context = json.dumps(relevant_data, indent=2)
    else:
        # Give her a clear directive instead of an error state
        context = "NO CITY SPECIFIED. You must ask the user to choose a specific city (Caloocan, Malabon, Navotas, or Valenzuela) before you can search your database for recommendations."

   # --- RAG STEP 2: AUGMENT ---
    system_prompt = f"""You are Navi, a cheerful, warm, and natural AI tourism guide for the CAMANAVA region (Caloocan, Malabon, Navotas, Valenzuela). 
    
    USER QUERY: {request.message}
    
    VERIFIED DATABASE FACTS:
    {context}
    
    CONVERSATIONAL RULES:
    1. BE NATURAL: You are a conversational AI. If the user simply says "hi", complains, or wants to chat casually, respond with empathy and natural conversation. 
    2. DROP THE LOOP: Do NOT force the user to pick a city in every single message. Let the conversation flow organically. 
    3. BREAK THE FOURTH WALL: If the user identifies as your developer, asks about your API, or asks technical questions, playfully acknowledge them! 
    4. FACTUAL TOURISM: When you DO recommend places, ONLY use the VERIFIED DATABASE FACTS. If the facts are empty or ask for a city, follow those instructions exactly.
    5. MOBILE FORMATTING: Keep your recommendations concise and easy to read on a phone. Use short bullet points. NEVER use markdown tables.
    """

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                *request.history,
                {"role": "user", "content": request.message}
            ],
            temperature=0.1,
            max_tokens=1024 # Increased from 500 to prevent cut-offs
        )

        response = completion.choices[0].message.content
        response = response.replace("{{user_loc}}", "your area")

        updated_history = request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ]

        return {"response": response, "history": updated_history[-10:]}
    except Exception as e:
        return {"response": "I'm having trouble accessing my records. Try again?", "history": request.history}

@app.get("/", response_class=HTMLResponse)
async def get_gui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()