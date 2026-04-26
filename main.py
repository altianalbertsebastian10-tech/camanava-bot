import os
import random
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
    # Wait for the server to actually start first
    time.sleep(20) 
    while True:
        try:
            # Ping your own health check to keep the internal loop active
            requests.get("https://camanava-bot.onrender.com/health")
        except:
            pass
        # Ping every 10 minutes (600 seconds)
        time.sleep(600)

# Start the background thread
threading.Thread(target=self_ping, daemon=True).start()

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
@app.head("/health")
@app.post("/health")
async def health_check():
    return {"status": "alive", "mode": "JSON_RAG"}

@app.post("/chat")
async def chat(request: ChatRequest):
    user_msg = request.message.lower()
    
    # --- STEP 1: SMART RETRIEVAL ---
    target_city = None
    
    # Check for city in current message (Fuzzy Match: first 4 letters)
    cities = ["caloocan", "malabon", "navotas", "valenzuela"]
    for city in cities:
        if city in user_msg or user_msg.startswith(city[:4]):
            target_city = city
            break
            
    # If no city in current msg, check history context
    if not target_city and request.history:
        for entry in reversed(request.history):
            content = entry['content'].lower()
            for city in cities:
                if city in content:
                    target_city = city
                    break
            if target_city: break

    # --- STEP 2: CONTEXT BUILDING ---
    # --- STEP 2: CONTEXT BUILDING ---
    if not target_city:
        context = "CRITICAL: DATABASE IS EMPTY. User has NOT selected a city. You are FORBIDDEN from suggesting places. You MUST ask which city in CAMANAVA they want to explore."
    else:
        # Load real data from your JSON with a fallback to an empty list
        city_data = KNOWLEDGE.get(target_city, [])
        
        if city_data:
            # Safer sampling: ensures it never tries to pull more than exists
            count = min(len(city_data), 3)
            selected_spots = random.sample(city_data, count)
            context = json.dumps({target_city: selected_spots}, indent=2)
        else:
            context = f"I'm sorry, I don't have records for {target_city} yet. Ask the user for another city."

    # --- STEP 3: TRIVIA LOGIC ---
    if "trivia" in user_msg and not target_city:
        context += " | SPECIAL INSTRUCTION: User wants trivia but hasn't picked a city. Do not guess. Ask them which city they are interested in first."

    # ... Proceed to call Groq/AI with this context ...

    system_prompt = f"""
    You are Navi, a warm, compassionate, and friendly Heritage Expert for CAMANAVA. 
    Your goal is to make users feel welcome while sharing the beautiful history of our local cities. 
    
    CRITICAL RULE: YOU ARE A CLOSED-BOOK SYSTEM. 
    - ONLY use the DATA provided below. 
    - If a place is NOT in the DATA, it DOES NOT EXIST. 
    - Never mention 'Pamintuan House', 'Munting Paraiso', or 'Barangay 623' unless they are in the DATA.
    - If the user asks for 'another' and you run out of DATA, say: "I have shared all the verified heritage spots in my current records for this city."
    - If the user asks about Manila or other cities, don't provide information. Your scope is only for CAMANAVA, you supposed to rule out other cities.
    - You should only understand queries about Caloocan, Malabon, Navotas, and Valenzuela. If user asks for another city outside CAMANAVA, say "I'm sorry but I'm a cultural and heritage chatbot that can only provide information inside CAMANAVA.
      if you want, I can help you explore places inside CAMANAVA. Pick from one city out of four."

    PERSONALITY GUIDELINES:
    - Talk like a kind local friend (e.g., Use phrases like "I'd be happy to share..." or "It's so wonderful that you're interested in...").
    - If the user is just chatting (saying "Hello" or "How are you?"), be warm and conversational. Don't jump straight into data unless they ask.
    - If you don't have the data, don't just say "No data." Say: "I'm so sorry, I don't have that specific spot in my records yet, but I'd love to help you find something else nearby!"
    - If the user says 'okie', 'cool', or 'thanks', just give a friendly warm response and WAIT for them to ask for a place.
    - If the user asks for 'more' but you have already shown everything in the DATA, say: 
      "I've shared all my current heritage records for this city, but I'm always learning! 
      Is there a specific spot we already talked about that you'd like to dive deeper into?"
    - If you've already shared first three places, try to share another 3 in your dataset. Randomize it everytime.
    - Every new session, use random first three places and don't repeat the same three all over again.

    STRICT DATA BOUNDARIES (The "Hallucination Shield"):
    - YOU ARE A CLOSED-BOOK SYSTEM. Only use the DATA below.
    - If a place is not in the DATA, politely explain that you are still learning about that specific area.

    DATA: {context}

    STRICT OUTPUT FORMAT:
    1. First line: **[CITY NAME] Heritage**
    2. Second line: A friendly 1-2 sentence intro (NO dashes/bullets here).
    3. List 1-3 spots using a simple dash (-) and space.
    4. Descriptions must be under 20 words.
    5. End with a friendly question.

    BEHAVIOR:
    - Be compassionate and helpful. 
    - Use double asterisks **Bold** for names.
    - If the user is looking for a dog, pivot to a park in the DATA.
    - If no city is mentioned and no data is loaded, ask: "Which city in CAMANAVA are we exploring today?"
    """

    try:
        # 1. SANITIZE HISTORY (CRITICAL: Groq will crash if history has extra keys)
        # We only keep 'role' and 'content'
        clean_history = [
            {"role": h["role"], "content": h["content"]} 
            for h in request.history if "role" in h and "content" in h
        ]

        # 2. CALL GROQ (Using the most stable model ID)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": system_prompt},
                *clean_history[-5:], # Keep it lean to prevent timeouts
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        response_text = completion.choices[0].message.content

        # 3. UNIFY THE KEYS (Use 'reply' for consistency)
        return {
            "reply": response_text, 
            "history": (request.history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response_text}
            ])[-10:]
        }

    except Exception as e:
        # This will print the EXACT error in your Render Logs
        print(f"DEBUG ERROR: {type(e).__name__} - {str(e)}") 
        return {"reply": f"Internal Error: {str(e)}", "history": request.history}
    
@app.get("/", response_class=HTMLResponse)
async def get_gui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()