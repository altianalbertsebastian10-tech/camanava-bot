import os
import json
import base64
import edge_tts
import emoji
import tempfile
from groq import Groq
from fastapi import FastAPI, UploadFile, File
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

async def generate_speech_base64(text: str, mood: str) -> str:
    """Generates natural neural TTS audio with subtle, human-like pacing."""
    try:
        voice = "en-US-AriaNeural"
        
        # Subtle, natural adjustments that won't make her sound like a child
        rate = "+0%"
        pitch = "+0Hz"
        
        if mood == "HAPPY":
            rate = "+5%"   # Just a tiny hint of energy, not rushing
            pitch = "+2Hz"  # Barely noticeable warmth, completely natural
        elif mood == "SAD":
            rate = "-8%"   # Just a gentle slow down for empathy
            pitch = "-3Hz"  # Slightly softer tone

        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
                
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print(f"Edge-TTS Error: {e}")
        return ""

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded audio chunk to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name
        
        # Send it to Groq's Whisper model
        with open(temp_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, audio_file.read()),
                model="whisper-large-v3",
                language="en" # You can change to 'tl' for Tagalog if needed
            )
        
        # Clean up the temp file
        os.remove(temp_audio_path)
        
        return {"text": transcription.text}
    except Exception as e:
        print(f"Transcription Error: {e}")
        return {"text": ""}

@app.post("/chat")
async def chat(request: ChatRequest):
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
        context = "SYSTEM OVERRIDE: NO CITY SPECIFIED. You have ZERO data loaded. You are STRICTLY FORBIDDEN from listing any spots. If the user mentions an interest or vibe (like food, culture, or nature), enthusiastically validate their choice, and then specifically ask them which of the 4 cities (Caloocan, Malabon, Navotas, Valenzuela) they want to do that activity in!"

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
    6. EMOTIONAL TAGGING: You MUST start every single response with a secret mood tag in brackets based on the tone of your message. Choose exactly one: [HAPPY] (for cheerful, excited, or welcoming responses), 
    [SAD] (for apologies, missing data, or empathy), or [NEUTRAL] (for standard facts). Example: "[HAPPY] I would love to help you with that!"
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
            max_tokens=1024 
        )

        response = completion.choices[0].message.content
        response = response.replace("{{user_loc}}", "your area").replace("{user_loc}", "your area")

        # --- INTERCEPT EMOTION TAG ---
        mood = "NEUTRAL"
        if response.strip().startswith("["):
            end_idx = response.find("]")
            if end_idx != -1:
                # Extract the tag (e.g., "HAPPY")
                mood = response[1:end_idx].upper()
                # Remove the tag from the text so the user doesn't see it in the UI!
                response = response[end_idx+1:].strip()

        # --- CLEAN TEXT FOR AUDIO ---
        # 1. Strip all emojis
        audio_text = emoji.replace_emoji(response, replace='')
        
        # 2. Strip formatting asterisks and hashtags so she doesn't say "hashtag"
        audio_text = audio_text.replace('*', '').replace('#', '')
        
        # 3. Replace colons with a comma so she takes a natural pause instead of saying "colon"
        audio_text = audio_text.replace(':', ',')

        # --- GENERATE FREE NEURAL AUDIO ---
        # Pass the cleaned text to the voice generator, but keep the original 'response' for the UI!
        audio_base64 = await generate_speech_base64(audio_text, mood)

        updated_history = request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ]

        return {
            "response": response,
            "audio": audio_base64,
            "history": updated_history[-10:]
        }
    except Exception as e:
        return {"response": "I'm having trouble accessing my records. Try again?", "history": request.history}

@app.get("/", response_class=HTMLResponse)
async def get_gui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

    #update