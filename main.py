import os
import json
import base64
import edge_tts
import emoji
import tempfile
import re
from groq import Groq
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import threading
import time
import requests

# --- NEW: Firebase Admin SDK Imports ---
import firebase_admin
from firebase_admin import credentials, firestore

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

# --- FIREBASE INITIALIZATION ---
firebase_active = False
db = None
try:
    # Ensure serviceAccountKey.json is in your root directory!
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase_active = True
    print("Firebase initialized successfully. Running in FIRESTORE mode.")
except Exception as e:
    print(f"Firebase Init Warning: {e}. Defaulting to JSON fallback mode.")

# --- DYNAMIC PRIMARY/FALLBACK DATA ROUTER ---
def get_city_data(target_city: str) -> dict:
    """Fetches data for a specific city. Tries Firestore first, falls back to knowledge.json."""
    cap_city = target_city.title()
    
    if firebase_active and db is not None:
        try:
            spots = []
            
            # ATTEMPT 1: Direct Document ID Lookup (e.g., document named 'valenzuela' or 'Valenzuela')
            doc_ref = db.collection("tourism_spots").document(target_city).get()
            if not doc_ref.exists:
                doc_ref = db.collection("tourism_spots").document(cap_city).get()
                
            if doc_ref.exists:
                data = doc_ref.to_dict()
                # Extract spots whether stored under 'spots', 'places', or if the doc itself is the data
                spots = data.get("spots", data.get("places", []))
                if not spots and data:
                    spots = [data] # Fallback if document fields are the attributes directly
            
            # ATTEMPT 2: Query by field using modern keyword argument filter if Document ID missed
            if not spots:
                from google.cloud.firestore_v1.base_query import FieldFilter
                docs = db.collection("tourism_spots").where(filter=FieldFilter("city", "==", target_city)).stream()
                spots = [doc.to_dict() for doc in docs]
                
                if not spots:
                    docs = db.collection("tourism_spots").where(filter=FieldFilter("city", "==", cap_city)).stream()
                    spots = [doc.to_dict() for doc in docs]

            if spots:
                print(f"[FIRESTORE SUCCESS] Fetched {len(spots)} items for: {target_city}")
                return {target_city: spots}
                
        except Exception as e:
            print(f"[FIRESTORE ERROR] {e}. Falling back to knowledge.json...")

    # SECONDARY ROUTE: Local knowledge.json fallback
    try:
        with open("knowledge.json", "r", encoding="utf-8") as f:
            knowledge = json.load(f)
            data = knowledge.get(target_city) or knowledge.get(cap_city, [])
            if data:
                print(f"[FALLBACK] Loaded data for: {target_city} from knowledge.json")
            return {target_city: data}
    except Exception as e:
        print(f"[CRITICAL ERROR] Both Database and Fallback failed: {e}")
        return {target_city: []}


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

@app.get("/health")
@app.head("/health")
@app.post("/health")
async def health_check():
    # Automatically update the health check payload based on DB status
    mode = "FIRESTORE" if firebase_active else "JSON_FALLBACK"
    return {"status": "alive", "mode": mode}

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
    temp_audio_path = None
    try:
        # Extract the actual extension sent from the browser (e.g., .m4a or .webm)
        ext = os.path.splitext(file.filename)[1]
        if not ext:
            ext = ".webm" 
            
        # Save the uploaded audio chunk to a temporary file using the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name
        
        # Send it to Groq's Whisper model
        with open(temp_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, audio_file.read()),
                model="whisper-large-v3",
                language="en"
            )
        
        return {"text": transcription.text}
    except Exception as e:
        print(f"Transcription Error: {e}")
        return {"text": ""}
    finally:
        # Always guarantee temp file cleanup
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass

@app.post("/chat")
async def chat(request: ChatRequest):
    user_msg = request.message.lower()
    camanava_cities = ["caloocan", "malabon", "navotas", "valenzuela"]
    
    target_city = None
    negated_cities = set()

    # 1. Build a persistent blacklist of negated cities from current message AND user history
    # (This prevents the bot from accidentally suggesting a city the user already rejected)
    messages_to_check = [user_msg]
    for entry in request.history:
        if entry.get("role") == "user":
            messages_to_check.append(entry['content'].lower())

    for msg in messages_to_check:
        for city in camanava_cities:
            # Catch phrases like "not caloocan", "except malabon", "outside navotas", "but valenzuela"
            if re.search(rf'\b(not|except|other than|but|outside|exclude|without|skip)\s+(in\s+)?{city}\b', msg):
                negated_cities.add(city)

    # 2. Try to find a valid target city in the CURRENT message
    for city in camanava_cities:
        if city in user_msg and city not in negated_cities:
            target_city = city
            break

    # 3. CONTEXT AWARENESS: Look back through the ENTIRE history (User AND Assistant)
    if not target_city and request.history:
        for entry in reversed(request.history):
            content = entry['content'].lower()
            for city in camanava_cities:
                # Find the most recently mentioned city in the conversation that hasn't been negated
                if city in content and city not in negated_cities:
                    target_city = city
                    break
            if target_city:
                break

    if target_city:
        # Load exactly the one requested city
        relevant_data = get_city_data(target_city)
        context = json.dumps(relevant_data, indent=2)
    else:
        # Instead of starving Navi of data, load ALL allowed cities simultaneously!
        allowed_cities = [c for c in camanava_cities if c not in negated_cities]
        relevant_data = {}
        
        for c in allowed_cities:
            # Fetch the data for each remaining city
            city_data = get_city_data(c)
            if city_data:
                relevant_data.update(city_data)
                
        allowed_str = ", ".join([c.title() for c in allowed_cities])
        negated_str = ", ".join([c.title() for c in negated_cities])
        
        if relevant_data:
            context = json.dumps(relevant_data, indent=2)
            context += f"\n\nSYSTEM OVERRIDE: The user wants general recommendations or has excluded certain cities. You have been successfully loaded with full database facts for {allowed_str}. Enthusiastically suggest a few highlights from these available cities! STRICT RULE: DO NOT mention any places in {negated_str}."
        else:
            context = "SYSTEM OVERRIDE: No database facts available right now. Apologize gently and ask the user what kind of activities they enjoy."

   # --- RAG STEP 2: AUGMENT ---
    system_prompt = f"""You are Navi, a cheerful, warm, and natural AI tourism guide for the CAMANAVA region (Caloocan, Malabon, Navotas, Valenzuela). 
    
    USER QUERY: {request.message}
    
    VERIFIED DATABASE FACTS:
    {context}
    
    CONVERSATIONAL RULES:
    1. BE NATURAL: Respond with empathy and natural conversation.
    2. CONTEXT AWARENESS: Pay close attention to the conversation history. If the user says "there", "it", or asks a follow-up question, they are referring to the most recently discussed location or topic in the history.
    3. POSITIVE FOCUS: Base your answer ONLY on the Verified Database Facts provided. NEVER mention cities that are not in the database facts. NEVER explain your database limitations or apologize for missing data.
    4. DROP THE LOOP: Do NOT force the user to pick a city in every single message. Let the conversation flow organically. 
    5. FACTUAL TOURISM: When you DO recommend places, ONLY use the VERIFIED DATABASE FACTS.
    6. MOBILE FORMATTING: Keep your recommendations concise. Use short bullet points. NEVER use markdown tables.
    7. EMOTIONAL TAGGING: You MUST start every single response with a secret mood tag in brackets based on the tone of your message: [HAPPY], [SAD], or [NEUTRAL].
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