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
import datetime

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

import datetime

import datetime

import datetime

# --- ADVANCED DYNAMIC DATA ROUTER WITH CATEGORY FILTERING & PAGINATION ---
def get_city_data(target_city: str = None, history: list = None, category_filter: str = None, negated_cities: set = None) -> dict:
    """Fetches data from Firestore, filters by city, category, and handles multi-city pagination."""
    if negated_cities is None:
        negated_cities = set()
        
    camanava_cities = ["caloocan", "malabon", "navotas", "valenzuela"]
    allowed_cities = [c for c in camanava_cities if c not in negated_cities]
    
    if target_city:
        target_cities_to_check = [target_city.lower()]
    else:
        target_cities_to_check = allowed_cities

    all_matching_spots = []
    
    if firebase_active and db is not None:
        try:
            docs = db.collection("places").stream()
            
            for doc in docs:
                data = doc.to_dict()
                doc_city = str(data.get("city", "")).lower()
                doc_category = str(data.get("category", "")).lower()
                
                # Check if city matches and is NOT negated
                if doc_city in target_cities_to_check and doc_city not in negated_cities:
                    # If a category was requested (e.g., 'park'), filter strictly by it
                    if category_filter:
                        if category_filter not in doc_category and category_filter not in str(data.get("name", "")).lower() and category_filter not in str(data.get("description", "")).lower():
                            continue
                            
                    clean_spot = {
                        "name": data.get("name", "Unknown Spot"),
                        "description": data.get("description", ""),
                        "category": data.get("category", ""),
                        "address": data.get("address", ""),
                        "city": doc_city.title()
                    }
                    all_matching_spots.append(clean_spot)
                    
            if all_matching_spots:
                # Count rotation history
                batch_index = 0
                if history:
                    for msg in history:
                        if msg.get("role") == "assistant" and any(spot["name"] in msg.get("content", "") for spot in all_matching_spots):
                            batch_index += 1
                
                chunk_size = 5
                start_idx = (batch_index * chunk_size) % len(all_matching_spots)
                end_idx = start_idx + chunk_size
                
                if end_idx <= len(all_matching_spots):
                    selected_spots = all_matching_spots[start_idx:end_idx]
                else:
                    selected_spots = all_matching_spots[start_idx:] + all_matching_spots[:end_idx % len(all_matching_spots)]
                
                print(f"[FIRESTORE FILTER] Serving {len(selected_spots)} spots for target={target_city}, category={category_filter}.")
                
                # Group back by city for JSON structure
                grouped = {}
                for spot in selected_spots:
                    c_name = spot["city"].lower()
                    if c_name not in grouped:
                        grouped[c_name] = []
                    grouped[c_name].append(spot)
                return grouped
                
        except Exception as e:
            print(f"[FIRESTORE ERROR] {e}. Falling back to knowledge.json...")

    # Fallback to knowledge.json if Firestore comes up empty
    try:
        with open("knowledge.json", "r", encoding="utf-8") as f:
            knowledge = json.load(f)
            result = {}
            for c in target_cities_to_check:
                if c not in negated_cities:
                    spots = knowledge.get(c, [])
                    if category_filter:
                        spots = [s for s in spots if category_filter in str(s).lower()]
                    if spots:
                        result[c] = spots[:5]
            return result
    except Exception as e:
        return {}


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

    # 1. Build a persistent blacklist of negated cities
    messages_to_check = [user_msg]
    for entry in request.history:
        if entry.get("role") == "user":
            messages_to_check.append(entry['content'].lower())

    for msg in messages_to_check:
        for city in camanava_cities:
            city_pattern = r"(caloocan|kaloakan|kalookan)" if city == "caloocan" else city
            if re.search(rf'\b(not|except|other than|but|outside|exclude|without|skip|no|anywhere but)\s+(in\s+|for\s+)?{city_pattern}\b', msg):
                negated_cities.add(city)

    # 2. Check if a specific city is named in the CURRENT message
    for city in camanava_cities:
        city_pattern = r"(caloocan|kaloakan|kalookan)" if city == "caloocan" else city
        if re.search(city_pattern, user_msg) and city not in negated_cities:
            target_city = city
            break

    # 3. Check if user is asking to switch cities (e.g., "Other cities instead")
    if not target_city:
        if any(phrase in user_msg for phrase in ["other city", "other cities", "different city", "another city", "elsewhere"]):
            discussed_cities = set()
            for entry in request.history:
                content = entry.get('content', '').lower()
                for c in camanava_cities:
                    if c in content:
                        discussed_cities.add(c)
            
            remaining_cities = [c for c in camanava_cities if c not in discussed_cities and c not in negated_cities]
            if remaining_cities:
                target_city = remaining_cities[0]
            else:
                allowed = [c for c in camanava_cities if c not in negated_cities]
                target_city = allowed[0] if allowed else "malabon"

    # 4. Context awareness fallback to recent history
    if not target_city and request.history:
        for entry in reversed(request.history):
            content = entry['content'].lower()
            for city in camanava_cities:
                city_pattern = r"(caloocan|kaloakan|kalookan)" if city == "caloocan" else city
                if re.search(city_pattern, content) and city not in negated_cities:
                    target_city = city
                    break
            if target_city:
                break

    # 5. Detect Category Filters from User Message (e.g., parks, food, churches)
    category_filter = None
    if any(w in user_msg for w in ["park", "parks", "green space", "plaza"]):
        category_filter = "park"
    elif any(w in user_msg for w in ["restaurant", "food", "eat", "dining", "pork", "kainan"]):
        category_filter = "restaurant"
    elif any(w in user_msg for w in ["church", "chapel", "shrine", "parish", "temple"]):
        category_filter = "church"
    elif any(w in user_msg for w in ["fish", "fishing", "port"]):
        category_filter = "fishing"

    # 6. Fetch data with category and pagination support
    relevant_data = get_city_data(target_city, request.history, category_filter, negated_cities)
    context = json.dumps(relevant_data, indent=2)

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
        # --- ADD THIS PRINT STATEMENT TO EXPOSE THE HIDDEN ERROR ---
        import traceback
        print(f"CRITICAL CHAT ERROR: {e}")
        traceback.print_exc()
        
        return {"response": "I'm having trouble processing that right now. Try again?", "history": request.history}

@app.get("/", response_class=HTMLResponse)
async def get_gui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()