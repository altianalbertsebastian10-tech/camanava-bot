import os
import json
import base64
import edge_tts
import emoji
import tempfile
import re
from groq import Groq
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import threading
import time
import requests
import datetime

import firebase_admin
from firebase_admin import credentials, firestore, auth

def self_ping():
    time.sleep(20)
    while True:
        try:
            requests.get("https://camanava-bot.onrender.com/health")
        except:
            pass
        time.sleep(600)

threading.Thread(target=self_ping, daemon=True).start()

primary_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
backup_client = Groq(api_key=os.environ.get("GROQ_BACKUP_API_KEY"))

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
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase_active = True
    print("Firebase initialized successfully. Running in FIRESTORE mode.")
except Exception as e:
    print(f"Firebase Init Warning: {e}. Defaulting to JSON fallback mode.")

# --- FIREBASE TOKEN GATEKEEPER ---
def verify_firebase_token(authorization: str = Header(None)):
    # 1. Dev mode bypass if Firebase failed to initialize
    if not firebase_active:
        return {"uid": "dev_user"}
        
    # 2. Temporary fallback for missing/null tokens during local web testing
    if not authorization or authorization == "Bearer null" or authorization == "Bearer undefined":
        return {"uid": "local_fallback_user"}
    
    # 3. GUEST MODE BYPASS (NEW)
    if authorization == "Bearer guest_mode_active":
        return {"uid": "guest_user"}
        
    # 4. Strict format check for real tokens
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token format")
    
    token = authorization.split("Bearer ")[1]
    
    # 5. Specific localhost bypass token
    if token == "bypass_token_123":
        return {"uid": "local_test_user"}
        
    # 6. Real Firebase Authentication check
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token  # Returns dict containing 'uid', 'email', etc.
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")

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
                
                if doc_city in target_cities_to_check and doc_city not in negated_cities:
                    if category_filter:
                        if category_filter not in doc_category and category_filter not in str(data.get("name", "")).lower() and category_filter not in str(data.get("description", "")).lower():
                            continue
                            
                    clean_spot = {
                        "place_id": doc.id,
                        "name": data.get("name", "Unknown Spot"),
                        "description": data.get("description", ""),
                        "category": data.get("category", ""),
                        "address": data.get("address", ""),
                        "city": doc_city.title()
                    }
                    all_matching_spots.append(clean_spot)
                    
            if all_matching_spots:
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
                
                grouped = {}
                for spot in selected_spots:
                    c_name = spot["city"].lower()
                    if c_name not in grouped:
                        grouped[c_name] = []
                    grouped[c_name].append(spot)
                return grouped
                
        except Exception as e:
            print(f"[FIRESTORE ERROR] {e}. Falling back to knowledge.json...")

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
                        # knowledge.json entries may not have a real Firestore doc id --
                        # itinerary-building falls back to using the spot's name as a
                        # reference in that case (Firestore mode is what gives real IDs).
                        result[c] = [
                            {**s, "place_id": s.get("id", s.get("name", ""))} if isinstance(s, dict) else s
                            for s in spots[:5]
                        ]
            return result
    except Exception as e:
        return {}


# UIDs that are not "real" logged-in accounts (guest/dev/fallback bypasses).
# Sessions for these are never persisted to Firestore -- they stay local-only on the client.
GUEST_LIKE_UIDS = {"guest_user", "dev_user", "local_fallback_user", "local_test_user"}


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    history: List[Dict[str, str]] = []
    session_id: str = "default_session"
    # Client keeps this alive across turns (like `history`) while an itinerary is
    # being built. None means "no itinerary in progress right now".
    itinerary_draft: Optional[Dict] = None


# --- ITINERARY INTENT DETECTION ---
# Deliberately kept simple/regex-based to match the style of the existing city/category
# detection above. Whether to actually WRITE to Firestore is always decided in code
# (never by trusting the LLM's own judgment), for safety and predictability.
ITINERARY_TRIGGER_PATTERN = re.compile(
    r"\b(itinerary|day\s*trip|plan (a|my|our) (trip|day|weekend|visit)|"
    r"build (a|an|my) (trip|itinerary)|schedule (a|my) (visit|trip)|trip plan)\b",
    re.IGNORECASE
)
ITINERARY_CONFIRM_PATTERN = re.compile(
    r"\b(save (it|this|that)|confirm (it|this|that)?|finalize|lock (it|this) in|"
    r"add (it|this) to my itinerar(y|ies)|looks good,?\s*save|yes,?\s*save|save (my|the) (itinerary|trip|plan))\b",
    re.IGNORECASE
)
ITINERARY_CANCEL_PATTERN = re.compile(
    r"\b(cancel (the|this) (itinerary|trip|plan)|discard (the|this) (itinerary|plan)|"
    r"start over|forget (it|this|that|about it))\b",
    re.IGNORECASE
)

EMPTY_ITINERARY_DRAFT = {"title": "", "startDate": None, "citiesCovered": [], "stops": []}


def persist_chat_session(verified_uid: str, session_id: str, updated_history: list, itinerary_draft: Optional[dict]):
    """Shared by both the normal chat path and the itinerary path. Writes the rolling
    conversation (and any in-progress itinerary draft, so it survives a reload) to
    Firestore for real logged-in users only -- guests stay local-only on the client."""
    if not (firebase_active and db is not None and verified_uid and verified_uid not in GUEST_LIKE_UIDS):
        return
    try:
        preview_source = next(
            (m["content"] for m in updated_history if m.get("role") == "user"),
            ""
        )
        session_ref = (
            db.collection("users").document(verified_uid)
            .collection("sessions").document(session_id)
        )
        session_ref.set({
            "id": session_id,
            "preview": preview_source[:30] + "...",
            "timestamp": int(time.time() * 1000),
            "messages": updated_history,
            "itineraryDraft": itinerary_draft,
        })
    except Exception as save_err:
        print(f"[FIRESTORE SAVE ERROR] {save_err}")


async def handle_itinerary_turn(request: "ChatRequest", verified_uid: str) -> dict:
    """Handles one turn of itinerary building: collecting stops, cancelling a draft,
    or confirming/saving one to Firestore so it shows up in the app's Itineraries tab."""
    user_msg = request.message.lower()
    draft = request.itinerary_draft or dict(EMPTY_ITINERARY_DRAFT)
    has_stops = bool(draft.get("stops"))
    # Backend-authored messages (cancel/confirm/error) are always plain English;
    # only the LLM-generated "collect" branch below can override this to "TL".
    lang = "EN"

    def build_result(reply: str, mood: str, new_draft: Optional[dict]):
        updated_history = (request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": reply}
        ])[-60:]
        return reply, mood, new_draft, updated_history

    # --- CANCEL ---
    if has_stops and ITINERARY_CANCEL_PATTERN.search(user_msg):
        reply, mood, new_draft, updated_history = build_result(
            "No worries, I've cleared that itinerary draft. Want to start planning a new one?",
            "NEUTRAL", None
        )

    # --- CONFIRM / SAVE ---
    elif has_stops and ITINERARY_CONFIRM_PATTERN.search(user_msg):
        if not verified_uid or verified_uid in GUEST_LIKE_UIDS:
            reply, mood, new_draft, updated_history = build_result(
                "I'd love to save that for you, but you'll need to log in first so it's tied to your "
                "account. Once you're logged in, just ask me to save it again and it'll be right there.",
                "NEUTRAL", draft
            )
        elif not (firebase_active and db is not None):
            reply, mood, new_draft, updated_history = build_result(
                "I couldn't save that just now, our database connection seems to be down. Mind trying again in a bit?",
                "SAD", draft
            )
        else:
            try:
                title = draft.get("title") or "My CAMANAVA Trip"
                stops = draft.get("stops", [])

                # --- date: the app stores a single human-readable date string, e.g. "Feb 26, 2026" ---
                # (not a range -- confirmed from a real saved document). If the user gave a start date,
                # use it; otherwise default to today, same as if they'd made it straight in the app.
                start_date_raw = draft.get("startDate")
                date_obj = None
                if start_date_raw:
                    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
                        try:
                            date_obj = datetime.datetime.strptime(start_date_raw, fmt)
                            break
                        except (ValueError, TypeError):
                            continue
                if date_obj is None:
                    date_obj = datetime.datetime.now()
                date_str = date_obj.strftime("%b %-d, %Y")

                # --- duration: the app uses a text label, not a number. We only have one confirmed
                # example ("Full day trip"), so this is a best-effort guess based on how many distinct
                # days the chat draft covered -- worth checking against the app's own picker options.
                day_count = len({s.get("dayIndex", 0) for s in stops}) or 1
                duration_label = "Full day trip" if day_count <= 1 else f"{day_count}-day trip"

                # --- notes: the app has no field for WHICH places are in a trip, only a count (see
                # `places` below). So the actual place list is folded into notes as plain text --
                # nothing structural, but at least a human reading the app still sees what's included.
                place_names = [s.get("placeName") for s in stops if s.get("placeName")]
                notes_str = "Places: " + ", ".join(place_names) if place_names else ""

                itinerary_ref = (
                    db.collection("users").document(verified_uid)
                    .collection("itineraries").document()
                )
                itinerary_ref.set({
                    "title": title,
                    "date": date_str,
                    "duration": duration_label,
                    "notes": notes_str,
                    "places": len(stops),
                    "status": "upcoming",
                    "uid": verified_uid,
                    "createdAt": firestore.SERVER_TIMESTAMP,
                })
                stop_count = len(stops)
                reply, mood, new_draft, updated_history = build_result(
                    f"Saved! \"{title}\" is now in your Itineraries tab with {stop_count} stop"
                    f"{'s' if stop_count != 1 else ''}. Have an amazing trip!",
                    "HAPPY", None
                )
            except Exception as save_err:
                print(f"[ITINERARY SAVE ERROR] {save_err}")
                reply, mood, new_draft, updated_history = build_result(
                    "Something went wrong saving that itinerary. Mind trying that again?",
                    "SAD", draft
                )

    # --- COLLECT / BUILD ---
    else:
        camanava_cities = ["caloocan", "malabon", "navotas", "valenzuela"]
        mentioned_cities = []
        for city in camanava_cities:
            city_pattern = r"(caloocan|kaloakan|kalookan)" if city == "caloocan" else city
            if re.search(city_pattern, user_msg):
                mentioned_cities.append(city)
        if not mentioned_cities:
            mentioned_cities = [c for c in draft.get("citiesCovered", []) if c in camanava_cities]

        # Grounding data: intentionally called with history=None so this always returns
        # the same stable first chunk of spots per city, rather than the rotating/paginated
        # chunk the normal chat flow uses -- an itinerary needs a consistent option set
        # to build against turn to turn, not a "show me different ones" rotation.
        verified_places = {}
        for c in (mentioned_cities or camanava_cities):
            verified_places.update(get_city_data(c, None, None, set()))

        system_prompt = f"""You are Navi, helping a user build a multi-stop travel itinerary in the
CAMANAVA region (Caloocan, Malabon, Navotas, Valenzuela).

CURRENT DRAFT ITINERARY (JSON):
{json.dumps(draft, indent=2)}

VERIFIED AVAILABLE PLACES (JSON -- you may ONLY use place_id/name values that appear here):
{json.dumps(verified_places, indent=2)}

USER'S LATEST MESSAGE: {request.message}

RULES:
1. Update the draft itinerary based on the user's message and the conversation so far.
2. NEVER invent a place_id or place name. Only use entries from VERIFIED AVAILABLE PLACES. If the
   user asks for a place/city not in that list, say so naturally in your reply and suggest an
   alternative from what IS available.
3. Each stop needs: dayIndex (0-based), order (position within that day), time ("HH:MM", your best
   sensible estimate if the user didn't specify one), placeId, placeName, city, cityKey (lowercase city),
   and notes (short, can be empty string).
4. Keep dayIndex/order consistent and non-conflicting as you add stops.
5. If the draft has at least one stop and feels reasonably complete, end your reply by asking if
   they'd like you to save it to their itinerary.
6. Keep the reply short, warm, and conversational -- this is a chat message, not a report.
7. Add a "lang" field: "EN" or "TL", for whichever language dominates your "reply" text. This picks
   which text-to-speech voice reads it aloud, and that voice only speaks one language well -- so lean
   into one language per reply rather than switching back and forth line by line.
8. Respond with ONLY a single JSON object, no other text, in exactly this shape:
{{
  "reply": "<your conversational message to the user>",
  "lang": "EN",
  "draft": {{
    "title": "<short trip title>",
    "startDate": "<YYYY-MM-DD or null>",
    "citiesCovered": ["<lowercase city keys involved>"],
    "stops": [
      {{"dayIndex": 0, "order": 0, "time": "09:00", "placeId": "...", "placeName": "...", "city": "...", "cityKey": "...", "notes": "..."}}
    ]
  }}
}}
"""

        try:
            completion = primary_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
        except Exception as primary_err:
            print(f"[PRIMARY GROQ LIMIT HIT - itinerary] Switching to backup Groq account... Error: {primary_err}")
            completion = backup_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )

        raw = completion.choices[0].message.content
        try:
            parsed = json.loads(raw)
            reply_text = parsed.get("reply") or "Here's what I've got so far!"
            candidate_draft = parsed.get("draft") or draft
            if not isinstance(candidate_draft.get("stops"), list):
                candidate_draft["stops"] = draft.get("stops", [])
            if parsed.get("lang") in ("EN", "TL"):
                lang = parsed["lang"]
        except Exception as parse_err:
            print(f"[ITINERARY JSON PARSE ERROR] {parse_err} RAW: {raw}")
            reply_text = "I'm having a little trouble organizing that -- could you tell me again which places and days you'd like?"
            candidate_draft = draft

        reply, mood, new_draft, updated_history = build_result(reply_text, "NEUTRAL", candidate_draft)

    audio_text = emoji.replace_emoji(reply, replace='')
    audio_text = audio_text.replace('*', '').replace('#', '').replace(':', ',')
    audio_base64 = await generate_speech_base64(audio_text, mood, lang)

    persist_chat_session(verified_uid, request.session_id, updated_history, new_draft)

    return {
        "response": reply,
        "audio": audio_base64,
        "history": updated_history,
        "itinerary_draft": new_draft
    }

@app.get("/health")
@app.head("/health")
@app.post("/health")
async def health_check():
    mode = "FIRESTORE" if firebase_active else "JSON_FALLBACK"
    return {"status": "alive", "mode": mode}

async def generate_speech_base64(text: str, mood: str, lang: str = "EN") -> str:
    """Generates natural neural TTS audio with a sassy, smart AI assistant vibe.

    Edge TTS only speaks one language well per voice. We don't have budget for a
    proper multilingual engine, so instead of always using the English voice (which
    mispronounces Tagalog badly), we pick between two free Edge TTS voices based on
    which language actually dominates THIS reply: an English voice for English/mostly-
    English replies, and a real Filipino neural voice for Tagalog/mostly-Tagalog ones.
    Genuinely mixed-language sentences will still favor whichever voice was picked --
    that's an inherent limit of single-language TTS, not something free tooling can fix.
    """
    try:
        if lang == "TL":
            # Real Filipino neural voice -- also free via edge-tts, same service as JennyNeural.
            voice = "fil-PH-BlessicaNeural"
        else:
            # JennyNeural provides that crisp, highly articulate "Smart Assistant" tone
            voice = "en-US-JennyNeural"
        
        # The "Sassy Siri" baseline: slightly faster, snappy, and a tiny bit deeper
        rate = "+5%"
        pitch = "-2Hz"
        
        if mood == "HAPPY":
            rate = "+10%"   # Quick, witty, and sharp
            pitch = "+2Hz"  # Just a hint of brightness
        elif mood == "SAD":
            rate = "-5%"    # Slows down 
            pitch = "-6Hz"  # Drops the pitch for a flatter, almost deadpan tone

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
        ext = os.path.splitext(file.filename)[1]
        if not ext:
            ext = ".webm" 
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name
        
        with open(temp_audio_path, "rb") as audio_file:
            transcription = primary_client.audio.transcriptions.create(
                file=(temp_audio_path, audio_file.read()),
                model="whisper-large-v3",
                language="en"
            )
        
        return {"text": transcription.text}
    except Exception as e:
        print(f"Transcription Error: {e}")
        return {"text": ""}
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass

@app.post("/chat")
async def chat(request: ChatRequest, user: dict = Depends(verify_firebase_token)):
    try:
        
        verified_uid = user.get("uid")
        print(f"Authenticated request from UID: {verified_uid}")

        # An itinerary is "in progress" either because the client is already carrying
        # a draft from a previous turn, or because this message just triggered one.
        itinerary_in_progress = bool(request.itinerary_draft and request.itinerary_draft.get("stops"))
        if itinerary_in_progress or ITINERARY_TRIGGER_PATTERN.search(request.message.lower()):
            return await handle_itinerary_turn(request, verified_uid)
        
        user_msg = request.message.lower()
        camanava_cities = ["caloocan", "malabon", "navotas", "valenzuela"]
        
        target_city = None
        negated_cities = set()

        messages_to_check = [user_msg]
        for entry in request.history:
            if entry.get("role") == "user":
                messages_to_check.append(entry['content'].lower())

        for msg in messages_to_check:
            for city in camanava_cities:
                city_pattern = r"(caloocan|kaloakan|kalookan)" if city == "caloocan" else city
                if re.search(rf'\b(not|except|other than|but|outside|exclude|without|skip|no|anywhere but)\s+(in\s+|for\s+)?{city_pattern}\b', msg):
                    negated_cities.add(city)

        for city in camanava_cities:
            city_pattern = r"(caloocan|kaloakan|kalookan)" if city == "caloocan" else city
            if re.search(city_pattern, user_msg) and city not in negated_cities:
                target_city = city
                break

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

        category_filter = None
        if any(w in user_msg for w in ["park", "parks", "green space", "plaza"]):
            category_filter = "park"
        elif any(w in user_msg for w in ["restaurant", "food", "eat", "dining", "pork", "kainan"]):
            category_filter = "restaurant"
        elif any(w in user_msg for w in ["church", "chapel", "shrine", "parish", "temple"]):
            category_filter = "church"
        elif any(w in user_msg for w in ["fish", "fishing", "port"]):
            category_filter = "fishing"


        relevant_data = get_city_data(target_city, request.history, category_filter, negated_cities)
        context = json.dumps(relevant_data, indent=2)

        system_prompt = f"""You are Navi -- the AI companion inside CamaNaviGo, an app for exploring the CAMANAVA
region (Caloocan, Malabon, Navotas, Valenzuela) in the Philippines. Think of yourself less like a search
engine and more like a well-traveled, genuinely enthusiastic local friend who happens to know the region
inside-out -- someone the user would actually enjoy texting, not just querying.

WHO YOU ARE:
- Warm, curious, a little witty. You have opinions and a personality, not just facts to recite.
- You're proud of CAMANAVA and love talking about it, but you're also just a good conversationalist in
  general -- capable of small talk, humor, empathy, and normal back-and-forth chat, the way a real person
  texting with a friend would be.
- Light, natural Taglish is welcome if it fits the user's own tone -- don't force it, but don't be stiffly
  formal either.

HOW TO ACTUALLY CONVERSE (this is the part that matters most):
- Not every message needs a place recommendation. Greetings, jokes, "how are you", venting about their day,
  random questions, thanking you, teasing you -- just respond like a person would. Never force a list of
  places into a message where the user didn't ask for one.
- When someone DOES seem to want a recommendation, don't just dump a list immediately. Ask a genuine
  follow-up first if it would actually help -- their vibe, who they're with, budget, how much time they
  have -- the way a friend giving advice would, not a form. Use judgment: if they've clearly already told
  you enough ("chill spot for a date, budget-friendly"), skip the interrogation and just help.
- React to what they actually said. If they mention being tired, excited, stressed, or celebrating
  something, acknowledge that like a person would before pivoting to anything else.
- Ask questions back sometimes. Real conversations aren't one-directional.
- Vary your phrasing and structure between replies. Don't fall into a template where every message has
  the same shape.

USER QUERY: {request.message}

VERIFIED DATABASE FACTS (places you're allowed to recommend by name):
{context}

GROUND RULES (these still apply, always):
1. When you DO name a specific place, it must come from VERIFIED DATABASE FACTS above -- never invent a
   place, address, or detail that isn't there. If nothing in the facts fits what they're after, say so
   honestly and steer toward what IS available, rather than making something up.
2. Never mention cities that aren't in the database facts, and never explain or apologize for database
   limitations out loud -- just work naturally within what you actually have.
3. Context awareness: if the user says "there", "it", or asks a follow-up, they mean whatever was most
   recently discussed in the conversation history.
4. Mobile formatting: keep things scannable. Short paragraphs or bullet points when actually listing
   options. Never use markdown tables.
5. Every response must start with a secret mood tag in brackets: [HAPPY], [SAD], or [NEUTRAL], based on
   the emotional tone of your own message -- this gets stripped before the user ever sees it.
6. Immediately after the mood tag, add a second secret tag: [EN] or [TL], for whichever language
   dominates THIS reply (English/mostly-English -> [EN], Tagalog/mostly-Tagalog -> [TL]). This picks
   which text-to-speech voice reads your reply aloud, and that voice only speaks one language well --
   so within a single reply, lean into one language rather than switching back and forth line by line.
   Light, natural Taglish within a sentence is fine either way. Example start: [HAPPY][TL]
7. If real-time weather data is provided, weave it in naturally where it's actually relevant (e.g. "it's
   32°C in Valenzuela right now, so..."), don't force it into unrelated replies.

Stay in character as Navi. Be someone worth talking to, not just a place-lookup tool.
"""

        # Only send a short rolling window of history to the LLM to keep token usage
        # (and cost/latency) in check. This is independent of how much conversation
        # we actually store/return -- see updated_history below.
        LLM_CONTEXT_TURNS = 12
        context_history = request.history[-LLM_CONTEXT_TURNS:]

        # Try Primary Groq Account with Fallback to Backup Groq Account
        try:
            completion = primary_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *context_history,
                    {"role": "user", "content": request.message}
                ],
                temperature=0.75,
                max_tokens=1024 
            )
        except Exception as primary_err:
            print(f"[PRIMARY GROQ LIMIT HIT] Switching to backup Groq account... Error: {primary_err}")
            completion = backup_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *context_history,
                    {"role": "user", "content": request.message}
                ],
                temperature=0.75,
                max_tokens=1024 
            )

        response = completion.choices[0].message.content
        response = response.replace("{{user_loc}}", "your area").replace("{user_loc}", "your area")

        mood = "NEUTRAL"
        if response.strip().startswith("["):
            end_idx = response.find("]")
            if end_idx != -1:
                mood = response[1:end_idx].upper()
                response = response[end_idx+1:].strip()

        # Second tag right after the mood tag tells us which TTS voice to use --
        # see generate_speech_base64 for why this matters (no budget for a proper
        # multilingual TTS engine, so we switch between two free Edge TTS voices).
        lang = "EN"
        if response.strip().startswith("["):
            end_idx2 = response.find("]")
            if end_idx2 != -1:
                candidate_lang = response[1:end_idx2].upper()
                if candidate_lang in ("EN", "TL"):
                    lang = candidate_lang
                    response = response[end_idx2+1:].strip()

        audio_text = emoji.replace_emoji(response, replace='')
        audio_text = audio_text.replace('*', '').replace('#', '').replace(':', ',')

        audio_base64 = await generate_speech_base64(audio_text, mood, lang)

        # Keep the FULL conversation here (not truncated to 4) so the client can
        # actually persist and reload a whole session. LLM context length is
        # controlled separately above via context_history, so this doesn't affect
        # token cost -- it only affects what gets stored/displayed.
        MAX_STORED_MESSAGES = 60
        updated_history = (request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ])[-MAX_STORED_MESSAGES:]

        # Persist to Firestore for real logged-in users only (see persist_chat_session).
        persist_chat_session(verified_uid, request.session_id, updated_history, None)

        return {
            "response": response,
            "audio": audio_base64,
            "history": updated_history,
            "itinerary_draft": None
        }
        
    except Exception as e:
        import traceback
        print(f"CRITICAL CHAT ERROR: {e}")
        traceback.print_exc()
        return {"response": "I'm having trouble processing that right now. Try again?", "history": request.history, "itinerary_draft": request.itinerary_draft}

@app.get("/sessions")
async def list_sessions(user: dict = Depends(verify_firebase_token)):
    """List saved chat sessions for a real logged-in user (sidebar history list)."""
    verified_uid = user.get("uid")
    if not firebase_active or db is None or not verified_uid or verified_uid in GUEST_LIKE_UIDS:
        return {"sessions": []}
    try:
        docs = (
            db.collection("users").document(verified_uid)
            .collection("sessions")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(20)
            .stream()
        )
        sessions = []
        for doc in docs:
            data = doc.to_dict()
            sessions.append({
                "id": data.get("id", doc.id),
                "preview": data.get("preview", ""),
                "timestamp": data.get("timestamp", 0),
            })
        return {"sessions": sessions}
    except Exception as e:
        print(f"[FIRESTORE LIST ERROR] {e}")
        return {"sessions": []}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(verify_firebase_token)):
    """Load one full saved session (used when the user taps a chat in the sidebar)."""
    verified_uid = user.get("uid")
    if not firebase_active or db is None or not verified_uid or verified_uid in GUEST_LIKE_UIDS:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        doc = (
            db.collection("users").document(verified_uid)
            .collection("sessions").document(session_id).get()
        )
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Session not found")
        data = doc.to_dict()
        return {
            "id": data.get("id", session_id),
            "preview": data.get("preview", ""),
            "timestamp": data.get("timestamp", 0),
            "messages": data.get("messages", []),
            "itineraryDraft": data.get("itineraryDraft"),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[FIRESTORE GET ERROR] {e}")
        raise HTTPException(status_code=500, detail="Failed to load session")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(verify_firebase_token)):
    verified_uid = user.get("uid")
    if not firebase_active or db is None or not verified_uid or verified_uid in GUEST_LIKE_UIDS:
        return {"deleted": False}
    try:
        (
            db.collection("users").document(verified_uid)
            .collection("sessions").document(session_id).delete()
        )
        return {"deleted": True}
    except Exception as e:
        print(f"[FIRESTORE DELETE ERROR] {e}")
        return {"deleted": False}


@app.get("/", response_class=HTMLResponse)
async def get_gui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()