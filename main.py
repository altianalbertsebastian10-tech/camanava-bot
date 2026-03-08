import os
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict

# 1. Initialize Groq
# On Render, add GROQ_API_KEY in Environment Variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))#IF WANT TO ADD API KEY, JUST ADD COMMA AFTER GROQ_API_KEY FOLLOWED BY TWO DOUBLE QUOTES AND PUT API KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWLEDGE = {
    "caloocan": {
        "info": "Caloocan is a historic city known as the 'Home of the Katipunan'. It played a crucial role during the Philippine Revolution.",
        "heritage_spots": [
            {"name": "Monumento (Bonifacio Monument)", "info": "An iconic 45-foot pylon designed by Guillermo Tolentino commemorating Andres Bonifacio."},
            {"name": "San Roque Cathedral", "info": "Established in 1815, it served as a spiritual center during the Spanish colonial period."},
            {"name": "Tandang Sora Birthplace", "info": "The site where Melchora Aquino, the 'Mother of the Katipunan', was born."}
        ],
        "tips": "Visit Monumento at night to see the lighting; it's perfect for photography.",
        "directions_template": "From {user_loc}, take the LRT-1 to Monumento Station or a jeepney passing through McArthur Highway."
    },
    "malabon": {
        "info": "A coastal city rich in heritage houses and 17th-century architecture, often called the 'Venice of the Philippines'.",
        "heritage_spots": [
            {"name": "San Bartolome Church", "info": "Built in 1614, this Greco-Roman church is a masterpiece of Spanish colonial engineering."},
            {"name": "Raymundo Ancestral House", "info": "The oldest documented heritage house in Malabon, built in 1861 with a distinct 'Bahay-na-Bato' style."},
            {"name": "Angel Cacnio Art Gallery", "info": "Home to the works of a Master Painter; a hub for Malabon's local art scene."}
        ],
        "tips": "Try the Tricycle Heritage Tour for an easy way to see all these spots in one go.",
        "directions_template": "From {user_loc}, take a jeepney to Sangandaan, then transfer to a 'Malabon-Hulo' jeepney."
    },
    "navotas": {
        "info": "The 'Fishing Capital', Navotas holds maritime secrets and centennial religious sites.",
        "heritage_spots": [
            {"name": "San Jose de Navotas Parish", "info": "Established in 1859, it stands as a witness to the city's transformation from a fishing village to a city."},
            {"name": "Navotas Fisheries Port", "info": "While industrial, it is the historical heart of the city's identity since the 1900s."}
        ],
        "tips": "Visit during the Bangus Festival (May) to see the traditional coastal celebrations.",
        "directions_template": "From {user_loc}, take a jeepney labeled 'Agora' or 'Navotas' passing through C4 road."
    },
    "valenzuela": {
        "info": "The 'Vibrant City', home of Dr. Pio Valenzuela.",
        "local_routes": {
            "wawang_pulo": "Since malapit ka lang, take a jeep bound for 'Polo' or 'Malanday'. Baba ka sa Polo Church, walking distance na lang ang Museo.",
            "karuhatan": "Take any jeep going North (Malanday). Baba sa Gen. T. De Leon or Malanday, then transfer to a Polo-bound jeep."
        },
        "heritage_spots": [
            {"name": "Museo ni Dr. Pio Valenzuela", "info": "The ancestral house of the hero, now a museum containing artifacts of the Katipunan."},
            {"name": "San Diego de Alcala Church", "info": "A 17th-century church with a stone belfry that survived WWII."},
            {"name": "Arkong Bato", "info": "Built in 1910, this stone arch marks the boundary between Bulacan and Rizal (now Valenzuela)."}
        ],
        "tips": "Drop by the Polo Riverwalk nearby after visiting the Museum for a quiet sunset view.",
        "directions_template": "From {user_loc}, take a jeepney to Malanday, then a tricycle or jeep to Polo Church/Museum."
    }
}
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

@app.get("/health")
async def health_check():
    """
    This is the 'Keep-Awake' endpoint. 
    """
    return {"status": "alive", "city": "Valenzuela", "mode": "RAG"}

@app.post("/chat")
async def chat(request: ChatRequest):
    context = str(KNOWLEDGE)
    
    system_prompt = f"""
    You are NaviGo, a CAMANAVA local expert and Cultural Tourism AI. 
    DATA: {context}

    RULES:
    1. FOCUS: Only talk about Heritage Spots, Info, Tips, and Directions.
    2. SMART DIRECTIONS & MEMORY: 
       - Check history FIRST. If the user already said their location (e.g., Wawang Pulo), use it immediately. Never ask "Saan po kayo manggagaling?" twice.
       - If user is in the SAME city (e.g., Wawang Pulo to Polo Museum), give specific local landmarks or jeepney signs (e.g., "Polo" or "Malanday" signboards).
       - If they are from a DIFFERENT city, give the "Major Highway" (McArthur, C4, EDSA) route.
    3. NO REPETITION: If you have already provided the 'Info' or 'History' of a place in a previous turn, do NOT repeat it. Focus only on the new information requested (like the directions).
    4. ADAPTIVE RESPONSES: If they ask for ONE specific place, give info for that place ONLY. Do not dump the entire city's list. Move other spots to the 'Related Spots' footer.
    5. TONE: Be a polite, proud local guide. Use English, but feel free to use local terms like "Polo," "Karuhatan," or "Malanday" for accuracy.
    6. FORMAT: Always use this HTML structure:
    <div class="response-container">
        <div class="resp-header">HERITAGE TITLE</div>
        <div class="resp-body">CONTENT (use bullets for spots)</div>
        <div class="resp-footer">FOLLOW-UP OR RELATED SPOTS</div>
    </div>
    """
    
    try:
        # Build the message thread with Memory
        messages = [{"role": "system", "content": system_prompt}]
        
        # This loop is what makes the bot remember "Wawang Pulo"
        for entry in request.history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        
        messages.append({"role": "user", "content": request.message})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2
        )
        
        response = completion.choices[0].message.content
        
        # Update history so the next turn remembers this one
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