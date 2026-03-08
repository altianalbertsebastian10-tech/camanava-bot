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
        "info": "Known historically as 'Polo', it is the home of the revolutionary hero Dr. Pio Valenzuela.",
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
    Point your cronjob/UptimeRobot here.
    """
    return {"status": "alive", "city": "Valenzuela", "mode": "RAG"}

@app.post("/chat")
async def chat(request: ChatRequest):
    # Ai response
    context = str(KNOWLEDGE)
    
    system_prompt = f"""
    You are NaviGo, a Cultural Tourism AI for CAMANAVA. 
    DATA: {str(KNOWLEDGE)}

    GUIDELINES:
    1. FOCUS: Only talk about Heritage Spots, Info, Tips, and Directions.
    2. DIRECTIONS RULE: If a user asks 'How to go to [Place]', you MUST ask: 'Saan po kayo manggagaling? (Where are you coming from?)' before giving the route.
    3. Once the user provides their location, use the 'directions_template' to fill in the route.
    4. TONE: Be a polite, proud local guide. Use English.
    5. FORMAT: Always use the HTML structure:
    <div class="response-container">
        <div class="resp-header">HERITAGE TITLE</div>
        <div class="resp-body">CONTENT (use bullets for spots)</div>
        <div class="resp-footer">ASK FOR STARTING LOCATION OR NEXT SPOT</div>
    </div>
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            temperature=0.2 # Keeps it focused on data
        )
        response = completion.choices[0].message.content
        return {"response": response, "history": request.history}
    except Exception as e:
        return {"response": f"System Error: {str(e)}", "history": request.history}

@app.get("/", response_class=HTMLResponse)
async def get_gui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()