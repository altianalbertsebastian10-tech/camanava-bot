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
        "info": "Caloocan is divided into two sections: South Caloocan (urban hub) and North Caloocan (residential). It is a major gateway for the CAMANAVA area.",
        "spots": ["Andres Bonifacio Monument (Monumento Circle)", "San Roque Cathedral Parish", "La Mesa Watershed", "Caloocan City People's Park"],
        "food": ["Arny Dading Peachy Peachy", "Padi's Point", "SM City Grand Central", "NDY Buffet"],
        "directions": ["From Valenzuela: Take Pier-South or LRT Monumento MCU jeep/bus.", "From Sangandaan: Take a jeepney to LRT Monumento MCU."],
        "malls": ["SM City Grand Central", "Victory Central Mall", "Araneta Square"],
        "tip": "Morning visits = cooler weather!"
    },
    "monumento": {
        "info": "Monumento is the heart of Caloocan, featuring the iconic Andrés Bonifacio Monument designed by Guillermo Tolentino.",
        "roads": ["Samson Road (to Malabon)", "EDSA (to QC)", "McArthur Highway (to Valenzuela)"],
        "food": ["Street food (kwek-kwek, isaw, fishball)", "Lugaw hubs", "Halo-halo stands"],
        "malls": ["SM City Grand Central", "Victory Plaza", "North Mall", "Araneta Square Mall"],
        "directions": ["LRT Line 1 - Yamaha Monumento Station", "Jeeps from Manila or Valenzuela drop off at Puregold Monumento."],
        "trivia": "Guillermo Tolentino interviewed Bonifacio's sister to ensure the monument's face was accurate."
    },
    "malabon": {
        "info": "Malabon is the culinary soul of CAMANAVA, famous for its heritage homes and 'Pancit Malabon'.",
        "spots": ["Malabon Zoo", "San Bartolome Church (1614)", "Raymundo Ancestral House", "Sy Juco Mansion"],
        "food": ["Pancit Malabon", "Dolor’s Kakanin", "Judy Ann’s Crispy Pata", "Hazel’s Puto", "Valencia Triangulo"],
        "directions": ["From Monumento: Take a jeepney labeled 'Malabon' or 'Hulo'.", "From Valenzuela: Take a jeepney to Sangandaan, then transfer to a Malabon-bound jeep."],
        "tip": "Visit on weekends for fresh kakanin demos!"
    },
    "navotas": {
        "info": "The 'Fishing Capital of the Philippines', Navotas is a coastal city known for shipyards and the freshest seafood.",
        "spots": ["Navotas Fisheries Port", "Centennial Park", "San Jose de Navotas Parish"],
        "food": ["Sinigang na Isda", "Seafood Paluto", "Puto Sulot", "Norma’s Pansit Luglog"],
        "restaurants": ["BABA's Shawarma", "Pia's Boodle Fight", "Bistro Kakamberta", "Samgyupan 199"],
        "directions": ["From Monumento: Take a jeepney labeled 'Navotas' or 'Agora'.", "From C4 Road: There are multiple jeepney routes passing through the Fisheries Port."],
        "tip": "May-June = Bangus Festival!"
    },
    "valenzuela": {
        "info": "The 'Vibrant City', blending industrial growth with heritage parks like the Tagalag Fishing Village.",
        "spots": ["Pio Valenzuela Ancestral House", "San Diego de Alcala Church", "Valenzuela City People’s Park", "Tagalag Fishing Village", "Polo Riverwalk"],
        "food": ["Putong Polo"],
        "restaurants": ["D'Pond", "Alvarez Park and Cafe", "Kamayan sa Palapat", "Snp 'n Roll"],
        "directions": ["Take any jeepney or bus along McArthur Highway labeled 'Malanday' or 'Meycauayan'.", "For Polo: Take a 'Malanday' labeled jeepney from Karuhatan and upon arriving to Malanday take a jeepney labeled 'Paco' and drop off at Polo."],
        "tip": "Sunset photos at Polo Riverwalk are highly recommended!"
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
    # This turns your dictionary into a string for the AI to read
    context = str(KNOWLEDGE)
    
    system_prompt = f"""
    You are NaviGo, a friendly local guide for CAMANAVA.
    DATA: {context}
    
    RULES:
    1. Use ONLY the DATA provided.
    2. If the user is rude or says "shut up", respond politely but stop the info dump.
    3. Use Taglish (Tagalog-English).
    4. Format your answer using this EXACT HTML:
       <div class="response-container">
         <div class="resp-header">TITLE</div>
         <div class="resp-body">CONTENT (use <ul><li> for lists)</div>
         <div class="resp-footer">FOLLOW UP</div>
       </div>
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            temperature=0.2 # Keeps it focused on your data
        )
        response = completion.choices[0].message.content
        return {"response": response, "history": request.history}
    except Exception as e:
        return {"response": f"System Error: {str(e)}", "history": request.history}

@app.get("/", response_class=HTMLResponse)
async def get_gui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()