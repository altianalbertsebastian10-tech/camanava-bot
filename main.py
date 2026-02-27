from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List
import os

app = FastAPI(title="🤖 Smart NaviGo CAMANAVA Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

# --- IMPROVED KNOWLEDGE BASE ---
KNOWLEDGE = {
    "caloocan": {
        "info": "Caloocan is divided into two sections: South Caloocan (urban hub) and North Caloocan (residential). It is a major gateway for the CAMANAVA area.",
        "spots": ["Andres Bonifacio Monument (Monumento Circle)", "San Roque Cathedral Parish", "La Mesa Watershed", "Caloocan City People's Park"],
        "food": ["Arny Dading Peachy Peachy", "Padi's Point", "SM City Grand Central", "NDY Buffet"],
        "go to": ["From Valenzuela: Take Pier-South or LRT Monumento MCU jeep/bus.", "From Sangandaan: Take a jeepney to LRT Monumento MCU."],
        "malls": ["SM City Grand Central", "Victory Central Mall", "Araneta Square"],
        "tip": "Morning visits = cooler weather!"
    },
    "monumento": {
        "info": "Monumento is the heart of Caloocan, featuring the iconic Andrés Bonifacio Monument designed by Guillermo Tolentino.",
        "roads": ["Samson Road (to Malabon)", "EDSA (to QC)", "McArthur Highway (to Valenzuela)"],
        "food": ["Street food (kwek-kwek, isaw, fishball)", "Lugaw hubs", "Halo-halo stands"],
        "malls": ["SM City Grand Central", "Victory Plaza", "North Mall"],
        "go to": ["LRT Line 1 - Yamaha Monumento Station", "Jeeps from Manila or Valenzuela drop off at Puregold Monumento."],
        "trivia": "Guillermo Tolentino interviewed Bonifacio's sister to ensure the monument's face was accurate."
    },
    "malabon": {
        "info": "Malabon is the culinary soul of CAMANAVA, famous for its heritage homes and 'Pancit Malabon'.",
        "spots": ["Malabon Zoo", "San Bartolome Church (1614)", "Raymundo Ancestral House", "Sy Juco Mansion"],
        "food": ["Pancit Malabon", "Dolor’s Kakanin", "Judy Ann’s Crispy Pata", "Hazel’s Puto", "Valencia Triangulo"],
        "go to": ["From Monumento: Take a jeepney labeled 'Malabon' or 'Hulo'.", "From Valenzuela: Take a jeepney to Sangandaan, then transfer to a Malabon-bound jeep."],
        "tip": "Visit on weekends for fresh kakanin demos!"
    },
    "navotas": {
        "info": "The 'Fishing Capital of the Philippines', Navotas is a coastal city known for shipyards and the freshest seafood.",
        "spots": ["Navotas Fisheries Port", "Centennial Park", "San Jose de Navotas Parish"],
        "food": ["Sinigang na Isda", "Seafood Paluto", "Puto Sulot", "Norma’s Pansit Luglog"],
        "restaurants": ["BABA's Shawarma", "Pia's Boodle Fight", "Bistro Kakamberta", "Samgyupan 199"],
        "go to": ["From Monumento: Take a jeepney labeled 'Navotas' or 'Agora'.", "From C4 Road: There are multiple jeepney routes passing through the Fisheries Port."],
        "tip": "May-June = Bangus Festival!"
    },
    "valenzuela": {
        "info": "The 'Vibrant City', blending industrial growth with heritage parks like the Tagalag Fishing Village.",
        "spots": ["Pio Valenzuela Ancestral House", "San Diego de Alcala Church", "Valenzuela City People’s Park", "Tagalag Fishing Village", "Polo Riverwalk"],
        "food": ["Putong Polo"],
        "restaurants": ["D'Pond", "Alvarez Park and Cafe", "Kamayan sa Palapat", "Snp 'n Roll"],
        "go to": ["Take any jeepney or bus along McArthur Highway labeled 'Malanday' or 'Meycauayan'.", "For Polo: Take a 'Polo' labeled jeepney from Karuhatan."],
        "tip": "Sunset photos at Polo Riverwalk are highly recommended!"
    }
}

class SmartBot:
    def _get_context_city(self, history: List) -> str:
        if not history: return None
        for i in range(-1, -4, -1):
            if i >= -len(history):
                combined = (history[i].get("bot", "") + " " + history[i].get("user", "")).lower()
                for city in KNOWLEDGE:
                    if city in combined: return city
        return None

    def think(self, message: str, history: List) -> str:
        msg = message.lower().strip()
        context_city = self._get_context_city(history)

        # 1. Handle Contextual Category (e.g., user asks "how about food?" after talking about Malabon)
        if context_city:
            for field in KNOWLEDGE[context_city]:
                if field != 'info' and field in msg:
                    return self._build_html(context_city, field)
            
            if any(w in msg for w in ['where','go to','paano','transport']):
                return self._build_html(context_city, 'go to')
            if any(w in msg for w in ['food','kain','restaurant']):
                f = 'restaurants' if 'restaurants' in KNOWLEDGE[context_city] else 'food'
                return self._build_html(context_city, f)

        # 2. Handle City + Category directly
        for city in KNOWLEDGE:
            if city in msg:
                for field in KNOWLEDGE[city]:
                    if field != 'info' and field in msg:
                        return self._build_html(city, field)
                return self._city_intro(city)

        # 3. Fallbacks
        if any(w in msg for w in ["hi", "hello", "start", "navigo"]):
            return "Hello! I am NaviGo. I can help you find food, spots, and directions in CAMANAVA. Which city are you visiting today?"
        
        return "NaviGo here! Try asking about 'Malabon food' or 'Valenzuela spots'."

    def _city_intro(self, city: str) -> str:
        data = KNOWLEDGE[city]
        fields = [f.title() for f in data.keys() if f != 'info']
        return f"""
        <div class="response-container">
            <div class="resp-header">{city.title()}</div>
            <div class="resp-body">{data['info']}<br><br><b>What to expect:</b> {', '.join(fields)}</div>
            <div class="resp-footer">Ask me about '{city} food' or '{city} go to'!</div>
        </div>
        """

    def _build_html(self, city: str, field: str) -> str:
        data = KNOWLEDGE[city][field]
        if isinstance(data, list):
            items = "".join([f"<li>{i}</li>" for i in data])
            body = f"<ul>{items}</ul>"
        else:
            body = data

        return f"""
        <div class="response-container">
            <div class="resp-header">{city.title()} {field.title()}</div>
            <div class="resp-body">{body}</div>
            <div class="resp-footer">Anything else you want to know about {city.title()}?</div>
        </div>
        """

bot = SmartBot()
BLOCKED_WORDS = ["tite", "puke", "burat", "pekpek", "gago", "puta", "bobo", "tanga"]

@app.get("/", response_class=HTMLResponse)
async def get_gui():
    try:
        # Changed to index.html as per your previous convention
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "NaviGo Online. (index.html not found)"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_msg = request.message.lower()
        if any(b in user_msg for b in BLOCKED_WORDS):
            resp = "Let's keep the conversation respectful while exploring CAMANAVA! 😊"
            return {"response": resp, "history": request.history + [{"user": request.message, "bot": resp}]}
        
        response = bot.think(request.message, request.history)
        return {"response": response, "history": (request.history + [{"user": request.message, "bot": response}])[-10:]}
    except Exception:
        return {"response": "I ran into a problem. Try again! 😅", "history": request.history}