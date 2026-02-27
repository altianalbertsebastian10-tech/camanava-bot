from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List
import os

app = FastAPI(title="🤖 Smart CAMANAVA Tourism Bot")

# Enable CORS for frontend flexibility
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

# --- KNOWLEDGE BASE ---
KNOWLEDGE = {
    "caloocan": {
        "info": "Caloocan is a unique city divided into two non-contiguous sections: South Caloocan, a dense urban hub, and North Caloocan, a rapidly developing residential area. It offers a rich mix of historical landmarks and local cultural spots.",
        "spots": ["Andres Bonifacio Monument (Monumento Circle)", "San Roque Cathedral Parish", "La Mesa Watershed", "Caloocan City People's Park"],
        "food": ["Arny Dading Peachy Peachy", "Padi's Point", "SM City Grand Central Food Court", "NDY Buffet"],
        "go to": ["From Valenzuela, take Pier-South or LRT Monumento MCU jeep/bus.", "From Sangandaan, take a jeepney going to LRT Monumento MCU."],
        "tip": "Morning visits = cooler weather!"
    },
    "monumento": {
        "info": "Monumento refers to the Andrés Bonifacio Monument, the most iconic landmark in Caloocan, serving as a major gateway for commuters.",
        "roads": ["Samson Road (to Malabon)", "EDSA (to QC)", "McArthur Highway (to Valenzuela)"],
        "food": ["Street foods (kwek-kwek, isaw, fishball)", "Comfort foods (lugaw)", "Halo-halo and gulaman"],
        "malls": ["SM City Grand Central", "Victory Plaza", "Araneta Square Mall"],
        "go to": ["LRT Line 1 Yamaha Monumento Station", "Jeepney from Sangandaan to Monumento/Puregold."],
        "trivia": "Designed by Guillermo Tolentino, who interviewed Bonifacio’s sister for accuracy."
    },
    "malabon": {
        "info": "Malabon is the culinary and heritage soul of the CAMANAVA area, a goldmine for heritage homes and local delicacies.",
        "spots": ["Malabon Zoo", "San Bartolome Church (1614)", "Raymundo Ancestral House", "Sy Juco Mansion"],
        "food": ["Pancit Malabon", "Dolor’s Kakanin", "Judy Ann’s Crispy Pata", "Hazel’s Puto", "Valencia Triangulo"],
        "go to": "This section is not available at this time. Please try again later.",
        "tip": "Weekend = fresh pancit demos!"
    },
    "navotas": {
        "info": "Known as the 'Fishing Capital of the Philippines', Navotas is defined by shipyards, fish markets, and coastal views.",
        "spots": ["Navotas Fisheries Port", "Centennial Park", "San Jose de Navotas Parish"],
        "food": ["Sinigang na Isda & Inihaw", "Seafood Paluto", "Puto Sulot", "Norma’s Pansit Luglog"],
        "restaurants": ["BABA's Shawarma", "Pia's Boodle Fight", "Bistro Kakamberta", "Samgyupan 199"],
        "go to": "This section is not available at this time. Please try again later.",
        "tip": "May-June = Bangus Festival!"
    },
    "valenzuela": {
        "info": "Known as the 'Vibrant City', it balances industrial power with peaceful heritage sites and a growing food scene.",
        "spots": ["Pio Valenzuela Ancestral House", "San Diego de Alcala Church", "Valenzuela City People’s Park", "Tagalag Fishing Village", "Polo Riverwalk"],
        "food": ["Putong Polo"],
        "restaurants": ["D'Pond", "Alvarez Park and Cafe", "Kamayan sa Palapat", "Snp 'n Roll"],
        "go to": "This section is not available at this time. Please try again later.",
        "tip": "Sunrise jog sa People's Park and sunset photos sa Polo."
    }
}

class SmartBot:
    def _get_context_city(self, history: List) -> str:
        if not history: return None
        for i in range(-1, -6, -1):
            if i >= -len(history):
                msg = (history[i]["bot"] + " " + history[i]["user"]).lower()
                for city in KNOWLEDGE:
                    if city in msg: return city
        return None
    
    def think(self, message: str, history: List) -> str:
        user_msg = message.lower().strip()
        context_city = self._get_context_city(history)
        
        # 1. Contextual Category Request
        if context_city:
            for field in KNOWLEDGE[context_city]:
                if field != 'info' and field in user_msg:
                    return self._format_response(context_city, field)
            
            # Semantic keyword detection
            if any(word in user_msg for word in ['where','nasaan','paano','go to','transport']):
                return self._format_response(context_city, 'go to')
            elif any(word in user_msg for word in ['food','kain','kainan','restaurant']):
                # Priority: Check restaurants field first, then food
                field = 'restaurants' if 'restaurants' in KNOWLEDGE[context_city] else 'food'
                return self._format_response(context_city, field)
        
        # 2. Direct City + Category Check
        for city in KNOWLEDGE:
            if city in user_msg:
                for field in KNOWLEDGE[city]:
                    if field != 'info' and field in user_msg:
                        return self._format_response(city, field)
                return self._city_intro(city)
        
        # 3. Fallbacks
        if any(word in user_msg for word in ["hi", "hello", "start", "kamusta"]):
            return "Hello! Kamusta? This is NaviGo, your AI Chatbot for CAMANAVA! Ready to explore? Try typing a city name like 'Valenzuela' or 'Malabon food'."
        
        return "NaviGo here! I'm best at finding Food, Spots, and Directions in CAMANAVA. Which city are you interested in?"

    def _city_intro(self, city: str) -> str:
        data = KNOWLEDGE[city]
        fields = ', '.join([f for f in data.keys() if f != 'info'])
        return f"{data['info']} • What to expect: {fields} • Ask me about '{city} food' or '{city} spots'!"

    def _format_response(self, city: str, field: str) -> str:
        items = KNOWLEDGE[city][field]
        if isinstance(items, list):
            list_str = " • ".join(items)
            return f"{city.title()} {field.title()} • {list_str} Anything else?"
        return f"{city.title()} {field.title()} {items} What next?"

bot = SmartBot()
BLOCKED_WORDS = ["tite", "puke", "burat", "pekpek", "gago", "puta", "bobo", "tanga", "nigga"]

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def get_gui():
    """Serves the NaviGo HTML GUI when visiting the root URL."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Welcome to NaviGo! (index.html not found in repository)"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_msg_lower = request.message.lower()
        if any(blocked in user_msg_lower for blocked in BLOCKED_WORDS):
            resp = "I'm here to help with CAMANAVA tourism! Let's keep the conversation friendly. 😊"
            return {"response": resp, "history": request.history + [{"user": request.message, "bot": resp}]}
        
        response = bot.think(request.message, request.history)
        return {"response": response, "history": (request.history + [{"user": request.message, "bot": response}])[-10:]}
    except Exception:
        err = "Sorry, I ran into a tiny problem. Please try again! 😅"
        return {"response": err, "history": request.history}