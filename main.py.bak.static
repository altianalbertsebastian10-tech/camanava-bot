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

# --- KNOWLEDGE BASE stays the same ---
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

class SmartBot:
    def __init__(self):
        self.INTENT_MAP = {
            "FOOD": ["food", "kain", "gutom", "hungry", "restaurant", "meryenda", "lunch", "dinner", "maki-kain"],
            "SPOTS": ["spot", "attraction", "visit", "punta", "pasyal", "park", "church", "historical", "place", "view"],
            "DIRECTIONS": ["direction", "paano", "way", "route", "sakay", "jeep", "bus", "transport", "lrt", "landmark", "how to"],
            "MALLS": ["mall", "shopping", "bili", "sm", "grocery"],
            "STOP": ["shut up", "stop", "quiet", "tama na", "manahimik", "ayoko na"],
            "GREET": ["hi", "hello", "hey", "kamusta", "start"]
        }

    def _get_entity(self, message: str) -> str:
        for city in KNOWLEDGE:
            if city in message:
                return city
        return None

    def _detect_intent(self, message: str) -> str:
        for intent, keywords in self.INTENT_MAP.items():
            if any(keyword in message for keyword in keywords):
                return intent
        return "GENERAL"

    def _get_context_city(self, history: List) -> str:
        if not history: return None
        for i in range(-1, -4, -1):
            if i >= -len(history):
                user_msg = history[i].get("user", "").lower()
                city = self._get_entity(user_msg)
                if city: return city
        return None

    def think(self, message: str, history: List) -> str:
        msg = message.lower().strip()
        intent = self._detect_intent(msg)
        
        # 1. Handle Command Intents First (Stop/Shut up)
        if intent == "STOP":
            return "Understood. I'll stop the tour info. What else can I do for you?"

        # 2. Handle Repeated Greetings
        if intent == "GREET":
            if len(history) > 0 and self._detect_intent(history[-1].get("user", "").lower()) == "GREET":
                return "Hello again! Just a reminder: I'm here to help with <b>CAMANAVA</b> tourism. Which city would you like to know more about?"
            return self._welcome_screen()

        # 3. Handle City Discovery
        target_city = self._get_entity(msg)
        
        # If no city in current message, check history IF the user is asking a relevant question
        if not target_city and intent != "GENERAL":
            target_city = self._get_context_city(history)

        # 4. Logic-Based Mapping
        if target_city:
            city_data = KNOWLEDGE[target_city]
            
            if intent == "FOOD":
                field = 'restaurants' if 'restaurants' in city_data else 'food'
                return self._build_html(target_city, field, "Food & Dining")
            elif intent == "DIRECTIONS":
                return self._build_html(target_city, 'directions', "How to get there")
            elif intent == "SPOTS":
                return self._build_html(target_city, 'spots', "Must-visit Attractions")
            
            # If they just mentioned a city with no intent, don't just dump info if they already got it.
            if len(history) > 0 and target_city in history[-1].get("bot", "").lower() and intent == "GENERAL":
                 return f"We're currently looking at <b>{target_city.title()}</b>. Do you want to know about its food or directions?"
            
            return self._city_intro(target_city)

        return "I'm not sure which city you're asking about. Try mentioning <b>Caloocan, Malabon, Navotas, or Valenzuela</b>!"

    def _welcome_screen(self):
        return """
        <div class="response-container">
            <div class="resp-header">Welcome back! 🤖</div>
            <div class="resp-body">I'm your tourism guide. You can ask me things like:<br>
            • <i>"Saan pwedeng kumain sa Malabon?"</i><br>
            • <i>"Valenzuela attractions"</i></div>
            <div class="resp-footer">How can I help you?</div>
        </div>
        """

    def _city_intro(self, city: str) -> str:
        data = KNOWLEDGE[city]
        return f"""
        <div class="response-container">
            <div class="resp-header">Exploring {city.title()}</div>
            <div class="resp-body">{data['info']}</div>
            <div class="resp-footer">Ask me about food or directions in {city.title()}!</div>
        </div>
        """

    def _build_html(self, city: str, field: str, label: str) -> str:
        data = KNOWLEDGE[city][field]
        items = "".join([f"<li>{i}</li>" for i in data]) if isinstance(data, list) else data
        body = f"<ul>{items}</ul>" if isinstance(data, list) else data
        return f"""
        <div class="response-container">
            <div class="resp-header">{city.title()} - {label}</div>
            <div class="resp-body">{body}</div>
            <div class="resp-footer">Anything else about {city.title()}?</div>
        </div>
        """

bot = SmartBot()
# BLOCKED_WORDS and Routes stay the same...
bot = SmartBot()
BLOCKED_WORDS = ["tite", "puke", "burat", "pekpek", "gago", "puta", "bobo", "tanga", "putang ina mo", "putanginamo", "tangina mo", "tanginamo"]

@app.get("/", response_class=HTMLResponse)
async def get_gui():
    try:
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