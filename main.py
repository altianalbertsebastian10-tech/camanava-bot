from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List
import os

app = FastAPI(title="🤖 NaviGo - Smart CAMANAVA Tourism Guide")

# Enable CORS for local and online development
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

# --- EXTENDED KNOWLEDGE BASE ---
KNOWLEDGE = {
    "caloocan": {
        "info": "Caloocan is a historic city divided into South (urban hub) and North (residential/nature). It is home to the iconic Monumento.",
        "spots": ["Andres Bonifacio Monument (Monumento Circle)", "San Roque Cathedral Parish", "La Mesa Watershed", "Caloocan City People's Park", "Gubat sa Ciudad"],
        "food": ["Arny Dading Peachy Peachy", "Padi's Point", "SM City Grand Central", "NDY Buffet"],
        "go to": ["From Valenzuela: Take Pier-South or LRT Monumento MCU jeep/bus.", "From Sangandaan: Take a jeepney to LRT Monumento MCU."],
        "malls": ["SM City Grand Central", "Victory Central Mall", "Araneta Square"],
        "tip": "Morning visits = cooler weather!"
    },
    "monumento": {
        "info": "Monumento is the heart of Caloocan, featuring the Andrés Bonifacio Monument—a masterpiece by National Artist Guillermo Tolentino.",
        "roads": ["Samson Road (to Malabon)", "EDSA (to QC)", "McArthur Highway (to Valenzuela)"],
        "food": ["Street food (kwek-kwek, isaw, fishball)", "Lugaw hubs", "Halo-halo stands"],
        "malls": ["SM City Grand Central", "Victory Plaza", "North Mall", "Araneta Square Mall"],
        "go to": ["LRT Line 1 - Yamaha Monumento Station", "Jeeps from Manila or Valenzuela drop off at Puregold Monumento."],
        "trivia": "The monument face was modeled after Bonifacio’s sister for historical accuracy."
    },
    "malabon": {
        "info": "Malabon is the 'Culinary Soul' of CAMANAVA, famous for its heritage homes and legendary Pancit Malabon.",
        "spots": ["Malabon Zoo", "San Bartolome Church (1614)", "Raymundo Ancestral House", "Sy Juco Mansion", "Malabon City Square"],
        "food": ["Pancit Malabon", "Dolor’s Kakanin", "Judy Ann’s Crispy Pata", "Hazel’s Puto", "Valencia Triangulo"],
        "go to": ["From Monumento: Take a jeepney labeled 'Malabon' or 'Hulo'.", "From Valenzuela: Take a jeepney to Sangandaan, then transfer to a Malabon-bound jeep."],
        "tip": "Visit on weekends for fresh kakanin demos!"
    },
    "navotas": {
        "info": "Known as the 'Fishing Capital of the Philippines', Navotas offers a unique look at local shipyards and coastal life.",
        "spots": ["Navotas Fisheries Port", "Centennial Park", "San Jose de Navotas Parish", "Agora Market"],
        "food": ["Sinigang na Isda", "Seafood Paluto", "Puto Sulot", "Norma’s Pansit Luglog"],
        "restaurants": ["BABA's Shawarma", "Pia's Boodle Fight", "Bistro Kakamberta", "Samgyupan 199"],
        "go to": ["From Monumento: Take a jeepney labeled 'Navotas' or 'Agora'.", "From C4 Road: Multiple jeepney routes pass through the Fisheries Port."],
        "tip": "May-June = Bangus Festival!"
    },
    "valenzuela": {
        "info": "The 'Vibrant City', blending industrial growth with peaceful heritage parks and the Tagalag Fishing Village.",
        "spots": ["Pio Valenzuela Ancestral House", "San Diego de Alcala Church", "Valenzuela City People’s Park", "Tagalag Fishing Village", "Polo Riverwalk", "Arkong Bato"],
        "food": ["Putong Polo"],
        "restaurants": ["D'Pond", "Alvarez Park and Cafe", "Kamayan sa Palapat", "Snp 'n Roll"],
        "go to": ["Take any jeepney/bus along McArthur Highway labeled 'Malanday' or 'Meycauayan'.", "To reach Polo: From Karuhatan, take a Malanday-bound jeep. At Malanday, transfer to a jeep labeled 'Paco' and drop off at Polo."],
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

        # 1. Handle Contextual Category
        if context_city:
            # Map common synonyms to knowledge base keys
            synonyms = {
                'spots': ['spots', 'attractions', 'places', 'visit', 'parks', 'monument', 'landmark'],
                'food': ['food', 'kain', 'restaurant', 'hungry', 'meryenda', 'delicacy'],
                'go to': ['where', 'go to', 'paano', 'transport', 'directions', 'bus', 'jeep']
            }

            for key, words in synonyms.items():
                if any(w in msg for w in words):
                    target_field = key
                    # Specific restaurant logic
                    if key == 'food' and 'restaurants' in KNOWLEDGE[context_city]:
                        target_field = 'restaurants'
                    return self._build_html(context_city, target_field)

        # 2. Handle City + Category directly
        for city in KNOWLEDGE:
            if city in msg:
                for field in KNOWLEDGE[city]:
                    if field != 'info' and field in msg:
                        return self._build_html(city, field)
                return self._city_intro(city)

        # 3. Fallbacks and Greetings
        if any(w in msg for w in ["hi", "hello", "start", "kamusta"]):
            return """
            <div class="response-container">
                <div class="resp-header">Welcome to NaviGo! 🤖</div>
                <div class="resp-body">I'm your tourism guide for the <b>CAMANAVA</b> area (Caloocan, Malabon, Navotas, Valenzuela).<br><br>
                Try asking me things like:
                <ul>
                    <li>"Tell me about Malabon food"</li>
                    <li>"Where to go in Valenzuela?"</li>
                    <li>"How to get to Monumento?"</li>
                </ul>
                </div>
                <div class="resp-footer">Which city are you exploring today?</div>
            </div>
            """
        
        return "NaviGo here! I'm best at finding Food, Spots, and Directions in CAMANAVA. Which city are you interested in?"

    def _city_intro(self, city: str) -> str:
        data = KNOWLEDGE[city]
        fields = [f.title() for f in data.keys() if f != 'info']
        return f"""
        <div class="response-container">
            <div class="resp-header">{city.title()}</div>
            <div class="resp-body">{data['info']}<br><br><b>Explore {city.title()} by asking about:</b> {', '.join(fields)}</div>
            <div class="resp-footer">Try: "{city} spots" or "{city} food"</div>
        </div>
        """

    def _build_html(self, city: str, field: str) -> str:
        data = KNOWLEDGE[city].get(field, "I don't have information on that yet.")
        if isinstance(data, list):
            items = "".join([f"<li>{i}</li>" for i in data])
            body = f"<ul>{items}</ul>"
        else:
            body = data

        return f"""
        <div class="response-container">
            <div class="resp-header">{city.title()} {field.title()}</div>
            <div class="resp-body">{body}</div>
            <div class="resp-footer">Would you like to know about other parts of {city.title()}?</div>
        </div>
        """

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
            resp = "I'm here to help you explore CAMANAVA! Let's keep our conversation respectful. 😊"
            return {"response": resp, "history": request.history + [{"user": request.message, "bot": resp}]}
        
        response = bot.think(request.message, request.history)
        return {"response": response, "history": (request.history + [{"user": request.message, "bot": response}])[-10:]}
    except Exception:
        return {"response": "I ran into a problem. Try again! 😅", "history": request.history}