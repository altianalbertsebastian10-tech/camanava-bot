from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="NaviGo Localized Logic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your Knowledge Base stays the same
KNOWLEDGE = { ... }

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class NaviGoLogic:
    def __init__(self):
        # We renamed GO_TO to DIRECTIONS and added more "Wayfinding" keywords
        self.intent_map = {
            "FOOD": ["food", "kain", "restaurant", "gutom", "meryenda", "pancit", "puto", "hungry"],
            "SPOTS": ["spots", "visit", "punta", "pasyal", "park", "church", "historical", "attraction", "place"],
            "DIRECTIONS": ["directions", "paano", "sakay", "how to", "transport", "jeep", "way", "daan", "pumunta", "route"]
        }

    def detect_intent(self, msg: str) -> str:
        for intent, keywords in self.intent_map.items():
            if any(word in msg for word in keywords):
                return intent
        return "GENERAL"

    def detect_city(self, msg: str) -> str:
        cities = ["valenzuela", "malabon", "navotas", "caloocan", "monumento"]
        for city in cities:
            if city in msg:
                return city
        return None

    def generate_response(self, message: str) -> str:
        msg = message.lower().strip()
        city = self.detect_city(msg)
        intent = self.detect_intent(msg)

        if not city:
            return """
            <div class="response-container">
                <div class="resp-header">Where to? 📍</div>
                <div class="resp-body">I'm ready to help! Please mention which city in <b>CAMANAVA</b> you want to explore (Valenzuela, Malabon, Navotas, or Caloocan).</div>
                <div class="resp-footer">Try: "How to go to Malabon?"</div>
            </div>
            """

        data = KNOWLEDGE.get(city)
        
        # 1. DIRECTIONS Intent
        if intent == "DIRECTIONS":
            return f"""
            <div class="response-container">
                <div class="resp-header">{city.title()} Directions 🗺️</div>
                <div class="resp-body">{data['go to']}</div>
                <div class="resp-footer">Need food or spots in {city.title()} too?</div>
            </div>
            """
        
        # 2. FOOD Intent
        if intent == "FOOD":
            items = data.get("restaurants") or data.get("food")
            list_items = "".join([f"<li>{i}</li>" for i in items])
            return f"""
            <div class="response-container">
                <div class="resp-header">{city.title()} Food Guide 🍴</div>
                <div class="resp-body"><ul>{list_items}</ul></div>
                <div class="resp-footer">Want the directions to these places?</div>
            </div>
            """
        
        # 3. SPOTS Intent
        if intent == "SPOTS":
            list_spots = "".join([f"<li>{i}</li>" for i in data['spots']])
            return f"""
            <div class="response-container">
                <div class="resp-header">{city.title()} Attractions 🏛️</div>
                <div class="resp-body"><ul>{list_spots}</ul></div>
                <div class="resp-footer">Ask me how to get to these spots!</div>
            </div>
            """

        # 4. GENERAL Intro
        return f"""
        <div class="response-container">
            <div class="resp-header">About {city.title()}</div>
            <div class="resp-body">{data['info']}</div>
            <div class="resp-footer">Try asking for "{city} directions" or "{city} food".</div>
        </div>
        """

bot_logic = NaviGoLogic()

@app.post("/chat")
async def chat(request: ChatRequest):
    response_text = bot_logic.generate_response(request.message)
    return {"response": response_text, "history": request.history}