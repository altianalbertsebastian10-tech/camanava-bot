from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import re

app = FastAPI(title="🤖 Smart CAMANAVA Tourism Bot")

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

# COMPLETE KNOWLEDGE BASE
KNOWLEDGE = {
    "caloocan": {
        "info": "Caloocan is a unique city divided into two non-contiguous sections: South Caloocan, which is a dense urban hub near Manila, and North Caloocan, a rapidly developing residential area. For your NaviGo! project, it offers a rich mix of historical landmarks, bustling commercial centers, and local cultural spots",
        "spots": ["Andres Bonifacio Monument (also known as Monumento Circle)", "San Roque Cathedral Parish (also known as Kalookan Cathedral)", "La Mesa Watershed (Novaliches Watershed), Caloocan City People's Park, Glorieta Park"],
        "food": ["Arny Dading Peachy Peachy", "Padi's Point", "SM City Grand Central Food Court", "NDY Buffet"],
        "go to": ["From McArthur Highway in Valenzuela, take jeep or buses labeled Pier-South or LRT Monumento MCU.","From Sangandaan, take a jeepney going to LRT Monumento MCU"],
        "tip": "Morning visits = cooler weather!"
    },
    "monumento": {
        "info": "Monumento refers to the Andrés Bonifacio Monument, a massive bronze memorial dedicated to the ""Father of the Philippine Revolution."" It’s the most iconic landmark in the city and serves as a major gateway for commuters.",
        "creator": "It was designed by Guillermo Tolentino, a National Artist for Sculpture. He actually interviewed Bonifacio’s sister and used her bone structure to ensure the ""Supremo’s"" face was accurate.",
        "roads": ["Samson Road leading to Malabon and Sangandaan, Caloocan", "EDSA leading to Quezon City and Marikina", "McArthur Highway leading to Valenzuela and Northern Luzon"],
        "food": ["Street foods like kwek-kwek, isaw, fishball, betamax, and more", "Comfort foods like lugaw", "Refreshments such as halo-halo and gulaman"],
        "go to": ["If coming from LRT Line 1 - Yamaha Monumento, it will be going to Manila", "If from Manila, take an LRT Line 1 going to Yamaha Monumento Station", "If from Sangandaan, Caloocan, take a jeepney going to Monumento or MCU and drop off at Puregold."],
        "time": "Morning visits are better for elder people. But if you are looking for restaurants, try visiting in the afternoon, most stores and restaurants open up at 3 onwards.",
        "malls": ["SM City Grand Central", "Victory Plaza", "Araneta Square Mall", "North Mall"],
        "trivia": "Four major roads meet at Monumento Circle. First, Samson Road leading to Malabon, EDSA to Quezon City, and McArthur Highway leading to Valenzuela and Northern Luzon."
    },
    "malabon": {
        "info": "Malabon is a neighboring city to Valenzuela and is widely considered the culinary and heritage soul of the CAMANAVA area. Malabon is a goldmine for ""Heritage & Food"" content.",
        "spots": ["Malabon Zoo", "San Bartolome Church (1614)", "Raymundo Ancestral House", "Sy Juco Mansion"],
        "food": ["Pancit Malabon", "Dolor’s Kakanin", "Judy Ann’s Crispy Pata", "Hazel’s Puto", "Valencia Triangulo"],
        "go to": "This section is not available at this time. Please try again later.",
        "tip": "Weekend = fresh pancit demos!"
    },
    "navotas": {
        "info": "Navotas, known as the ""Fishing Capital of the Philippines"", is a vital part of the CAMANAVA region. It is a city defined by its relationship with the sea, featuring a landscape of shipyards, fish markets, and coastal views that offer a unique perspective.",
        "spots": ["Navotas Fisheries Port", "Centennial Park", "Diocesan Shrine and Parish of San Jose de Navotas", "Pia's Boodle Fight", "Bistro Kakamberta"],
        "food": ["Sinigang na Isda & Inihaw", "Seafood Paluto", "Puto Sulot", "Norma’s Pansit Luglog", "Don Benito’s (Navotas Agora)"],
        "restaurants": ["BABA's Shawarma (Tangos North)", "Pia's Boodle Fight (M. Naval St.)", "Bistro Kakamberta (C-4 Road)", "Samgyupan 199"],
        "go to": "This section is not available at this time. Please try again later.",
        "tip": "May-June = Bangus Festival!"
    },
    "valenzuela": {
        "info": "Known as the ""Vibrant City"", it balances its reputation as an industrial powerhouse with surprisingly peaceful heritage sites and a rapidly growing food scene.",
        "spots": ["Pio Valenzuela Ancestral House", "San Diego de Alcala Church (Polo)", "Valenzuela City People’s Park", "Tagalag Fishing Village", "Polo Riverwalk Park", "Arkong Bato Park", "Valenzuela City Skate Park", "Valenzuela City Family Park", "Valenzuela City Sports Park"],
        "food": ["Putong Polo"],
        "restaurants":["D'Pond", "Alvarez Park and Cafe", "Kamayan sa Palapat", "Snp 'n Roll", "Mang Inasal", "Jollibee"],
        "go to": "This section is not available at this time. Please try again later.",
        "tip": "Sunrise jog sa park and sunset photos sa Polo"
    }
}

class SmartBot:
    def __init__(self):
        self.current_city = None

    def _get_context_city(self, history: List) -> str:
        """Find city context from conversation history"""
        if not history:
            return None
        
        # Check last 5 messages for city name
        for i in range(-1, -6, -1):
            if i >= -len(history):
                msg = (history[i]["bot"] + " " + history[i]["user"]).lower()
                for city in KNOWLEDGE:
                    if city in msg:
                        return city
        
        return None
    
    def think(self, message: str, history: List) -> str:
        user_msg = message.lower().strip()
    
        # 1. HISTORY CONTEXT FIRST (most important!)
        context_city = self._get_context_city(history)
        
        if context_city:
            # 🔥 CATEGORY REQUEST WITH CITY CONTEXT
            for field in KNOWLEDGE[context_city]:
                if field != 'info' and field in user_msg:
                    return self._smart_followup_context(context_city, user_msg)
            
            # Semantic natural language
            if any(word in user_msg for word in ['where','nasaan','paano']):
                if 'transport' in KNOWLEDGE[context_city]:
                    return self._smart_followup_context(context_city, 'transport')
            elif any(word in user_msg for word in ['food','kain','murang','sariwa']):
                if 'food' in KNOWLEDGE[context_city]:
                    return self._smart_followup_context(context_city, 'food')
            elif any(word in user_msg for word in ['mall','shopping']):
                if 'malls' in KNOWLEDGE[context_city]:
                    return self._smart_followup_context(context_city, 'malls')
            elif any(word in user_msg for word in ['yes','more','sige']):
                fields = [f for f in KNOWLEDGE[context_city].keys() if f != 'info']
                return f"**{context_city.title()}** - What next?\n• {chr(10).join(fields[:4])}"
        
        # 2. CITY + CATEGORY (like "food in malabon")
        for city in KNOWLEDGE:
            if city in user_msg:
                # Check if user wants specific category
                for field in KNOWLEDGE[city]:
                    if field != 'info' and field in user_msg:
                        return self._smart_followup_context(city, field)
                # No category found, show intro
                return self._city_intro(city)
        
        # 3. General fallback
        if any(word in user_msg for word in ["mall", "malls"]):
            return self._proactive_malls()
        if any(word in user_msg for word in ["food", "pagkain"]):
            return self._food_overview()
        
        return self._greeting()
    
    def _city_intro(self, city: str) -> str:
        data = KNOWLEDGE[city]
        fields = ', '.join([f for f in data.keys() if f != 'info'])
        return f"{data['info']}\n\n. What to expect: \n{fields}\n\n. You might want to ask about '{city} roads' or '{city} food'"
    
    def _smart_followup(self, msg: str) -> str:
        city = self.current_city
        data = KNOWLEDGE[city]
        
        
        msg_lower = msg.lower()
        for field in data:
            if field in msg_lower:
                items = data[field]
                if isinstance(items, list):
                    return f" {city.title()} {field.title()}:\n• {chr(10).join(items)}\n\nAnything else?"
                return f" {city.title()} {field.title()}: {items}\n\nAnother one?"
        
        return f"{city.title()} has: {', '.join(data.keys())} - Try '{city} roads'"
    
    def _smart_followup_context(self, city: str, msg: str) -> str:
        data = KNOWLEDGE[city]
        msg_lower = msg.lower()
    
        for field in data:
            if field != 'info' and field in msg_lower:
                items = data[field]
                if isinstance(items, list):
                    
                    list_items = "\n".join([f"• {item}" for item in items])
                    return f"{city.title()} {field.title()}\n\n{list_items}\n\nAnything else?"
                return f"{city.title()} {field.title()}\n\n{items}\n\nWhat next?"
        
        return f"**{city.title()}** info: {', '.join([f for f in data.keys() if f != 'info'])}"

    def _food_overview(self) -> str:
        return " CAMANAVA Food:\n• Caloocan: Halo-halo\n• Malabon: Pancit\n• Navotas: Bangus\n• Karuhatan: Manok sa Laban\n\n*City first?*"
    
    def _itinerary(self) -> str:
        return " 3-Day Plan:\nDay 1: Caloocan\nDay 2: Malabon+Navotas\nDay 3: Valenzuela+Karuhatan\n\nBudget: P1,500"
    
    def _transport(self) -> str:
        return " Transport:\nMRT → Jeepneys ('Karuhatan Market', 'Caloocan Centro')\nGrab: P100-200"
    
    def _greeting(self) -> str:
        return """Hello! Kamusta? This is NaviGo, your AI Chatbot for CAMANAVA! Are you ready?
        Doesn't know where to start? Try typing: "Valenzuela parks, Caloocan malls, Hotels near me" """

    def _proactive_malls(self) -> str:
        return """**CAMANAVA MALLS Guide!**

    Popular Malls:
    •*Karuhatan: Puregold Karuhatan, SM City Valenzuela (5km)
    • Caloocan: SM North EDSA (near), Victory Central Mall
    • Malabon: SM Marilao (close)
    • Navotas: Navotas Blvd commercial area

    Nasaan ka ngayon? 
    • "karuhatan" • "caloocan" • "malabon" • "navotas"
    → Gets you **NEAREST MALLS + DIRECTIONS**! """


bot = SmartBot()

# BLOCKED WORDS LIST
BLOCKED_WORDS = [
    "tite", "puke", "burat", "pekpek", "suso", "dede", 
    "putanginamo", "damn", "fucking", "shit", "fuck", "gago",
    "puta", "yawa", "bobo", "tanga", "bwisit","tanginamo", "putang ina mo", "tangina mo", "g@g0", "faggot", "nigga"
]

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 🔥 PROFANITY FILTER FIRST
        user_msg_lower = request.message.lower()
        if any(blocked_word in user_msg_lower for blocked_word in BLOCKED_WORDS):
            friendly_response = "Hey! Maybe you are frustrated, I won't respond to that though. 😊"
            new_history = request.history + [{"user": request.message, "bot": friendly_response}]
            return {"response": friendly_response, "history": new_history[-10:]}
        
        # Normal bot logic
        response = bot.think(request.message, request.history)
        new_history = request.history + [{"user": request.message, "bot": response}]
        return {"response": response, "history": new_history[-10:]}
    
    except Exception as e:
        # FRIENDLY ERROR MESSAGE (instead of server crash)
        error_msg = "Sorry, I think I ran into a problem. It's not you, it's me. Please try again, shall we? 😅"
        new_history = request.history + [{"user": request.message, "bot": error_msg}]
        return {"response": error_msg, "history": new_history[-10:]}

@app.get("/")
async def root():
    return {"message": "🤖 CAMANAVA Bot - 100% Working!"}
