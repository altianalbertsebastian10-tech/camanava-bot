🤖 NaviGo: Smart CAMANAVA Tourism Guide
NaviGo is a localized, intelligent tourism chatbot designed to help users explore the CAMANAVA area (Caloocan, Malabon, Navotas, and Valenzuela). Unlike generic chatbots, NaviGo utilizes Intent Detection and Logic-Based Mapping to provide accurate food recommendations, historical spot information, and precise travel directions—all optimized for low-spec mobile devices.

🌟 Key Features
Localized Intent Detection: Understands natural language and Taglish phrases (e.g., "Gutom na ako" or "Saan may meryenda?") to identify user needs like FOOD, DIRECTIONS, or SPOTS.
Entity Recognition: Automatically identifies which city in CAMANAVA the user is inquiring about.
Offline-Ready Architecture: Designed for a Native AI transition using TensorFlow Lite, ensuring the app stays lightweight and functional without heavy model downloads.
Modern Glassmorphism UI: A high-end frontend built with Inter and Poppins typography, featuring responsive dark mode and mobile-first design.
Digital Inclusion: Optimized to run smoothly on devices like the Oppo A55 and iPhone 12 by minimizing RAM and storage usage.

🛠️ Technical Stack
Backend: Python with FastAPI.
Logic Engine: Custom Python class for Intent & Entity mapping.
Frontend: HTML5, CSS3 (Modern Glassmorphism), and Vanilla JavaScript.
Deployment: Hosted on Render (for the web prototype).
Mobile Target: Android (Java/Kotlin) and iOS (Swift) using Native Intent Classification.

📂 Project Structure
├── main.py              # FastAPI Backend & Logic Engine
├── index.html           # Modern Glassmorphism Frontend
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

🚀 Getting Started
Prerequisites:
Python 3.10+
pip

Installation

Clone the repository:

git clone https://github.com/yourusername/NaviGo.git
cd NaviGo
Install dependencies:


pip install -r requirements.txt
Run the local server:


uvicorn main:app --reload

Access the GUI:
Open your browser and navigate to http://127.0.0.1:8000.

🏛️ Localized Knowledge Base
NaviGo contains deep-dive research into:
Caloocan: Monumento Circle, San Roque Cathedral, and local malls.
Malabon: Culinary heritage, ancestral houses, and the famous Pancit Malabon.
Navotas: Fishing port culture and fresh seafood guides.
Valenzuela: Tagalag Fishing Village, People's Park, and heritage spots like Putong Polo.

🛡️ Digital Ethics & Privacy
NaviGo is built with a "Privacy First" mindset. By moving toward Native On-Device AI, we ensure that user data stays on the device and the application remains accessible to tourists with limited data plans or low-storage smartphones.

This repository is a test environment for the NaviGo project. It is used to validate the Logic-Based Mapping and Intent Detection systems before they are integrated into the final native mobile application.
As a test repository, the focus is on:
Prototyping: Testing the FastAPI backend and logic classes.
Refining Intents: Improving how the system identifies FOOD, DIRECTIONS, and SPOTS.
UI/UX Iteration: Polishing the glassmorphism frontend for mobile compatibility.

WE HOPE YOU FIND SOMETHING YOU NEED.
