import os

print("🔄 Loading environment variables for API keys...")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY or not TAVILY_API_KEY:
    raise ValueError("API keys for Gemini and Tavily must be set in the .env file.")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

print("✅ API keys loaded successfully.")