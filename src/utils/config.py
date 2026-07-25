"""Configuration management"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
    DATABASE_URL = os.getenv("DATABASE_URL")  # Railway Postgres addon sets this automatically
 
     # Validation
    if not GROQ_API_KEY:
         raise ValueError("GROQ_API_KEY not set in environment variables")
    if not TAVILY_API_KEY:
         raise ValueError("TAVILY_API_KEY not set in environment variables")
