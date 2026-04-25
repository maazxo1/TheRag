"""
Shared requests.Session — one connection pool for all Ollama calls across modules.
"""
import requests

session = requests.Session()
