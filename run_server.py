import os
import sys
import uvicorn
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    env = os.environ.get("ENVIRONMENT", "development")
    reload_flag = env.lower() == "development"
    print(f"Starting Kryon Academic Regulations QA System [{env}] on http://{host}:{port}")
    uvicorn.run("src.api.server:app", host=host, port=port, reload=reload_flag)
