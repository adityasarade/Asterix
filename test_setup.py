# test_setup.py
import os
import requests
from letta_client import Letta
from dotenv import load_dotenv
load_dotenv()


def test_letta_connection():
    try:
        client = Letta(base_url='http://localhost:8283')
        models = client.models.list()
        print("✅ Letta server connection: SUCCESS")
        print(f"Available models: {len(models)} found")
        return True
    except Exception as e:
        print(f"❌ Letta server connection: FAILED - {e}")
        return False

def test_api_keys():
    groq_key = os.getenv('GROQ_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_key = os.getenv('QDRANT_API_KEY')
    
    print(f"✅ GROQ_API_KEY: {'SET' if groq_key else 'NOT SET'}")
    print(f"✅ OPENAI_API_KEY: {'SET' if openai_key else 'NOT SET'}")
    print(f"✅ QDRANT_URL: {'SET' if qdrant_url else 'NOT SET'}")
    print(f"✅ QDRANT_API_KEY: {'SET' if qdrant_key else 'NOT SET'}")
    
    return all([groq_key, openai_key, qdrant_url, qdrant_key])

if __name__ == "__main__":
    print("=== MemGPT Setup Verification ===")
    letta_ok = test_letta_connection()
    keys_ok = test_api_keys()
    
    if letta_ok and keys_ok:
        print("\n🎉 Setup verification PASSED! Ready for Step 2.")
    else:
        print("\n⚠️  Setup verification FAILED. Please check the issues above.")