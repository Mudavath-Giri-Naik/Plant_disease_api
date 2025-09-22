import google.generativeai as genai
import json
from PIL import Image
import io

# Configure Gemini API
API_KEY = "AIzaSyAs7-TxN98PFupb3tko2TxCEFjAV7jPdAU"
genai.configure(api_key=API_KEY)

try:
    # Test with a simple text prompt first
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello, respond with JSON: {\"test\": \"success\"}")
    print("Text response:", response.text)
    
    # Test with image generation config
    model_with_config = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={
            "temperature": 0.1,
        }
    )
    response2 = model_with_config.generate_content("Respond with JSON: {\"status\": \"working\"}")
    print("JSON response:", response2.text)
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
