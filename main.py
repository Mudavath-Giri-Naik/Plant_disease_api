from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import google.generativeai as genai
import json
import io
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using Google Gemini 2.5 Flash",
    version="1.0.0"
)

# Configure Gemini API
API_KEY = "AIzaSyAs7-TxN98PFupb3tko2TxCEFjAV7jPdAU"
genai.configure(api_key=API_KEY)

# Initialize the model
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config={
        "temperature": 0.1,
    }
)

# Allowed image types
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/jpg', 'image/png'}

def validate_image_file(file: UploadFile) -> bool:
    """
    Validate that the uploaded file is an image with allowed extension and MIME type.
    """
    if not file.filename:
        return False
    
    # Check file extension
    file_extension = '.' + file.filename.split('.')[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return False
    
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        return False
    
    return True

def process_image_for_gemini(image_bytes: bytes) -> Image.Image:
    """
    Process image bytes and return a PIL Image object for Gemini.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary (Gemini works better with RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

def clean_json_response(response_text: str) -> dict:
    """
    Clean and parse JSON response from Gemini, removing any markdown fences.
    """
    try:
        # Remove markdown code fences if present
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        if cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        
        cleaned_text = cleaned_text.strip()
        
        # Parse JSON
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}, response: {response_text}")
        raise HTTPException(status_code=500, detail="Failed to parse AI response")

@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image using Google Gemini 2.5 Flash.
    
    Args:
        file: Image file (jpg, jpeg, png)
    
    Returns:
        JSON with disease, cause, and treatment information
    """
    try:
        # Validate file
        if not validate_image_file(file):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a JPG, JPEG, or PNG image."
            )
        
        # Read and process image
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process image for Gemini
        image = process_image_for_gemini(image_bytes)
        
        # Create prompt for Gemini
        prompt = """
        Analyze this image and determine if it shows a plant leaf or plant part. 
        If it's clearly a plant leaf, identify any disease present and provide detailed information.
        If it's clearly not a plant leaf (human, animal, object, etc.), return the specific response for non-plant images.
        
        IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text, explanations, or markdown formatting.
        
        The JSON object must contain exactly these three fields:
        - "disease": The name of the disease (or "Unknown" if not a plant leaf)
        - "cause": The cause of the disease (or "Not a valid plant leaf" if not a plant leaf)  
        - "treatment": Recommended treatment (or "N/A" if not a plant leaf)
        
        Example format:
        {"disease": "Leaf Spot", "cause": "Fungal infection", "treatment": "Apply fungicide"}
        
        Be thorough in your analysis and provide specific, actionable information.
        """
        
        # Generate response using Gemini
        try:
            logger.info("Sending request to Gemini API...")
            response = model.generate_content([prompt, image])
            logger.info(f"Gemini response received: {response.text[:200]}...")
            response_text = response.text
            
            # Clean and parse JSON response
            result = clean_json_response(response_text)
            logger.info(f"Parsed result: {result}")
            
            # Validate required fields
            required_fields = ['disease', 'cause', 'treatment']
            for field in required_fields:
                if field not in result:
                    raise HTTPException(
                        status_code=500,
                        detail=f"AI response missing required field: {field}"
                    )
            
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"AI service error: {str(e)}"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict_disease": "POST /predict-disease"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "service": "Plant Disease Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
