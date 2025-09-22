# Plant Disease Detection API

A FastAPI application that uses Google Gemini AI to detect plant diseases from uploaded images.

## Features

- Upload plant images (JPG, JPEG, PNG)
- AI-powered disease detection using Google Gemini 2.5 Flash
- Returns disease name, cause, and treatment recommendations
- Handles non-plant images gracefully
- Swagger UI documentation at `/docs`

## API Endpoints

- `POST /predict-disease` - Upload image for disease detection
- `GET /health` - Health check
- `GET /docs` - Swagger UI documentation

## Environment Variables

- `GEMINI_API_KEY` - Your Google Gemini API key (required)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GEMINI_API_KEY="your-api-key-here"

# Run the application
uvicorn main:app --reload
```

## Deployment

This application is configured for deployment on Render.com with the provided Procfile.
