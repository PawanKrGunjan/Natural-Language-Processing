from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import Voice_assistent  # Your custom module

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

app = FastAPI()

# Serve static files (e.g., CSS, JS) - create a 'static' folder if needed
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for HTML (place your index.html in a 'templates' folder)
templates = Jinja2Templates(directory="templates")

# Initialize the chatbot
chatbot = Voice_assistent.VoiceChatbot('secrets.json')

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_audio")
async def process_audio(audio_data: UploadFile = File(...)):
    # Optional: validate file type
    if not audio_data.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File is not audio")
    
    audio_path = os.path.join("uploads", "user_audio.wav")
    
    # Save the uploaded audio
    with open(audio_path, "wb") as f:
        contents = await audio_data.read()
        f.write(contents)
    
    print(f"Audio saved: {audio_path}")
    
    # Convert voice to text
    text = chatbot.convert_voice_to_text(audio_path)
    print(f"User said: {text}")
    
    return JSONResponse({"text": text})

class TextRequest(BaseModel):
    text: str

@app.post("/get_response")
async def get_response(req: TextRequest):
    response_text = chatbot.get_response(req.text)
    print(f"Chatbot response: {response_text}")
    
    return JSONResponse({"response_text": response_text})

@app.get("/get_audio/{filename}")
async def get_audio(filename: str):
    audio_response_path = chatbot.speak_text(filename, Play=False)
    print(f"AUDIO RESPONSE PATH: {audio_response_path}")
    
    if os.path.exists(audio_response_path):
        return FileResponse(audio_response_path, media_type="audio/wav")
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

# Run with: uvicorn your_filename:app --reload