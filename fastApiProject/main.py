from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import json
import webbrowser
import aiofiles

# Import your model and preprocessing functions
from models.inception_model import InceptionModel
print("InceptionModel loaded successfully.")
from models.preprocess import preprocess_image, preprocess_sequence
from models.captioning_model import CaptioningModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



import pickle

# Load wordtoix dictionary
with open('wordtoix.pkl', 'rb') as f:
    wordtoix = pickle.load(f)

# Load ixtoword dictionary
with open('ixtoword.pkl', 'rb') as f:
    ixtoword = pickle.load(f)

# Ensure vocabulary size is correct
vocab_size = len(ixtoword)
print(f"Vocabulary Size: {vocab_size}")

max_length = 33  # Set the max_length value based on your model's training

# Load the models (inception model and captioning model)
inception_model = InceptionModel()
print("Loading CaptioningModel...")
captioning_model = CaptioningModel('latest_model.h5', wordtoix, ixtoword, max_length)
print("CaptioningModel loaded successfully.")
print(captioning_model.model.input)

@app.get("/")
async def get_html():
    async with aiofiles.open("index.html", mode="r") as f:
        content = await f.read()
    return HTMLResponse(content=content)

@app.post("/caption/")
async def caption_image(file: UploadFile = File(...)):
    try:
        print("Received file:", file.filename)
        # Read the image file
        image = Image.open(io.BytesIO(await file.read()))
        print("Image opened successfully.")

        # Preprocess the image
        processed_image = preprocess_image(image, inception_model)
        print("Image processed:", processed_image.shape)

        # Generate caption using an empty sequence as a starting point
        initial_sequence = "startseq"
        processed_sequence = preprocess_sequence(initial_sequence, wordtoix, max_length)
        print("Sequence processed:", processed_sequence.shape)


        # Run the model prediction
        caption = captioning_model.predict(processed_image, processed_sequence)
        print("Caption generated:", caption)

        return {"caption": caption}
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import time

    def open_browser():
        time.sleep(2)  # Wait a few seconds for the server to start
        webbrowser.open("http://localhost:8000")

    # Open the browser in a separate thread
    import threading
    threading.Thread(target=open_browser).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
