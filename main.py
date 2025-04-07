from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import io
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3)) 

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=15)  # Use a larger TTF font if available
    except:
        font = ImageFont.load_default()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        color = random_color()

        box = [round(i, 2) for i in box.tolist()]
        x0, y0, x1, y1 = box
        pad = 5
        x0 -= pad
        y0 -= pad
        x1 += pad
        y1 += pad

        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)

        text_size = draw.textbbox((x0, y0), label_text, font=font)
        text_bg = [text_size[0], text_size[1], text_size[2] + 4, text_size[3] + 4]
        draw.rectangle(text_bg, fill=color)
        draw.text((x0 + 2, y0 + 2), label_text, fill="white", font=font)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/jpeg")