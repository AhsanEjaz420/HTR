# HTR
Handwritten Text Recognition (HTR) repositories contain code and models designed to detect and convert handwritten text into machine-readable format. These repositories typically use deep learning techniques like CNNs and RNNs with OCR for accurate text recognition.
# Handwritten Text Recognition (HTR) using TrOCR

This project implements a Handwritten Text Recognition (HTR) system using Microsoft's **TrOCR (Transformer-based OCR)** model. TrOCR leverages a pre-trained Vision Transformer (ViT) encoder and a GPT-2 decoder to accurately convert handwritten images into editable text.

## Features
- Uses pre-trained TrOCR model from Hugging Face 🤗
- Supports image input of handwritten text
- Outputs machine-readable recognized text

## Requirements
- Python 3.7+
- torch
- transformers
- torchvision
- PIL

## Usage
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Load and process image
image = Image.open("your_handwritten_image.jpg").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Predict text
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
