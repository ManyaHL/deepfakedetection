# Save this file as app.py
import torch
import cv2
import numpy as np
import os
import mimetypes
import argparse
import piexif
import piexif.helper
import gradio as gr

from PIL import Image
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
from datetime import timedelta
from imwatermark import WatermarkDecoder

# --- NEW IMPORTS FOR WEB SERVER ---
from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS # For allowing browser communication
import time

try:
    import pillow_avif
    import pillow_heif
except ImportError:
    print("⚠️ Install 'pillow-avif-plugin' and 'pillow-heif' for full format support.")

# --- FLASK APP SETUP ---
app = Flask(__name__)
# Allow all origins (for development)
CORS(app) 

# --- MODEL LOADING ---
accelerator = Accelerator()
decoder = WatermarkDecoder("bytes", 32)

def load_models():
    """Loads CLIP and ViT models for image feature extraction."""
    try:
        print("🔄 Initializing models...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(
            accelerator.device
        )
        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", use_fast=False
        )
        vit_model = ViTModel.from_pretrained("google/vit-large-patch32-224-in21k").to(
            accelerator.device
        )
        vit_processor = ViTImageProcessor.from_pretrained(
            "google/vit-large-patch32-224-in21k"
        )
        print("✅ Models are ready.")
        return clip_model, clip_processor, vit_model, vit_processor
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        exit(1)

# --- LOAD MODELS ONCE AT STARTUP ---
CLIP_MODEL, CLIP_PROCESSOR, VIT_MODEL, VIT_PROCESSOR = load_models()

# --- ALL YOUR ANALYSIS FUNCTIONS (Unchanged) ---
def load_image(image_path):
    try:
        # Note: This is opening a file path, not reading file data directly
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
def estimate_noise(image):
    gray_image = np.array(image.convert("L"))
    return np.std(gray_image)
def analyze_texture(image):
    gray_image = np.array(image.convert("L"))
    edges = cv2.Canny(gray_image, 50, 150)
    return np.sum(edges) / edges.size
def detect_repeating_patterns(image):
    img_np = np.array(image.convert("L"))
    fft = np.fft.fft2(img_np)
    magnitude_spectrum = np.log1p(np.abs(np.fft.fftshift(fft)))
    return np.mean(magnitude_spectrum)
def analyze_metadata(image_path):
    try:
        exif_data = piexif.load(image_path)
        if piexif.ExifIFD.UserComment in exif_data["Exif"]:
            exif_dict = piexif.helper.UserComment.load(
                exif_data["Exif"][piexif.ExifIFD.UserComment]
            )
            if "Stable Diffusion" in exif_dict:
                return "AI tool detected in metadata"
        return "No AI tool detected in metadata"
    except piexif._exceptions.InvalidImageDataError:
        # This can happen if the image is a frame from a video
        return "No valid EXIF data"
    except Exception as e:
        # print(f"❌ Error analyzing metadata: {e}")
        return "Metadata analysis failed"
def analyze_color_distribution(image):
    np_image = np.array(image)
    hist_r, _ = np.histogram(np_image[:, :, 0], bins=256, range=(0, 256))
    hist_g, _ = np.histogram(np_image[:, :, 1], bins=256, range=(0, 256))
    hist_b, _ = np.histogram(np_image[:, :, 2], bins=256, range=(0, 256))
    return np.std(hist_r) + np.std(hist_g) + np.std(hist_b)
def detect_watermark(image):
    try:
        # Frames from video won't have 'exif' in info
        if "exif" not in image.info:
            return "No watermark detected"
        exif_data = piexif.load(image.info.get("exif", b""))
        if piexif.ExifIFD.UserComment in exif_data["Exif"]:
            watermark = piexif.helper.UserComment.load(
                exif_data["Exif"][piexif.ExifIFD.UserComment]
            )
            return watermark
        return "No watermark detected"
    except piexif._exceptions.InvalidImageDataError:
        return "No valid EXIF data"
    except Exception as e:
        # print(f"❌ Error detecting watermark: {e}")
        return "No watermark detected"

# --- *** MODIFIED classify_image *** ---
# Now it returns a dictionary (JSON) instead of a string
def classify_image(
    image, image_path, clip_model, clip_processor, vit_model, vit_processor
):
    try:
        noise_level = estimate_noise(image)
        edge_density = analyze_texture(image)
        pattern_score = detect_repeating_patterns(image)
        metadata_info = analyze_metadata(image_path)
        color_distribution = analyze_color_distribution(image)
        watermark_info = detect_watermark(image)

        clip_inputs = clip_processor(images=image, return_tensors="pt").to(
            accelerator.device
        )
        clip_outputs = (
            clip_model.get_image_features(**clip_inputs).detach().cpu().numpy()
        )
        clip_confidence = np.clip(
            np.interp(np.median(clip_outputs), [-0.3, 0.3], [0, 100]), 0, 100
        )

        vit_inputs = vit_processor(images=image, return_tensors="pt").to(
            accelerator.device
        )
        vit_outputs = (
            vit_model(**vit_inputs).last_hidden_state.mean(dim=1).detach().cpu().numpy()
        )
        vit_confidence = np.clip(
            np.interp(np.median(vit_outputs), [-0.3, 0.3], [0, 100]), 0, 100
        )

        # This is the AI-generated confidence
        combined_confidence = (clip_confidence + vit_confidence) / 2

        # ✅ --- CORRECTED LOGIC ---
        # We set a threshold. 50% is a good default.
        AI_THRESHOLD = 50.0 

        if combined_confidence > AI_THRESHOLD:
            classification = "AI-Generated"
        else:
            classification = "Real"
        
        # Return a dictionary
        return {
            "classification": classification,
            "aiConfidence": round(combined_confidence, 2),
            "humanConfidence": round(100 - combined_confidence, 2),
            "noiseLevel": round(noise_level, 2),
            "edgeDensity": round(edge_density, 2),
            "metadataInfo": metadata_info,
            "watermarkInfo": watermark_info,
        }

    except Exception as e:
        print(f"❌ Error classifying image: {e}")
        return {"error": str(e)}

@app.route("/")
def serve_home():
    """Serves the index.html file for the main page."""
    return send_file("index.html")

@app.route("/<path:path>")
def serve_static(path):
    """Serves any other file requested (like CSS or JS if you add them)."""
    try:
        return send_file(path)
    except:
        return send_file("index.html")

# --- NEW FLASK API ENDPOINT ---
@app.route("/analyze-image", methods=["POST"])
def handle_image_upload():
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save file temporarily
        # We use a unique name to avoid conflicts if multiple users analyze
        temp_path = f"temp_{int(time.time() * 1000)}.png"
        file.save(temp_path)
        
        # Load the image
        image = None
        try:
            image = load_image(temp_path)
        except Exception as e:
            print(f"Error loading temp file: {e}")
            try:
                os.remove(temp_path)
            except: pass
            return jsonify({"error": "Could not load image"}), 500
        
        if image is None:
            return jsonify({"error": "Could not load image"}), 500

        # Run analysis
        analysis_results = classify_image(
            image, temp_path, CLIP_MODEL, CLIP_PROCESSOR, VIT_MODEL, VIT_PROCESSOR
        )
        
        # Add processing time to the results
        analysis_results["processingTime"] = round(time.time() - start_time, 2)
        
        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file: {e}")

        return jsonify(analysis_results)

    return jsonify({"error": "Unknown error"}), 500


# --- RUN THE FLASK APP ---
if __name__ == "__main__":
    # We remove all the argparse and Gradio logic
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000) # Runs the web server