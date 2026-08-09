import os
import uuid
import numpy as np
from PIL import Image
from flask import Flask, request, render_template

from segmentation import run_all

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose an image first.")

    # Save the uploaded file with a unique name so repeated uploads don't clash
    unique_id = uuid.uuid4().hex[:8]
    upload_filename = f"{unique_id}_{file.filename}"
    upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
    file.save(upload_path)

    # Load it into a numpy array for segmentation.py
    img = Image.open(upload_path).convert("RGB")
    img_array = np.array(img)

    # Run all 4 algos
    results = run_all(img_array)

    # Save each output image, build a list for the template
    output_data = []
    for name, data in results.items():
        out_filename = f"{unique_id}_{name.replace(' ', '_')}.png"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)
        Image.fromarray(data["image"]).save(out_path)

        output_data.append({
            "name": name,
            "runtime": data["runtime"],
            "image_path": out_path,  # e.g. static/outputs/xxx.png
        })

    original_path = upload_path  # e.g. static/uploads/xxx.jpg

    return render_template(
        "results.html",
        original_image=original_path,
        results=output_data,
    )


if __name__ == "__main__":
    app.run(debug=True)