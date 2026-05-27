from flask import Flask, request

import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def classify(image_path):
    image = Image.open(image_path)
    classes = ["a photo of a fist",
               "a photo of a flat hand",
               "a photo of a victory sign"]

    inputs = processor(text=classes, images=image, return_tensors="pt",
                       padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        # Calculate probabilities (logits)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1)

    r, p, s = probs[0]
    if r >= 0.5: cl = "R"
    elif s >= 0.5: cl = "S"
    elif p >= 0.5 and s <= 0.35: cl = "P"
    else: cl = "S"

    return f"{cl} {image_path} {r:.2f}:{p:.2f}:{s:.2f}"

app = Flask(__name__)

@app.route("/bulk")
def bulk():
    print("new web request")
    s = "<html><body><pre>"
    p = Path("/Users/dave/raw/rock")
    for file in p.iterdir():
        s += classify(file) + "\n"
    p = Path("/Users/dave/raw/paper")
    for file in p.iterdir():
        s += classify(file) + "\n"
    p = Path("/Users/dave/raw/scissors")
    for file in p.iterdir():
        s += classify(file) + "\n"
    s += "</pre></body></html>"
    return s

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        file.save("/tmp/foo")
        return "<pre>"+classify("/tmp/foo")+"</pre>"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
