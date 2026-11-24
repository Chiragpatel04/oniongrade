# ============================================================================
# OnionGrade AI - Final-year project, HOSTED version for Render (free)
# SIH 2026 project: AI-Based Quality Assessment and Grading of Onions
#
# This file is app_hosted.py. It runs on the internet, so the app opens from
# any phone with a normal https address and can be installed as a mobile app.
#
# FILES IN THE GITHUB REPOSITORY (all in the top folder):
#   app_hosted.py       <- this file
#   requirements.txt    <- list of packages Render installs
#   .python-version     <- tells Render to use Python 3.12 (contains: 3.12)
#   onion_model.tflite  <- the AI model, made from onion_model.keras in Colab
#
# RENDER SETTINGS:
#   Build command:  pip install -r requirements.txt
#   Start command:  python app_hosted.py
#   Instance type:  Free
#
# IMPORTANT ABOUT THE FREE SERVER:
#   - It goes to sleep after about 15 minutes with no visitors. The first
#     open after a sleep takes about 1 minute. Open the address a few minutes
#     before showing it to someone.
#   - Photo grading and the heatmap are a little slower than on your laptop
#     (a few seconds) because the free server is small.
#   - The database lives inside the server. When the server sleeps, restarts
#     or you upload a new version, accounts and history are wiped. Just sign
#     up again. This is fine for a demo.
#
# INSTALL AS A MOBILE APP (PWA):
#   Open your https://....onrender.com address in Chrome on the phone, log in,
#   tap the 3-dot menu, then tap "Install app".
#
# WHAT THE APP DOES:
#   1. LOGIN: create an account (username + password, minimum 6 characters).
#      Passwords are stored hashed, never as plain text. Grading needs login.
#   2. MY HISTORY: every grading is saved for the user with date, grade,
#      confidence and small photos.
#   3. HEATMAP ("why this grade"): after grading you see the photo and a
#      heatmap side by side. The app covers small parts of the photo one at a
#      time and checks how much the AI's confidence drops. Warm colours show
#      the parts that matter most to the decision (occlusion sensitivity).
#      The laptop version uses Grad-CAM; this server version uses occlusion
#      because the light model file cannot do Grad-CAM.
#   4. Grading is the same as the laptop version: 224x224 photo, MobileNetV2
#      scaling, healthy = 0 / unhealthy = 1, threshold 0.5.
#
# Dependencies: flask, ai-edge-litert (runs the .tflite model), pillow, numpy.
# No TensorFlow is needed on the server.
# ============================================================================

from flask import (Flask, request, render_template_string, Response, jsonify,
                   redirect, url_for, session)
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import sqlite3
import secrets
import re
import datetime
import threading
import time
from PIL import Image, ImageFilter
import numpy as np
import io
import os

# Light model runner (no TensorFlow needed). tflite_runtime is the old name.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

# ----------------------------------------------------------------------------
# Settings - these EXACTLY match how the model was trained in the Colab
# notebook (Onion_Grading_Step1). Do not change these three lines.
# ----------------------------------------------------------------------------
IMG_SIZE = 224                                  # model was trained on 224x224 photos
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onion_model.tflite")  # model file, same folder
CLASS_NAMES = ["healthy", "unhealthy"]          # 0 = healthy, 1 = unhealthy (as in training)

# Load the AI model once, when the website starts
print("Loading the onion model, please wait...")


class TfliteModel:
    """Small wrapper so the grading code can call model.predict(batch)."""

    def __init__(self, path):
        self.interpreter = Interpreter(model_path=path, num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.lock = threading.Lock()   # the interpreter handles one photo at a time

    def predict(self, batch, verbose=0):
        with self.lock:
            self.interpreter.set_tensor(self.input_index, np.asarray(batch, dtype=np.float32))
            self.interpreter.invoke()
            return np.array(self.interpreter.get_tensor(self.output_index))


model = TfliteModel(MODEL_FILE)
print("Model loaded. Website is ready!")

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Single-file responsive UI. Model and route behavior are unchanged.
# ----------------------------------------------------------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#204f3b">
<link rel="manifest" href="/manifest.json">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="OnionGrade">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<link rel="icon" type="image/png" sizes="192x192" href="/icon-192.png">
<link rel="apple-touch-icon" href="/icon-192.png">
<title>OnionGrade AI | Onion Quality Assessment</title>
<style>
:root{--green:#204f3b;--green-dark:#153b2b;--ink:#20372c;--muted:#627369;--paper:#f7f8f2;--line:#dfe5dc;--purple:#76516e;--lavender:#f1e9f0;--red:#a53942;--soft-red:#fcf0ee}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.6}a{color:inherit}button,input{font:inherit}button,a,label,input{-webkit-tap-highlight-color:transparent}a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid var(--purple);outline-offset:5px}.wrap{max-width:1180px;padding:0 36px;margin:auto}.nav{display:flex;align-items:center;justify-content:space-between;gap:24px;min-height:100px;border-bottom:1px solid var(--line)}.brand{text-decoration:none;display:flex;align-items:center;gap:12px;font-size:21px;font-weight:800;letter-spacing:-.7px}.logo{width:43px;height:43px;display:grid;place-items:center;background:var(--green);color:white;border-radius:14px}.logo svg{width:30px;height:30px}.brand b{color:var(--purple)}.nav-links{display:flex;align-items:center;gap:28px;font-size:13px;font-weight:650}.nav-links a{text-decoration:none}.nav-links a:hover{color:var(--purple)}.nav-pill{padding:9px 15px;background:#eaf0e5;border-radius:30px;color:var(--green);font-size:11px;letter-spacing:.8px;text-transform:uppercase}.hero{display:grid;grid-template-columns:1fr 1fr;gap:70px;padding:64px 0 60px;align-items:start}.eyebrow{font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:var(--purple);margin:0 0 18px;display:flex;align-items:center;gap:10px}.eyebrow:before{content:"";height:1px;width:26px;background:var(--purple)}h1{font-size:clamp(42px,4.5vw,60px);line-height:1.1;letter-spacing:-2.5px;margin:0 0 22px;font-weight:750}h1 span{color:var(--purple)}.project-title{font-size:17px;line-height:1.6;font-weight:600;max-width:420px;margin:0 0 14px}.intro{color:var(--muted);font-size:15px;max-width:420px;margin:0}.steps{margin:30px 0 0;padding:0;list-style:none;display:flex;gap:20px;flex-wrap:wrap}.steps li{display:flex;gap:8px;align-items:center;font-size:12px;color:var(--muted)}.steps span{border:1px solid #d2dbcd;width:25px;height:25px;border-radius:50%;display:grid;place-items:center;font-size:11px;color:var(--green);font-weight:700}.metric-row{margin-top:38px;padding-top:25px;border-top:1px solid var(--line);display:flex;gap:36px}.metric strong{display:block;font-size:31px;line-height:1.2;letter-spacing:-1px;font-weight:750}.metric small{color:var(--muted);font-size:11px}.model-note{font-size:11px;color:var(--muted);margin:14px 0 0;max-width:410px}.grader{background:white;border:1px solid var(--line);border-radius:24px;box-shadow:0 14px 45px #20372c0a;overflow:hidden}.card-heading{padding:23px 26px 18px;display:flex;justify-content:space-between;align-items:center;gap:12px}.card-heading h2{font-size:19px;margin:0;letter-spacing:-.5px}.tag{font-size:10px;color:var(--purple);background:var(--lavender);font-weight:750;border-radius:6px;padding:5px 9px;letter-spacing:.7px;white-space:nowrap}.form-body{padding:0 26px 23px}.upload{position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:190px;text-align:center;border:1.5px dashed #b6c6b0;background:#f7faf4;border-radius:15px;padding:24px 16px;cursor:pointer;transition:background .2s,border-color .2s}.upload:hover,.upload:focus-within{background:#edf4e8;border-color:var(--green)}.upload input{position:absolute;inset:0;opacity:0;width:100%;height:100%;cursor:pointer}.camera{width:48px;height:48px;display:grid;place-items:center;background:#e6eedf;border-radius:15px;margin-bottom:13px;color:var(--green)}.camera svg{width:25px;height:25px}.upload strong{font-size:15px}.upload small{font-size:12px;color:var(--muted);margin-top:5px}.file-status{font-size:12px;color:var(--purple);margin:10px 0 0;overflow-wrap:anywhere}.submit{width:100%;border:0;border-radius:11px;background:var(--green);color:white;font-size:14px;font-weight:700;padding:15px 20px;margin-top:17px;display:flex;align-items:center;justify-content:center;gap:15px;cursor:pointer}.submit:hover{background:var(--green-dark)}.submit:disabled{opacity:.75;cursor:wait}.privacy{font-size:10px;color:var(--muted);text-align:center;margin:10px 0 0}.report{padding:24px 26px;background:#f1f6ed;border-top:1px solid var(--line)}.report.bad{background:var(--soft-red);--result-color:var(--red)}.report.good{--result-color:var(--green)}.report-label{font-size:10px;letter-spacing:1.6px;font-weight:800;text-transform:uppercase;color:var(--result-color);margin:0 0 15px}.result-top{display:flex;align-items:center;gap:19px}.preview{width:102px;height:112px;object-fit:contain;background:white;border-radius:12px;border:1px solid #20372c16;flex-shrink:0}.result-copy{flex:1;min-width:0}.grade{font-size:25px;line-height:1.25;font-weight:750;color:var(--result-color);margin:0 0 8px;letter-spacing:-.8px}.result-description{font-size:12px;color:var(--muted);margin:0}.confidence-label{display:flex;justify-content:space-between;font-size:12px;gap:8px;margin:20px 0 7px}.confidence-label strong{color:var(--result-color)}.bar{height:7px;background:#20372c13;border-radius:8px;overflow:hidden}.bar span{display:block;height:100%;background:var(--result-color);border-radius:8px}.report-note{font-size:10px;color:var(--muted);margin:13px 0 0}.empty-report{border-top:1px solid var(--line);padding:17px 26px;display:flex;gap:10px;align-items:center;background:#fafbf8;color:var(--muted);font-size:11px}.empty-dot{width:6px;height:6px;border-radius:50%;background:#9da995;flex-shrink:0}.error{margin:0 26px 23px;padding:12px 15px;background:var(--soft-red);border:1px solid #f1d2d0;border-radius:10px;color:var(--red);font-size:13px}.about{border-top:1px solid var(--line);padding:43px 0 48px}.section-head{display:flex;justify-content:space-between;gap:30px;align-items:start;margin-bottom:25px}.section-head h2{font-size:27px;letter-spacing:-1px;margin:0}.section-head p{max-width:440px;margin:0;color:var(--muted);font-size:13px}.features{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.feature{background:#fff;border:1px solid var(--line);border-radius:15px;padding:23px}.feature-number{color:var(--purple);font-size:10px;font-weight:800;letter-spacing:1px}.feature h3{font-size:15px;letter-spacing:-.3px;margin:10px 0 8px}.feature p{font-size:12px;line-height:1.8;color:var(--muted);margin:0}.limits{font-size:11px;color:var(--muted);margin:19px 0 0;padding-left:13px;border-left:2px solid #c8b5c4}.footer{background:var(--green-dark);color:#ecf2e9}.footer-inner{padding:24px 0;display:flex;justify-content:space-between;gap:20px;align-items:center}.footer strong{font-size:13px;font-weight:600}.footer p{font-size:11px;margin:4px 0 0;color:#c0d0c2}.footer-guide{font-size:11px;color:#d0ddd1;text-align:right}
@media(max-width:900px){.hero{gap:35px}h1{font-size:45px}.wrap{padding:0 25px}.nav-links{gap:18px}.nav-pill{display:none}.metric-row{gap:23px}.metric strong{font-size:28px}.grade{font-size:22px}}
@media(max-width:680px){.wrap{padding:0 20px}.nav{min-height:78px;gap:12px}.brand{font-size:18px;gap:9px}.logo{width:35px;height:35px;border-radius:11px}.logo svg{width:25px}.nav-links{gap:15px;font-size:11px}.hero{grid-template-columns:1fr;gap:29px;padding:34px 0 36px}.eyebrow{font-size:9px;letter-spacing:1.6px;margin-bottom:15px}h1{font-size:43px;letter-spacing:-1.8px;max-width:400px;margin-bottom:17px}.project-title{font-size:15px;margin-bottom:11px;max-width:360px}.intro{font-size:13px}.steps{margin-top:23px;gap:17px}.steps li{font-size:11px}.metric-row{margin-top:24px;padding-top:18px;gap:35px}.metric strong{font-size:27px}.metric small{font-size:10px}.model-note{margin-top:10px;font-size:10px}.card-heading{padding:21px 21px 17px}.form-body{padding:0 21px 21px}.upload{min-height:180px}.report{padding:21px}.error{margin:0 21px 21px}.empty-report{padding:16px 21px}.section-head{display:block;margin-bottom:20px}.section-head h2{font-size:25px;margin-bottom:10px}.features{grid-template-columns:1fr;gap:12px}.feature{padding:20px}.feature h3{margin:6px 0}.about{padding:32px 0}.footer-inner{display:block;padding:24px 0}.footer-guide{text-align:left;margin-top:13px}.privacy{font-size:10px}}
.nav-links .who{color:var(--muted);font-size:12px}.auth{max-width:440px;margin:48px auto 60px}.auth .grader{padding:8px 0 6px}.auth form{padding:0 26px 20px}.auth label.f{display:block;font-size:12px;font-weight:700;margin:14px 0 6px}.auth input[type=text],.auth input[type=password]{width:100%;padding:13px 14px;border:1px solid #cfd9ca;border-radius:10px;background:#fbfcf8;color:var(--ink);font-size:15px}.auth input:focus{outline:3px solid var(--lavender);border-color:var(--purple)}.auth .alt{padding:0 26px 22px;font-size:13px;color:var(--muted)}.auth .alt a{color:var(--purple);font-weight:700}.hist-head{display:flex;justify-content:space-between;align-items:end;gap:20px;flex-wrap:wrap;padding:42px 0 22px}.hist-head h1{font-size:34px;margin:0;letter-spacing:-1.2px}.hist-stats{display:flex;gap:26px;font-size:12px;color:var(--muted)}.hist-stats strong{display:block;font-size:24px;color:var(--ink);line-height:1.2}.hist-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;padding-bottom:50px}.hist-item{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px}.hist-imgs{display:flex;gap:8px}.hist-imgs img{width:50%;aspect-ratio:1/1;object-fit:cover;border-radius:10px;border:1px solid var(--line)}.hist-meta{display:flex;justify-content:space-between;align-items:center;margin-top:12px;gap:8px}.badge{font-size:12px;font-weight:800;padding:4px 11px;border-radius:20px}.badge.good{background:#e6eedf;color:var(--green)}.badge.bad{background:var(--soft-red);color:var(--red)}.hist-time{font-size:11px;color:var(--muted);margin:6px 0 0}.hist-empty{border:1px dashed #b6c6b0;border-radius:16px;padding:40px 20px;text-align:center;color:var(--muted);margin-bottom:50px}.cam{margin-top:22px;padding-top:18px;border-top:1px solid #20372c16}.cam-title{font-size:10px;letter-spacing:1.6px;font-weight:800;text-transform:uppercase;color:var(--purple);margin:0 0 12px}.cam-pair{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:0}.cam-pair figure{margin:0}.cam-pair img{width:100%;border-radius:12px;border:1px solid #20372c16;background:#fff;display:block}.cam-pair figcaption{font-size:11px;color:var(--muted);text-align:center;margin-top:6px}.cam-note{font-size:12px;color:var(--ink);margin:12px 0 0}.saved-note{font-size:10px;color:var(--muted);margin:8px 0 0}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
<header class="nav">
<a href="#" class="brand" aria-label="OnionGrade AI home"><span class="logo" aria-hidden="true"><svg viewBox="0 0 40 40" fill="none"><path d="M20 6c0 8-13 10-13 20a13 10 0 0 0 26 0C33 16 20 14 20 6Z" stroke="currentColor" stroke-width="1.7"/><path d="M20 10c-5 7-7 11-7 17s14 7 14 0-3-11-7-17Zm0 0v24M17 4l3 4 4-5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg></span><span>OnionGrade <b>AI</b></span></a>
<nav class="nav-links" aria-label="Main navigation"><a href="#grade">Grade onion</a><a href="#model">The model</a>{% if username %}<a href="/history">My History</a><a href="/logout">Logout <span class="who">({{ username }})</span></a>{% else %}<span class="nav-pill">Engineering project</span>{% endif %}</nav>
</header>
<main>
<section class="hero" aria-labelledby="hero-title">
<div class="hero-copy">
<p class="eyebrow">From photo to quality insight</p>
<h1 id="hero-title">A closer look.<br>A <span>smarter grade.</span></h1>
<p class="project-title">AI-Based Quality Assessment and Grading of Onions</p>
<p class="intro">Turn a photo of an onion into a clear quality prediction, powered by deep learning.</p>
<ol class="steps"><li><span>01</span> Take a photo</li><li><span>02</span> Run the model</li><li><span>03</span> View the grade</li></ol>
<div class="metric-row" aria-label="Model test results"><div class="metric"><strong>95.8<span style="font-size:20px">%</span></strong><small>Test accuracy</small></div><div class="metric"><strong>1,203</strong><small>Unseen test photos</small></div><div class="metric"><strong>2</strong><small>Quality classes</small></div></div>
<p class="model-note">Evaluation results from the trained model. Real-world results can vary with lighting, framing and onion condition.</p>
</div>
<section id="grade" class="grader" aria-labelledby="grade-title">
<div class="card-heading"><h2 id="grade-title">Grade your onion</h2><span class="tag">AI ASSESSMENT</span></div>
<form method="POST" action="/" enctype="multipart/form-data" id="grade-form">
<div class="form-body">
<label class="upload" for="photo"><span class="camera" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M8 6l1.5-2h5L16 6h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2Z"/><circle cx="12" cy="12.5" r="3.5"/></svg></span><strong>Tap to take a photo</strong><small>or choose an image from your device</small><small>One onion, clear light, simple background</small><input id="photo" type="file" name="photo" accept="image/*" capture="environment" aria-label="Take or upload a photo of one onion" required></label>
<p id="file-status" class="file-status" aria-live="polite" hidden></p>
<button type="submit" class="submit" id="submit-button"><span id="button-text">Assess onion quality</span><span aria-hidden="true">&#8599;</span></button>
<p class="privacy">Photo &#8594; MobileNetV2 &#8594; Quality prediction</p>
</div>
</form>
{% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}
{% if result %}
<section class="report {{ css_class }}" aria-labelledby="result-title" aria-live="polite">
<p class="report-label">Assessment complete</p>
<div class="result-top"><img class="preview" src="data:image/jpeg;base64,{{ photo_b64 }}" alt="Uploaded onion used for this assessment"><div class="result-copy"><h3 class="grade" id="result-title">{{ result }}</h3><p class="result-description">{% if css_class == 'good' %}The model predicts a healthy onion.{% else %}The model predicts an unhealthy onion.{% endif %}</p></div></div>
<div class="confidence-label"><span>Model confidence</span><strong>{{ confidence }}%</strong></div><div class="bar" role="progressbar" aria-label="Prediction confidence" aria-valuenow="{{ confidence }}" aria-valuemin="0" aria-valuemax="100"><span style="width: {{ confidence }}%"></span></div>
{% if heat_b64 %}<div class="cam"><p class="cam-title">Why this grade? Heatmap</p><figure class="cam-pair"><figure><img src="data:image/jpeg;base64,{{ base_b64 }}" alt="Onion photo as the model sees it"><figcaption>Photo</figcaption></figure><figure><img src="data:image/jpeg;base64,{{ heat_b64 }}" alt="Heatmap over the onion photo"><figcaption>Heatmap</figcaption></figure></figure><p class="cam-note">Highlighted areas show where the AI looked to decide. Covering the warm parts changes the result most.</p></div>{% endif %}
{% if saved %}<p class="saved-note">Saved to <a href="/history">My History</a>.</p>{% endif %}
<p class="report-note">Confidence is the model's score for its predicted class, not a guarantee of quality or food safety. Take another photo to assess again.</p>
</section>
{% else %}
<div class="empty-report"><span class="empty-dot" aria-hidden="true"></span>Your grade, confidence score and photo will appear here.</div>
{% endif %}
</section>
</section>
<section id="model" class="about" aria-labelledby="model-title">
<div class="section-head"><div><p class="eyebrow" style="margin-bottom:9px">Behind the prediction</p><h2 id="model-title">Small image. Deep learning.</h2></div><p>A two-class image classifier that learns visual patterns to distinguish healthy and unhealthy onions.</p></div>
<div class="features">
<article class="feature"><span class="feature-number">01 / ARCHITECTURE</span><h3>Built on MobileNetV2</h3><p>A convolutional neural network processes each photo at 224 &#215; 224 pixels using the same preprocessing used during training.</p></article>
<article class="feature"><span class="feature-number">02 / EVALUATION</span><h3>Tested on unseen photos</h3><p>The trained model achieved 95.8% test accuracy across 1,203 unseen photos. This is a test-set result, not a promise for every photo.</p></article>
<article class="feature"><span class="feature-number">03 / OUTPUT</span><h3>Two classes. A clear score.</h3><p>Each assessment predicts Healthy or Unhealthy and displays a confidence score for that class alongside your photo.</p></article>
</div>
<p class="limits">Project scope: image-based classification only. This version does not identify individual defects, assign commercial grades, or certify that an onion is safe to eat.</p>
</section>
</main>
</div>
<footer class="footer"><div class="wrap"><div class="footer-inner"><div><strong>Created by Chirag Patel</strong><p>D.N.P College of Engineering (DBATU)</p></div><div class="footer-guide">Guide: Prof. V.S. Mahajan<br>AI-Based Quality Assessment and Grading of Onions</div></div></div></footer>
<script>
// Progressive enhancement only. Uploading and grading also work without JavaScript.
const photo = document.getElementById('photo');
const status = document.getElementById('file-status');
photo.addEventListener('change', function () {
    status.hidden = !photo.files.length;
    status.textContent = photo.files.length ? 'Ready: ' + photo.files[0].name : '';
});
document.getElementById('grade-form').addEventListener('submit', function () {
    document.getElementById('submit-button').disabled = true;
    document.getElementById('button-text').textContent = 'Assessing photo...';
});
</script>
<script>if("serviceWorker" in navigator){window.addEventListener("load",function(){navigator.serviceWorker.register("/sw.js").catch(function(){});});}</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------
# Database (SQLite). The file is created automatically next to app.py.
# ----------------------------------------------------------------------------
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oniongrade.db")


def db():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS gradings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            grade TEXT NOT NULL,
            confidence REAL NOT NULL,
            photo_thumb TEXT NOT NULL,
            heat_thumb TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    row = con.execute("SELECT value FROM settings WHERE key = 'secret_key'").fetchone()
    if row is None:
        con.execute("INSERT INTO settings (key, value) VALUES ('secret_key', ?)",
                    (secrets.token_hex(32),))
        con.commit()
        row = con.execute("SELECT value FROM settings WHERE key = 'secret_key'").fetchone()
    secret = row["value"]
    con.commit()
    con.close()
    return secret


# The secret key signs the login cookie. It is stored in the database so
# people stay logged in even after the app is restarted.
app.secret_key = init_db()
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(days=30)


def current_user():
    """Return the logged-in user's row, or None."""
    uid = session.get("user_id")
    if not uid:
        return None
    con = db()
    user = con.execute("SELECT id, username FROM users WHERE id = ?", (uid,)).fetchone()
    con.close()
    if user is None:
        session.clear()
    return user


@app.context_processor
def inject_user():
    user = current_user()
    return {"username": user["username"] if user else None}


# ----------------------------------------------------------------------------
# Login / signup / history pages. They reuse the CSS of the main page.
# ----------------------------------------------------------------------------
PAGE_CSS = PAGE.split("<style>")[1].split("</style>")[0]

_SHELL_TOP = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#204f3b">
<link rel="manifest" href="/manifest.json">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="OnionGrade">
<link rel="icon" type="image/png" sizes="192x192" href="/icon-192.png">
<link rel="apple-touch-icon" href="/icon-192.png">
<title>{{ title }} | OnionGrade AI</title>
<style>{{ css|safe }}</style>
</head>
<body>
<div class="wrap">
<header class="nav">
<a href="/" class="brand" aria-label="OnionGrade AI home"><span class="logo" aria-hidden="true"><svg viewBox="0 0 40 40" fill="none"><path d="M20 6c0 8-13 10-13 20a13 10 0 0 0 26 0C33 16 20 14 20 6Z" stroke="currentColor" stroke-width="1.7"/><path d="M20 10c-5 7-7 11-7 17s14 7 14 0-3-11-7-17Zm0 0v24M17 4l3 4 4-5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg></span><span>OnionGrade <b>AI</b></span></a>
<nav class="nav-links" aria-label="Main navigation">{% if username %}<a href="/">Grade onion</a><a href="/history">My History</a><a href="/logout">Logout <span class="who">({{ username }})</span></a>{% else %}<span class="nav-pill">Engineering project</span>{% endif %}</nav>
</header>
<main>
"""
_SHELL_BOTTOM = """
</main>
</div>
<footer class="footer"><div class="wrap"><div class="footer-inner"><div><strong>Created by Chirag Patel</strong><p>D.N.P College of Engineering (DBATU)</p></div><div class="footer-guide">Guide: Prof. V.S. Mahajan<br>AI-Based Quality Assessment and Grading of Onions</div></div></div></footer>
<script>if("serviceWorker" in navigator){window.addEventListener("load",function(){navigator.serviceWorker.register("/sw.js").catch(function(){});});}</script>
</body>
</html>
"""

AUTH_PAGE = _SHELL_TOP + """
<div class="auth">
<section class="grader" aria-labelledby="auth-title">
<div class="card-heading"><h2 id="auth-title">{{ heading }}</h2><span class="tag">{{ tag }}</span></div>
{% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}
<form method="POST" action="{{ action }}" autocomplete="on">
<label class="f" for="username">Username</label>
<input type="text" id="username" name="username" value="{{ form_username }}" autocomplete="username" autocapitalize="none" required>
<label class="f" for="password">Password</label>
<input type="password" id="password" name="password" autocomplete="{{ 'new-password' if signup else 'current-password' }}" required>
{% if signup %}<label class="f" for="confirm">Confirm password</label>
<input type="password" id="confirm" name="confirm" autocomplete="new-password" required>{% endif %}
<button type="submit" class="submit"><span>{{ button }}</span><span aria-hidden="true">&#8599;</span></button>
</form>
<p class="alt">{{ alt_text }} <a href="{{ alt_link }}">{{ alt_label }}</a></p>
</section>
</div>
""" + _SHELL_BOTTOM

HISTORY_PAGE = _SHELL_TOP + """
<div class="hist-head">
<div><p class="eyebrow">Saved gradings</p><h1>My History</h1></div>
<div class="hist-stats"><div><strong>{{ total }}</strong>Total</div><div><strong>{{ healthy }}</strong>Healthy</div><div><strong>{{ unhealthy }}</strong>Unhealthy</div></div>
</div>
{% if rows %}
<div class="hist-list">
{% for r in rows %}
<article class="hist-item">
<div class="hist-imgs"><img src="data:image/jpeg;base64,{{ r.photo_thumb }}" alt="Onion photo">{% if r.heat_thumb %}<img src="data:image/jpeg;base64,{{ r.heat_thumb }}" alt="Heatmap">{% endif %}</div>
<div class="hist-meta"><span class="badge {{ 'good' if r.grade == 'Healthy' else 'bad' }}">{{ r.grade }}</span><strong>{{ r.confidence }}%</strong></div>
<p class="hist-time">{{ r.when }}</p>
</article>
{% endfor %}
</div>
{% else %}
<div class="hist-empty">No gradings yet. Go to <a href="/">Grade onion</a> and assess your first onion.</div>
{% endif %}
""" + _SHELL_BOTTOM


def _auth(**kw):
    base = dict(css=PAGE_CSS, error=None, form_username="", signup=False)
    base.update(kw)
    return render_template_string(AUTH_PAGE, **base)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user():
        return redirect(url_for("home"))
    page = dict(title="Create account", heading="Create account", tag="NEW USER",
                action="/signup", button="Sign up", signup=True,
                alt_text="Already have an account?", alt_link="/login", alt_label="Log in")
    if request.method == "GET":
        return _auth(**page)
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""
    page["form_username"] = username
    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        return _auth(error="Username must be 3 to 30 letters, numbers or underscores.", **page)
    if len(password) < 6:
        return _auth(error="Password must be at least 6 characters.", **page)
    if password != confirm:
        return _auth(error="The two passwords do not match.", **page)
    con = db()
    try:
        cur = con.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.datetime.now().isoformat()))
        con.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        con.close()
        return _auth(error="That username is already taken. Please choose another.", **page)
    con.close()
    session.clear()
    session["user_id"] = user_id
    session.permanent = True
    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("home"))
    page = dict(title="Log in", heading="Log in", tag="WELCOME BACK",
                action="/login", button="Log in",
                alt_text="New here?", alt_link="/signup", alt_label="Create account")
    if request.method == "GET":
        return _auth(**page)
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    page["form_username"] = username
    con = db()
    user = con.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    con.close()
    if user is None or not check_password_hash(user["password_hash"], password):
        return _auth(error="Wrong username or password.", **page)
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = True
    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/history")
def history():
    user = current_user()
    if not user:
        return redirect(url_for("login"))
    con = db()
    found = con.execute(
        "SELECT created_at, grade, confidence, photo_thumb, heat_thumb FROM gradings "
        "WHERE user_id = ? ORDER BY id DESC", (user["id"],)).fetchall()
    con.close()
    rows = []
    for r in found:
        try:
            when = datetime.datetime.fromisoformat(r["created_at"]).strftime("%d %b %Y, %I:%M %p")
        except ValueError:
            when = r["created_at"]
        rows.append(dict(when=when, grade=r["grade"], confidence=r["confidence"],
                         photo_thumb=r["photo_thumb"], heat_thumb=r["heat_thumb"]))
    healthy = sum(1 for r in rows if r["grade"] == "Healthy")
    return render_template_string(HISTORY_PAGE, css=PAGE_CSS, title="My History", rows=rows,
                                  total=len(rows), healthy=healthy, unhealthy=len(rows) - healthy)


# ----------------------------------------------------------------------------
# Heatmap by occlusion (the light model cannot do Grad-CAM):
# 1. Cover one square patch of the photo with neutral grey.
# 2. Ask the model again. If its score for the predicted answer drops a lot,
#    that patch was important.
# 3. Slide the patch over the whole photo, average the drops, scale to 0..1.
# If anything goes wrong here, grading still works and just skips the heatmap.
# ----------------------------------------------------------------------------
OCC_PATCH = 64        # patch size in pixels (on the 224x224 model photo)
OCC_STRIDE = 32       # step between patches -> 6 x 6 = 36 extra checks
OCC_MAX_SECONDS = 40  # give up on the heatmap if the server is too slow


def compute_heatmap(batch, unhealthy):
    """Return a 2D numpy array (values 0 to 1), or None if it cannot be made."""
    try:
        started = time.time()

        def score(x):
            p = float(model.predict(x, verbose=0)[0][0])
            return p if unhealthy else 1.0 - p      # score of the predicted class

        base_score = score(batch)
        total = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        count = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for top in range(0, IMG_SIZE - OCC_PATCH + 1, OCC_STRIDE):
            for left in range(0, IMG_SIZE - OCC_PATCH + 1, OCC_STRIDE):
                covered = batch.copy()
                covered[:, top:top + OCC_PATCH, left:left + OCC_PATCH, :] = 0.0   # 0 = grey
                drop = max(base_score - score(covered), 0.0)
                total[top:top + OCC_PATCH, left:left + OCC_PATCH] += drop
                count[top:top + OCC_PATCH, left:left + OCC_PATCH] += 1.0
                if time.time() - started > OCC_MAX_SECONDS:
                    return None
        heat = total / np.maximum(count, 1.0)
        peak = float(heat.max())
        if peak <= 1e-9:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        heat = heat / peak
        # soften the blocky edges a little
        smooth = Image.fromarray((heat * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10))
        heat = np.asarray(smooth, dtype=np.float32)
        return heat / max(float(heat.max()), 1e-9)
    except Exception as err:
        print("Heatmap skipped:", err)
        return None


def jet_colors(v):
    """Turn values 0..1 into RGB colours (blue -> green -> yellow -> red)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.float32)


def fit_image(img, max_side):
    img = img.copy()
    img.thumbnail((max_side, max_side))
    return img


def jpeg_b64(img, quality=82):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_heatmap_images(image_bytes, cam):
    """Return (photo_b64, heatmap_b64, photo_thumb_b64, heat_thumb_b64)."""
    base = fit_image(Image.open(io.BytesIO(image_bytes)).convert("RGB"), 480)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
    v = np.asarray(cam_img, dtype=np.float32) / 255.0
    mix = 0.6 * np.asarray(base, dtype=np.float32) + 0.4 * jet_colors(v)
    heat = Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8))
    return (jpeg_b64(base), jpeg_b64(heat),
            jpeg_b64(fit_image(base, 160), 70), jpeg_b64(fit_image(heat, 160), 70))


def prepare_image(image_bytes):
    """Take the uploaded photo and prepare it for the model.

    This must match the training notebook exactly:
    - resize to 224x224
    - scale pixels with MobileNetV2's preprocess_input (range -1 to 1)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")   # open photo, force 3 color channels
    img = img.resize((IMG_SIZE, IMG_SIZE))                     # resize to 224x224
    arr = np.asarray(img, dtype=np.float32)                    # to numbers
    arr = arr / 127.5 - 1.0                                    # same as MobileNetV2 preprocess_input (range -1 to 1)
    return np.expand_dims(arr, axis=0)                         # model expects a batch of photos


@app.route("/", methods=["GET", "POST"])
def home():
    # You must be logged in to use the grader
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    # First visit (GET): just show the empty page
    if request.method == "GET":
        return render_template_string(PAGE)

    # Form submitted (POST): a photo was uploaded
    file = request.files.get("photo")
    if not file or file.filename == "":
        return render_template_string(PAGE, error="No photo received. Please try again.")

    image_bytes = file.read()

    try:
        batch = prepare_image(image_bytes)
    except Exception:
        return render_template_string(PAGE,
            error="That file does not look like a photo. Please upload a JPG or PNG image.")

    # The model outputs one number: the probability that the onion is UNHEALTHY
    p_unhealthy = float(model.predict(batch, verbose=0)[0][0])

    if p_unhealthy >= 0.5:
        grade = "Unhealthy"
        css_class = "bad"
        confidence = p_unhealthy * 100          # how sure the model is it is unhealthy
    else:
        grade = "Healthy"
        css_class = "good"
        confidence = (1 - p_unhealthy) * 100    # how sure the model is it is healthy

    # Show the uploaded photo back on the result page
    import base64
    photo_b64 = base64.b64encode(image_bytes).decode("ascii")

    # Heatmap (optional: grading still works if this part fails)
    base_b64 = heat_b64 = None
    saved = False
    try:
        cam = compute_heatmap(batch, p_unhealthy >= 0.5)
        if cam is not None:
            base_b64, heat_b64, photo_thumb, heat_thumb = make_heatmap_images(image_bytes, cam)
        else:
            photo_thumb = jpeg_b64(fit_image(Image.open(io.BytesIO(image_bytes)).convert("RGB"), 160), 70)
            heat_thumb = None
        # Save this grading in the user's history (small thumbnails only)
        con = db()
        con.execute(
            "INSERT INTO gradings (user_id, created_at, grade, confidence, photo_thumb, heat_thumb) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user["id"], datetime.datetime.now().isoformat(), grade,
             round(confidence, 1), photo_thumb, heat_thumb))
        con.commit()
        con.close()
        saved = True
    except Exception as err:
        print("History/heatmap step skipped:", err)

    return render_template_string(PAGE,
        result="Grade: " + grade,
        css_class=css_class,
        confidence=round(confidence, 1),
        photo_b64=photo_b64,
        base_b64=base_b64,
        heat_b64=heat_b64,
        saved=saved)


# ----------------------------------------------------------------------------
# PWA (installable app) parts. Icons are embedded so no extra files are needed.
# ----------------------------------------------------------------------------
ICON_192_B64 = "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"
ICON_512_B64 = "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"

SERVICE_WORKER = """
const CACHE = "oniongrade-shell-v2";
self.addEventListener("install", function (e) { self.skipWaiting(); });
self.addEventListener("activate", function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) { return k !== CACHE; })
      .map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
self.addEventListener("fetch", function (e) {
  var url = new URL(e.request.url);
  // Only plain page loads (GET /) are cached, as an offline fallback.
  // Login, grading (POST), history and everything else go to the network.
  if (e.request.method === "GET" && url.origin === location.origin && url.pathname === "/") {
    e.respondWith(
      fetch(e.request).then(function (r) {
        if (r.ok && !r.redirected) {
          var copy = r.clone();
          caches.open(CACHE).then(function (c) { c.put("/", copy); });
        }
        return r;
      }).catch(function () { return caches.match("/"); })
    );
  }
});
"""


@app.route("/manifest.json")
def manifest():
    data = {
        "name": "OnionGrade AI",
        "short_name": "OnionGrade",
        "description": "AI-based quality assessment and grading of onions",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "orientation": "portrait",
        "background_color": "#f7f8f2",
        "theme_color": "#204f3b",
        "icons": [
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
        ],
    }
    return Response(jsonify(data).get_data(), mimetype="application/manifest+json")


@app.route("/sw.js")
def service_worker():
    resp = Response(SERVICE_WORKER, mimetype="application/javascript")
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/icon-192.png")
def icon_192():
    return Response(base64.b64decode(ICON_192_B64), mimetype="image/png")


@app.route("/icon-512.png")
def icon_512():
    return Response(base64.b64decode(ICON_512_B64), mimetype="image/png")


if __name__ == "__main__":
    # Hosted: listen on all addresses and use the port Render gives us (default 10000).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")), debug=False)
