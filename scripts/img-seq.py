import os
import random
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, render_template_string, abort

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CAM1_DIR = BASE_DIR / 'assets' / 'visual' / 'cam1'
CAM2_DIR = BASE_DIR / 'assets' / 'visual' / 'cam2'

# Supported extensions
EXTS = {'.png', '.jpg', '.jpeg', '.gif'}

# Number of layers before clearing canvas
CLEAR_AFTER = 100

# HTML template with fullscreen button and canvas overlay
TEMPLATE = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Image Sequencer</title>
  <style>
    html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background-color: #000; }}
    .container {{ display: flex; height: 100vh; }}
    .panel {{ flex: 1; position: relative; background-color: #000; }}
    canvas {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }}
    #fsBtn {{ position: fixed; top: 10px; right: 10px; z-index: 10000;
             background: rgba(255,255,255,0.3); color: #fff; border: none;
             padding: 10px 14px; font-size: 16px; cursor: pointer; border-radius: 4px; }}
  </style>
</head>
<body>
  <button id="fsBtn">⛶ Fullscreen</button>
  <div class="container">
    <div class="panel"><canvas id="cam1Canvas"></canvas></div>
    <div class="panel"><canvas id="cam2Canvas"></canvas></div>
  </div>
  <script>
    const CLEAR_AFTER = {CLEAR_AFTER};

    function initSlideshow(cam, canvasId) {{
      fetch(`/api/images/${{cam}}`)
        .then(res => res.json())
        .then(images => {{
          const canvas = document.getElementById(canvasId);
          const ctx = canvas.getContext('2d');
          let counter = 0;

          function resizeCanvas() {{
            canvas.width = canvas.parentNode.clientWidth;
            canvas.height = canvas.parentNode.clientHeight;
          }}
          window.addEventListener('resize', resizeCanvas);
          resizeCanvas();

          function drawNext() {{
            const img = new Image();
            img.src = images[Math.floor(Math.random() * images.length)];
            img.onload = () => {{
              if (counter > 0 && counter % CLEAR_AFTER === 0) ctx.clearRect(0, 0, canvas.width, canvas.height);
              // Draw image to COVER the canvas (object-fit: cover)
              const scale = Math.max(canvas.width / img.width, canvas.height / img.height);
              const w = img.width * scale;
              const h = img.height * scale;
              const x = (canvas.width - w) / 2;
              const y = (canvas.height - h) / 2;
              ctx.drawImage(img, x, y, w, h);
              counter++;
            }};
            setTimeout(drawNext, 5000 + Math.random() * 2000);
          }}
          drawNext();
        }});
    }}

    document.getElementById('fsBtn').addEventListener('click', () => {{
      const el = document.documentElement;
      if (el.requestFullscreen) el.requestFullscreen();
      else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
      else if (el.msRequestFullscreen) el.msRequestFullscreen();
    }});
    document.addEventListener('fullscreenchange', () => {{
      document.getElementById('fsBtn').style.display = document.fullscreenElement ? 'none' : 'block';
    }});

    window.onload = () => {{
      initSlideshow('cam1', 'cam1Canvas');
      initSlideshow('cam2', 'cam2Canvas');
    }};
  </script>
</body>
</html>'''

# Utility: list all images under a directory
def list_images(cam_dir):
    return [p.relative_to(cam_dir).as_posix() for p in cam_dir.rglob('*') if p.suffix.lower() in EXTS]

from flask import Flask, jsonify, send_from_directory, render_template_string, abort

# API endpoint: JSON list of image URLs for each cam
@app.route('/api/images/<cam>')
def api_images(cam):
    if cam == 'cam1':
        rels = list_images(CAM1_DIR)
        urls = [f'/images/cam1/{r}' for r in rels]
    elif cam == 'cam2':
        rels = list_images(CAM2_DIR)
        urls = [f'/images/cam2/{r}' for r in rels]
    else:
        abort(404)
    return jsonify(urls)

# Serve image files
@app.route('/images/cam1/<path:filename>')
def serve_cam1(filename):
    return send_from_directory(CAM1_DIR, filename)
@app.route('/images/cam2/<path:filename>')
def serve_cam2(filename):
    return send_from_directory(CAM2_DIR, filename)

# Main page
@app.route('/')
def index():
    return render_template_string(TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
