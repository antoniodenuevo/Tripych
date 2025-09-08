#!/usr/bin/env python3
# collage-seq-strip.py — 3-column layout (independent strips)
# Adds: full-screen glitch overlay when receiving group:"glitch"
# Keeps: random in-strip placement, per-strip overlay resets, custom aspect ratio
# last updated 14/08/2025

import os
import json
import queue
from flask import Flask, Response, request, render_template_string, send_from_directory, abort

# --------------------
# Server config
# --------------------
BASE_DIR = os.path.dirname(__file__)
COLLAGE_DIR = os.path.join(BASE_DIR, '..', 'assets', 'collage')
MAX_OVERLAYS = 500
EVENT_QUEUE = queue.Queue()

app = Flask(__name__)
EXTS = {'.png', '.jpg', '.jpeg', '.gif'}

def list_images(group):
    grp_dir = os.path.join(COLLAGE_DIR, group)
    if not os.path.isdir(grp_dir):
        return []
    urls = []
    for root, _, files in os.walk(grp_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in EXTS:
                rel = os.path.relpath(os.path.join(root, fname), COLLAGE_DIR)
                urls.append(f'/images/{rel.replace(os.path.sep, "/")}')
    return urls

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    if not data or 'group' not in data:
        return ('Bad Request', 400)
    EVENT_QUEUE.put(data)
    print(f"[webhook] queued: {data}")
    return ('', 204)

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            evt = EVENT_QUEUE.get()
            yield f"data: {json.dumps(evt)}\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/images/<path:group>')
def api_images(group):
    imgs = list_images(group)
    if not imgs:
        return ('Not Found', 404)
    return Response(json.dumps(imgs), mimetype='application/json')

@app.route('/images/<path:relpath>')
def serve_images(relpath):
    file_path = os.path.join(COLLAGE_DIR, *relpath.split('/'))
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(COLLAGE_DIR, relpath)

# --------------------
# Client (HTML/JS)
# --------------------
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Collage Sequencer — 3 Columns (per-strip reset + glitch overlay)</title>
  <style>
    body { margin:0; background:#000; overflow:hidden; }
    #stage { position: relative; }
    canvas { display:block; position:absolute; }
    #glitchOverlay {
      position:absolute; display:none; z-index:20;
      object-fit:cover; /* fill canvas area */
      image-rendering:auto;
    }
    #fsBtn {
      position: fixed; top: 10px; right: 10px; z-index: 30;
      background: rgba(255,255,255,0.25); color: #fff; border: 0;
      padding: 8px 12px; font-size: 14px; cursor: pointer; border-radius: 6px;
    }
  </style>
</head>
<body>
  <button id="fsBtn">⛶ Fullscreen</button>
  <div id="stage">
    <canvas id="collageCanvas"></canvas>
    <img id="glitchOverlay" alt="glitch"/>
  </div>
  <script>
    const MAX_OVERLAYS = {{ max_overlays }};
    const BG = '#000';
    const PALETTE = ['#000000'];

    // Aspect Ratio (fit-to-window, centered). Change with setAspectRatio(16,9) etc.
    let ASPECT_RATIO = 3.75 / 3;
    function setAspectRatio(w, h) {
      if (typeof h === 'number' && h > 0) {
        ASPECT_RATIO = w / h;
      } else if (typeof w === 'number' && w > 0) {
        ASPECT_RATIO = w;
      }
      resize();
      paintAllStrips();
      if (overlayVisible) positionOverlay(); // keep glitch img aligned
    }
    window.setAspectRatio = setAspectRatio;

    // Random sizing
    const MIN_SCALE = 0.15;
    const MAX_SCALE = 0.50;

    function cameraToCol(cam) {
      if (cam === 1) return 0;
      if (cam === 2) return 1;
      if (cam === 3) return 2;
      return 1;
    }

    // Per-camera/strip state
    const cams = {
      1: { currentGroup: null, timerId: null, version: 0, overlays: 0, color: PALETTE[0] },
      2: { currentGroup: null, timerId: null, version: 0, overlays: 0, color: PALETTE[1 % PALETTE.length] },
      3: { currentGroup: null, timerId: null, version: 0, overlays: 0, color: PALETTE[2 % PALETTE.length] },
    };

    let globalOverlayCount = 0;
    const stage  = document.getElementById('stage');
    const canvas = document.getElementById('collageCanvas');
    const ctx = canvas.getContext('2d', { alpha: false });
    const glitchImg = document.getElementById('glitchOverlay');
    let overlayVisible = false;
    let overlayEndsAt = 0;
    const imageCache = {};

    function resize() {
      const winW = window.innerWidth;
      const winH = window.innerHeight;

      let targetH = winH;
      let targetW = Math.round(targetH * ASPECT_RATIO);
      if (targetW > winW) {
        targetW = winW;
        targetH = Math.round(targetW / ASPECT_RATIO);
      }

      canvas.width = targetW;
      canvas.height = targetH;

      // center the stage; keep overlay aligned
      stage.style.width  = targetW + "px";
      stage.style.height = targetH + "px";
      stage.style.position = "absolute";
      stage.style.left = ((winW - targetW) / 2) + "px";
      stage.style.top  = ((winH - targetH) / 2) + "px";

      // Clear full canvas before repainting strips
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      positionOverlay();
    }

    function positionOverlay() {
      glitchImg.style.left = "0px";
      glitchImg.style.top  = "0px";
      glitchImg.style.width  = canvas.width + "px";
      glitchImg.style.height = canvas.height + "px";
    }

    window.addEventListener('resize', () => { resize(); paintAllStrips(); });
    resize();

    // Paint all strips with their current color (used on init and resize)
    function paintAllStrips() {
      [1,2,3].forEach(cam => paintStrip(cameraToCol(cam), cams[cam].color));
    }

    // Fullscreen
    const fsBtn = document.getElementById('fsBtn');
    fsBtn.onclick = () => {
      const el = document.documentElement;
      (el.requestFullscreen||el.webkitRequestFullscreen||el.msRequestFullscreen).call(el);
    };
    document.addEventListener('fullscreenchange', () => {
      fsBtn.style.display = document.fullscreenElement ? 'none' : 'block';
    });

    // Helpers for strip geometry + painting
    function colRect(colIndex) {
      const W = canvas.width;
      const colW = Math.floor(W / 3);
      const x = colIndex * colW;
      return { x, y: 0, w: colW, h: canvas.height };
    }

    function paintStrip(colIndex, color) {
      const col = colRect(colIndex);
      ctx.save();
      ctx.beginPath();
      ctx.rect(col.x, col.y, col.w, col.h);
      ctx.clip();
      ctx.clearRect(col.x, col.y, col.w, col.h);
      ctx.fillStyle = color;
      ctx.fillRect(col.x, col.y, col.w, col.h);
      ctx.restore();
    }

    // Initial paint
    paintAllStrips();

    // SSE listener
    const source = new EventSource('/stream');
    source.onmessage = async (e) => {
      const evt = JSON.parse(e.data);
      const cam = Number(evt.camera ?? 2);
      const grp = evt.group;

      // GLITCH overlay handling
      if (grp === 'glitch') {
        const dur = Number(evt.duration_ms ?? 8000);
        await showGlitchOverlay(dur);
        return;
      }

      const colState = cams[cam] || cams[2];

      // bump version to invalidate old timers
      colState.version++;
      const v = colState.version;

      // cancel old timer
      if (colState.timerId) {
        clearTimeout(colState.timerId);
        colState.timerId = null;
      }

      colState.currentGroup = grp;

      if (!imageCache[grp]) {
        try {
          const res = await fetch(`/api/images/${grp}`);
          imageCache[grp] = res.ok ? await res.json() : [];
        } catch {
          imageCache[grp] = [];
        }
      }

      if (v !== cams[cam].version) return; // drop stale
      addImage(cam);        // immediate
      scheduleNext(cam);   // then keep going
    };

    // Load glitch list lazily
    async function ensureGlitchList() {
      if (!imageCache['glitch']) {
        try {
          const res = await fetch('/api/images/glitch');
          imageCache['glitch'] = res.ok ? await res.json() : [];
        } catch {
          imageCache['glitch'] = [];
        }
      }
      return imageCache['glitch'];
    }

    async function showGlitchOverlay(durationMs) {
      const list = await ensureGlitchList();
      const pick = list.length ? list[Math.floor(Math.random() * list.length)] : null;

      if (pick) glitchImg.src = pick;
      else {
        // fallback to solid color if no images
        glitchImg.src = '';
        glitchImg.style.background = '#fff';
      }

      overlayVisible = true;
      overlayEndsAt = Date.now() + durationMs;
      glitchImg.style.display = 'block';
      positionOverlay();

      // extend window if new glitches come in while active
      const thisCallEnds = overlayEndsAt;
      setTimeout(() => {
        // only hide if we're past the last scheduled end
        if (Date.now() >= overlayEndsAt) {
          overlayVisible = false;
          glitchImg.style.display = 'none';
          glitchImg.style.background = 'transparent';
        }
      }, durationMs + 50);
    }

    function scheduleNext(cam) {
      const colState = cams[cam];
      const delay = 60 + Math.random() * 200;
      colState.timerId = setTimeout(() => {
        addImage(cam);
        scheduleNext(cam);
      }, delay);
    }

    function addImage(cameraId) {
      const colState = cams[cameraId];
      const grp = colState.currentGroup;
      if (!grp || !imageCache[grp]?.length) return;

      const url = imageCache[grp][Math.floor(Math.random() * imageCache[grp].length)];
      const img = new Image();
      img.onload = () => {
        if (grp !== cams[cameraId].currentGroup) return;

        colState.overlays++;
        globalOverlayCount++;

        const colIndex = cameraToCol(cameraId);
        const col = colRect(colIndex);

        if (colState.overlays > MAX_OVERLAYS) {
          colState.overlays = 1;
          colState.color = '#000000';
          paintStrip(colIndex, colState.color);
        }

        const fitCap = Math.min(
          4,
          Math.max(0.05, canvas.width / img.width, canvas.height / img.height)
        );
        const s = MIN_SCALE + Math.random() * (MAX_SCALE - MIN_SCALE);
        const scale = Math.min(s, fitCap);

        const w = img.width * scale;
        const h = img.height * scale;

        const x = col.x + Math.random() * Math.max(1, (col.w - w));
        const y = col.y + Math.random() * Math.max(1, (col.h - h));

        ctx.save();
        ctx.beginPath();
        ctx.rect(col.x, col.y, col.w, col.h);
        ctx.clip();
        ctx.drawImage(img, x, y, w, h);
        ctx.restore();
      };
      img.src = url;
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(TEMPLATE, max_overlays=MAX_OVERLAYS)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
