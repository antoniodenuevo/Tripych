#!/usr/bin/env python3
# cam3.py
# last updated 11/08/25
# Goal: behave like cam1 for collage timing/logic, but route audio to Headphones.
# - Webhook order matches cam1 (send AFTER audio swap)
# - Only .wav files
# - Prints "Detected group: X" like cam1

import os, random, time, threading
import numpy as np
import cv2
import mediapipe as mp
import requests
import sounddevice as sd
import soundfile as sf

# ———————————————————————————————————————————————————————————
# Collage webhook (same endpoint as cam1)
# ———————————————————————————————————————————————————————————
WEBHOOK_URL = "http://localhost:5000/webhook"

# ———————————————————————————————————————————————————————————
# Detection / ROI (same defaults as cam1)
# ———————————————————————————————————————————————————————————
CAM_INDEX       = 3
FACE_CONFIDENCE = 0.6
OBJ_CONFIDENCE  = 0.5
ROI_WIDTH_FRAC  = 0.5
ROI_HEIGHT_FRAC = 0.5

# ———————————————————————————————————————————————————————————
# Model/labels paths
# ———————————————————————————————————————————————————————————
SCRIPT_DIR  = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(SCRIPT_DIR, "..", "model", "frozen_inference_graph.pb")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "model", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
LABELS_PATH = os.path.join(SCRIPT_DIR, "..", "model", "object_detection_classes_coco.txt")

# ———————————————————————————————————————————————————————————
# Groups + audio folders (cam3)
# ———————————————————————————————————————————————————————————
GROUPS = {
    'person':      ['person'],
    'transport':   ['bicycle','car','motorcycle','airplane','bus','train','truck','boat'],
    'street':      ['traffic light','street sign','stop sign','bench'],
    'home':        ['bottle','plate','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','bed','dining table','desk','mirror','microwave','oven','toaster','sink','refrigerator','blender','toilet','hair drier','toothbrush','potted plant'],
    'objects':     ['book','clock','vase','window','door','tv','scissors','suitcase'],
    'electronics': ['laptop','mouse','remote','keyboard','cell phone'],
    'animals':     ['elephant','bird']
}
LABEL_TO_GROUP = {label: grp for grp, labels in GROUPS.items() for label in labels}
AUDIO_DIRS = {grp: os.path.join(SCRIPT_DIR, '..', 'assets', 'audio', 'cam3', grp) for grp in GROUPS}

# ———————————————————————————————————————————————————————————
# Detectors
# ———————————————————————————————————————————————————————————
face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=FACE_CONFIDENCE)
net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, CONFIG_PATH)

# ———————————————————————————————————————————————————————————
# Audio: route to Headphones via sounddevice
# (UPDATED: persistent stream, no allocs in callback, hot-swap buffers)
# ———————————————————————————————————————————————————————————
PREFERRED_DEVICE_NAMES = ["Headphones"]
ENABLE_HPF    = True
HPF_CUTOFF_HZ = 320.0
HPF_Q         = 0.707

def _choose_output_device():
    devices = sd.query_devices()
    for want in PREFERRED_DEVICE_NAMES:
        for idx, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and want in d.get("name",""):
                return idx, d["name"]
    default = sd.default.device
    if isinstance(default, (list, tuple)) and len(default) == 2 and default[1] is not None:
        idx = default[1]
        return idx, sd.query_devices(idx)["name"]
    for idx, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            return idx, d.get("name", f"Device {idx}")
    raise RuntimeError("No output-capable audio device found.")

def _design_hpf_biquad(fc, fs, Q):
    import math
    w0   = 2 * math.pi * fc / fs
    cosw = math.cos(w0); sinw = math.sin(w0)
    alpha = sinw / (2 * Q)
    b0 =  (1 + cosw) / 2
    b1 = -(1 + cosw)
    b2 =  (1 + cosw) / 2
    a0 =  1 + alpha
    a1 = -2 * cosw
    a2 =  1 - alpha
    return (b0/a0, b1/a0, b2/a0), (1.0, a1/a0, a2/a0)

def _biquad_filter_inplace(x_f32: np.ndarray, b, a):
    b0, b1, b2 = b
    _,  a1, a2 = a
    x1 = x2 = y1 = y2 = 0.0
    for n in range(x_f32.shape[0]):
        xn = float(x_f32[n])
        yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2
        x_f32[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn

_HPF_COEFFS_CACHE = {}
def _get_hpf_for(sr: int):
    key = (sr, HPF_CUTOFF_HZ, HPF_Q)
    if key not in _HPF_COEFFS_CACHE:
        _HPF_COEFFS_CACHE[key] = _design_hpf_biquad(HPF_CUTOFF_HZ, sr, HPF_Q)
    return _HPF_COEFFS_CACHE[key]

def _load_wav_mono(path: str):
    """WAV -> mono float32 [-1,1], sr int. (stereo summed defensively) + optional HPF."""
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data[:,0] if data.shape[1] == 1 else 0.5*(data[:,0]+data[:,1])
    if ENABLE_HPF:
        b,a = _get_hpf_for(sr)
        _biquad_filter_inplace(mono, b, a)
    np.clip(mono, -1.0, 1.0, out=mono)
    return mono, int(sr)

class LoopedPlayer:
    """
    UPDATED:
    - keeps a single OutputStream open (no reopen on every swap)
    - writes directly into outdata (no per-callback allocations)
    - hot-swaps loop buffer via set_loop()
    """
    def __init__(self, device_index: int, device_name: str, engine_sr: int = 44100):
        self.device_index = device_index
        self.device_name  = device_name
        self.engine_sr    = engine_sr  # persistent stream SR; your files are 44.1k
        self.stream = None
        self.buf = None               # mono float32
        self._pos = 0
        self._lock = threading.RLock()
        self._stopping = False
        self._start_stream_once()

    def _start_stream_once(self):
        if self.stream is not None:
            return
        def _callback(outdata, frames, time_info, status):
            if status:
                # avoid prints in callback
                pass
            outdata[:] = 0.0
            with self._lock:
                if self._stopping or self.buf is None:
                    return
                N = self.buf.shape[0]
                if N == 0:
                    return
                i0 = self._pos
                i1 = i0 + frames
                if i1 <= N:
                    seg = self.buf[i0:i1]
                    self._pos = i1
                    outdata[:seg.shape[0], 0] = seg
                    outdata[:seg.shape[0], 1] = seg
                else:
                    a = N - i0
                    b = frames - a
                    seg1 = self.buf[i0:N]
                    seg2 = self.buf[0:b]
                    outdata[:a, 0] = seg1; outdata[:a, 1] = seg1
                    outdata[a:frames, 0] = seg2; outdata[a:frames, 1] = seg2
                    self._pos = b

        self.stream = sd.OutputStream(
            device=self.device_index,
            samplerate=self.engine_sr,
            channels=2,
            dtype='float32',
            blocksize=512,
            callback=_callback
        )
        self.stream.start()
        print(f"cam3 → audio stream started on '{self.device_name}' @ {self.engine_sr} Hz")

    def set_loop(self, mono_f32: np.ndarray, samplerate: int):
        """Swap the playing loop without reopening the stream."""
        if samplerate != self.engine_sr:
            # For stability, avoid live resampling. If all files are 44.1k, this won't trigger.
            raise ValueError(f"Asset SR {samplerate} != engine SR {self.engine_sr}")
        with self._lock:
            self.buf = mono_f32.astype(np.float32, copy=False)
            self._pos = 0

    def mute(self):
        with self._lock:
            self.buf = None
            self._pos = 0

    def stop(self):
        with self._lock:
            self._stopping = True
        try:
            if self.stream is not None:
                self.stream.stop(); self.stream.close()
        finally:
            with self._lock:
                self.stream = None
                self.buf = None

# Choose device once
_DEVICE_INDEX, _DEVICE_NAME = _choose_output_device()
# Persistent engine at 44.1k (matches your assets)
_PLAYER = LoopedPlayer(_DEVICE_INDEX, _DEVICE_NAME, engine_sr=44100)

# ———————————————————————————————————————————————————————————
# Labels file
# ———————————————————————————————————————————————————————————
with open(LABELS_PATH) as f:
    o_co_co = [l.strip() for l in f]

# ———————————————————————————————————————————————————————————
# Capture + main loop (mirrors cam1 flow)
# ———————————————————————————————————————————————————————————
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
if not cap.isOpened():
    print(f"Could not open camera #{CAM_INDEX}")
    raise SystemExit(1)

window_name = "Cam3 Preview (ROI)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
print(f"🔊 cam3 audio device → '{_DEVICE_NAME}'")
print("Starting detection + preview with ROI (Cam3). Press 'q' to quit.")

current_group = None

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # ROI rectangle (same as cam1)
        h, w = frame.shape[:2]
        rw, rh = int(w*ROI_WIDTH_FRAC), int(h*ROI_HEIGHT_FRAC)
        x1, y1 = (w - rw)//2, (h - rh)//2
        x2, y2 = x1 + rw, y1 + rh

        preview = frame.copy()
        cv2.rectangle(preview, (x1,y1), (x2,y2), (0,0,255), 3)
        cv2.imshow(window_name, preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Detect in ROI
        roi = frame[y1:y2, x1:x2]
        detections = []

        # Face
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        faces = face_detector.process(rgb)
        if faces.detections:
            detections.append(('person', float(faces.detections[0].score[0])))

        # COCO objects
        blob = cv2.dnn.blobFromImage(roi, size=(300,300), swapRB=True)
        net.setInput(blob)
        dets = net.forward()
        for i in range(dets.shape[2]):
            score = float(dets[0,0,i,2])
            if score < OBJ_CONFIDENCE: continue
            cid = int(dets[0,0,i,1])
            name = o_co_co[cid-1]
            grp = LABEL_TO_GROUP.get(name)
            if grp: detections.append((grp, score))

        first_group = detections[0][0] if detections else None
        if first_group:
            print(f"Detected group: {first_group}")

        # On change → hot-swap audio (NO reopen), THEN webhook (same order as your file)
        if first_group and first_group != current_group:
            current_group = first_group

            # choose a random WAV and set loop
            clips = [f for f in os.listdir(AUDIO_DIRS[current_group]) if f.lower().endswith('.wav')]
            if clips:
                path = os.path.join(AUDIO_DIRS[current_group], random.choice(clips))
                try:
                    mono, sr = _load_wav_mono(path)
                    _PLAYER.set_loop(mono, sr)   # <-- persistent stream, no stop/start
                except Exception as e:
                    print(f"Failed to set loop {path}: {e}")
                    _PLAYER.mute()
            else:
                _PLAYER.mute()

            # webhook AFTER audio swap (matches your current timing)
            payload = {"camera": CAM_INDEX, "group": current_group, "timestamp": int(time.time())}
            try:
                requests.post(WEBHOOK_URL, json=payload, timeout=0.5)
            except Exception:
                pass

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    _PLAYER.stop()
    cap.release()
    cv2.destroyAllWindows()
