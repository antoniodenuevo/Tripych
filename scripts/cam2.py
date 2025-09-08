#!/usr/bin/env python3
# cam2.py
# last updated 09/08/25
# Cam2: sums audio to mono and routes to RIGHT channel only.
# Optional high-pass filter (HPF) to tame bassy rooms.
# Includes cam preview, ROI crop, audio, and collage-webhook trigger.
# To disable the bass cut during soundcheck, set ENABLE_HPF = False.
# To tweak the cut, adjust HPF_CUTOFF_HZ (e.g., 120 or 180).


import os
import random
import time
import cv2
import mediapipe as mp
import pygame
import pygame.sndarray
import numpy as np
import requests

# ———————————————————————————————————————————————————————————
# Webhook (placeholder)
# ———————————————————————————————————————————————————————————
WEBHOOK_URL = "http://localhost:5000/webhook"

# ———————————————————————————————————————————————————————————
# Detection / ROI config
# ———————————————————————————————————————————————————————————
CAM_INDEX       = 2      
FACE_CONFIDENCE = 0.6
OBJ_CONFIDENCE  = 0.5
ROI_WIDTH_FRAC  = 0.5
ROI_HEIGHT_FRAC = 0.5

# ———————————————————————————————————————————————————————————
# Audio output & filter config (EDIT DURING SOUNDCHECK)
# ———————————————————————————————————————————————————————————
SAMPLE_RATE     = 44100
ENABLE_HPF      = True      # ← toggle on/off quickly here
HPF_CUTOFF_HZ   = 320.0     # 
HPF_Q           = 0.707     # ~Butterworth
AUDIO_BUFFER    = 512       # latency/CPU tradeoff

# ———————————————————————————————————————————————————————————
# Model/labels paths
# ———————————————————————————————————————————————————————————
SCRIPT_DIR  = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(SCRIPT_DIR, "..", "model", "frozen_inference_graph.pb")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "model", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
LABELS_PATH = os.path.join(SCRIPT_DIR, "..", "model", "object_detection_classes_coco.txt")

# ———————————————————————————————————————————————————————————
# Groups and audio folders
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

# Use cam2 audio folders
AUDIO_DIRS = {grp: os.path.join(SCRIPT_DIR, '..', 'assets', 'audio', 'cam2', grp) for grp in GROUPS}

# ———————————————————————————————————————————————————————————
# Init detectors
# ———————————————————————————————————————————————————————————
face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=FACE_CONFIDENCE)
net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, CONFIG_PATH)

# ———————————————————————————————————————————————————————————
# Audio init (stereo device; we’ll pack to RIGHT-only)
# ———————————————————————————————————————————————————————————
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=AUDIO_BUFFER)
current_group = None

# Cache: filepath -> preprocessed pygame Sound (mono-summed, HPF (optional), RIGHT-only)
_RIGHT_SOUND_CACHE = {}

# ———————————————————————————————————————————————————————————
# High-pass filter helpers (biquad)
# ———————————————————————————————————————————————————————————
def _design_hpf_biquad(fc, fs, Q):
    """Return (b, a) biquad coeffs for a 2nd-order HPF."""
    import math
    w0   = 2 * math.pi * fc / fs
    cosw = math.cos(w0)
    sinw = math.sin(w0)
    alpha = sinw / (2 * Q)
    b0 =  (1 + cosw) / 2
    b1 = -(1 + cosw)
    b2 =  (1 + cosw) / 2
    a0 =  1 + alpha
    a1 = -2 * cosw
    a2 =  1 - alpha
    return (
        (b0/a0, b1/a0, b2/a0),
        (1.0,   a1/a0, a2/a0)
    )

def _biquad_filter_inplace(x_f32: np.ndarray, b, a):
    """In-place Direct Form I on float32 mono array."""
    b0, b1, b2 = b
    _,  a1, a2 = a
    x1 = x2 = y1 = y2 = 0.0
    for n in range(x_f32.shape[0]):
        xn = float(x_f32[n])
        yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2
        x_f32[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn

# Precompute HPF coeffs once
_HPF_COEFFS = _design_hpf_biquad(HPF_CUTOFF_HZ, SAMPLE_RATE, HPF_Q)

# ———————————————————————————————————————————————————————————
# Audio preprocessing: sum→mono, optional HPF, pack RIGHT-only, cache
# ———————————————————————————————————————————————————————————
def _make_right_only_sound(path: str) -> pygame.mixer.Sound:
    abspath = os.path.abspath(path)
    if abspath in _RIGHT_SOUND_CACHE:
        return _RIGHT_SOUND_CACHE[abspath]

    base_sound = pygame.mixer.Sound(abspath)
    arr = pygame.sndarray.array(base_sound)  # int16; mono:(N,), stereo:(N,2)

    # Sum stereo -> mono (int16)
    if arr.ndim == 2 and arr.shape[1] == 2:
        mono32 = (arr[:, 0].astype(np.int32) + arr[:, 1].astype(np.int32)) // 2
        mono_i16 = np.clip(mono32, -32768, 32767).astype(np.int16)
    else:
        mono_i16 = arr.astype(np.int16)

    # Optional HPF
    if ENABLE_HPF:
        mono_f32 = mono_i16.astype(np.float32) / 32768.0
        b, a = _HPF_COEFFS
        _biquad_filter_inplace(mono_f32, b, a)
        mono_i16 = np.clip(mono_f32 * 32767.0, -32768, 32767).astype(np.int16)

    # Pack to stereo with Left muted (RIGHT-only)
    stereo = np.zeros((mono_i16.shape[0], 2), dtype=np.int16)
    stereo[:, 1] = mono_i16  # Right channel

    right_sound = pygame.sndarray.make_sound(stereo)
    _RIGHT_SOUND_CACHE[abspath] = right_sound
    return right_sound

# ———————————————————————————————————————————————————————————
# Labels
# ———————————————————————————————————————————————————————————
o_co_co = []
with open(LABELS_PATH) as f:
    o_co_co = [l.strip() for l in f]

# ———————————————————————————————————————————————————————————
# Capture + loop
# ———————————————————————————————————————————————————————————
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
if not cap.isOpened():
    print(f"Could not open camera #{CAM_INDEX}")
    exit(1)

window_name = 'Cam2 Preview (ROI)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
print("Starting detection + preview with ROI (Cam2). Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ROI box
        h, w = frame.shape[:2]
        rw, rh = int(w * ROI_WIDTH_FRAC), int(h * ROI_HEIGHT_FRAC)
        x1, y1 = (w - rw)//2, (h - rh)//2
        x2, y2 = x1 + rw, y1 + rh

        # preview
        preview = frame.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.imshow(window_name, preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # detect in ROI
        roi = frame[y1:y2, x1:x2]
        detections = []

        # faces
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        faces = face_detector.process(rgb)
        if faces.detections:
            detections.append(('person', float(faces.detections[0].score[0])))

        # objects
        blob = cv2.dnn.blobFromImage(roi, size=(300,300), swapRB=True)
        net.setInput(blob)
        dets = net.forward()
        for i in range(dets.shape[2]):
            score = float(dets[0,0,i,2])
            if score < OBJ_CONFIDENCE:
                continue
            cid = int(dets[0,0,i,1])
            name = o_co_co[cid-1]
            grp = LABEL_TO_GROUP.get(name)
            if grp:
                detections.append((grp, score))

        first_group = detections[0][0] if detections else None
        if first_group:
            print(f"Detected group: {first_group}")

        # on change → audio + webhook
        if first_group and first_group != current_group:
            pygame.mixer.stop()
            current_group = first_group

            # play random clip (RIGHT-only, cached)
            clips = [f for f in os.listdir(AUDIO_DIRS[current_group]) if f.lower().endswith('.wav')]
            if clips:
                sound_path = os.path.join(AUDIO_DIRS[current_group], random.choice(clips))
                try:
                    _make_right_only_sound(sound_path).play(loops=-1)
                except Exception as e:
                    print(f"Failed to play {sound_path}: {e}")

            # webhook
            payload = {"camera": CAM_INDEX, "group": current_group, "timestamp": int(time.time())}
            try:
                requests.post(WEBHOOK_URL, json=payload, timeout=0.5)
            except Exception:
                pass

except KeyboardInterrupt:
    print("\nStopped by user.")

# ———————————————————————————————————————————————————————————
# Cleanup
# ———————————————————————————————————————————————————————————
cap.release()
cv2.destroyAllWindows()
