#!/usr/bin/env python3
# cam1.py
# last updated 14/08/25
# Cam1: sums audio to mono and routes to LEFT channel only.
# Adds "glitch" trigger: every N label changes, play a glitch sound for 5s
# and send a 'glitch' webhook to the collage server.
#
# Glitch is now on its own mixer channel (cannot be interrupted by normal),
# loops for full duration, and normal audio is paused while glitch is active.

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
CAM_INDEX       = 1
FACE_CONFIDENCE = 0.6
OBJ_CONFIDENCE  = 0.5
ROI_WIDTH_FRAC  = 0.5
ROI_HEIGHT_FRAC = 0.5

# ———————————————————————————————————————————————————————————
# Audio output & filter config (EDIT THESE DURING SOUNDCHECK)
# ———————————————————————————————————————————————————————————
SAMPLE_RATE     = 44100
ENABLE_HPF      = True
HPF_CUTOFF_HZ   = 320.0
HPF_Q           = 0.707
AUDIO_BUFFER    = 512

# ———————————————————————————————————————————————————————————
# Glitch config
# ———————————————————————————————————————————————————————————
GLITCH_EVERY_N     = 15        # trigger glitch every N label changes
GLITCH_DURATION_MS = 8000     # 5 seconds

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
AUDIO_DIRS = {grp: os.path.join(SCRIPT_DIR, '..', 'assets', 'audio', 'cam1', grp) for grp in GROUPS}
AUDIO_DIR_GLITCH = os.path.join(SCRIPT_DIR, '..', 'assets', 'audio', 'cam1', 'glitch')

# ———————————————————————————————————————————————————————————
# Init detectors
# ———————————————————————————————————————————————————————————
face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=FACE_CONFIDENCE)
net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, CONFIG_PATH)

# ———————————————————————————————————————————————————————————
# Audio init — reserve channels
# ———————————————————————————————————————————————————————————
# channel 0 => GLITCH (priority)
# channel 1 => NORMAL loop
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=AUDIO_BUFFER)
pygame.mixer.set_num_channels(2)
CHAN_GLITCH = pygame.mixer.Channel(0)
CHAN_NORMAL = pygame.mixer.Channel(1)

current_group = None
label_change_count = 0

# Glitch state
glitch_active = False
glitch_ends_at = 0.0

# Cache: filepath -> preprocessed pygame Sound (mono-summed, HPF (optional), LEFT-only)
_LEFT_SOUND_CACHE = {}

# ———————————————————————————————————————————————————————————
# High-pass filter helpers (biquad)
# ———————————————————————————————————————————————————————————
def _design_hpf_biquad(fc, fs, Q):
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
    b0, b1, b2 = b
    _,  a1, a2 = a
    x1 = x2 = y1 = y2 = 0.0
    for n in range(x_f32.shape[0]):
        xn = float(x_f32[n])
        yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2
        x_f32[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn

_HPF_COEFFS = _design_hpf_biquad(HPF_CUTOFF_HZ, SAMPLE_RATE, HPF_Q)

# ———————————————————————————————————————————————————————————
# Audio preprocessing: sum→mono, optional HPF, pack LEFT-only, cache
# ———————————————————————————————————————————————————————————
def _make_left_only_sound(path: str) -> pygame.mixer.Sound:
    abspath = os.path.abspath(path)
    if abspath in _LEFT_SOUND_CACHE:
        return _LEFT_SOUND_CACHE[abspath]

    base_sound = pygame.mixer.Sound(abspath)
    arr = pygame.sndarray.array(base_sound)  # int16; mono:(N,), stereo:(N,2)

    if arr.ndim == 2 and arr.shape[1] == 2:
        mono32 = (arr[:, 0].astype(np.int32) + arr[:, 1].astype(np.int32)) // 2
        mono_i16 = np.clip(mono32, -32768, 32767).astype(np.int16)
    else:
        mono_i16 = arr.astype(np.int16)

    if ENABLE_HPF:
        mono_f32 = mono_i16.astype(np.float32) / 32768.0
        b, a = _HPF_COEFFS
        _biquad_filter_inplace(mono_f32, b, a)
        mono_i16 = np.clip(mono_f32 * 32767.0, -32768, 32767).astype(np.int16)

    stereo = np.zeros((mono_i16.shape[0], 2), dtype=np.int16)
    stereo[:, 0] = mono_i16  # LEFT only

    left_sound = pygame.sndarray.make_sound(stereo)
    _LEFT_SOUND_CACHE[abspath] = left_sound
    return left_sound

# ———————————————————————————————————————————————————————————
# Labels
# ———————————————————————————————————————————————————————————
with open(LABELS_PATH) as f:
    o_co_co = [l.strip() for l in f]

# ———————————————————————————————————————————————————————————
# Audio controls
# ———————————————————————————————————————————————————————————
def _start_normal_loop(group_name: str):
    """Start/replace the looping sound for a group on the normal channel."""
    clips = [f for f in os.listdir(AUDIO_DIRS[group_name]) if f.lower().endswith('.wav')]
    if not clips:
        return
    sound_path = os.path.join(AUDIO_DIRS[group_name], random.choice(clips))
    try:
        snd = _make_left_only_sound(sound_path)
        CHAN_NORMAL.stop()
        CHAN_NORMAL.play(snd, loops=-1)
    except Exception as e:
        print(f"Failed to play {sound_path}: {e}")

def _start_glitch():
    """Start a glitch window: stop normal, play glitch on dedicated channel, send webhook."""
    global glitch_active, glitch_ends_at, current_group

    glitch_active = True
    glitch_ends_at = time.time() + (GLITCH_DURATION_MS / 1000.0)

    # Stop ONLY normal channel; leave glitch channel free
    CHAN_NORMAL.stop()

    # Play a random glitch WAV (looping) on the dedicated channel
    try:
        clips = [f for f in os.listdir(AUDIO_DIR_GLITCH) if f.lower().endswith('.wav')]
        if clips:
            sound_path = os.path.join(AUDIO_DIR_GLITCH, random.choice(clips))
            snd = _make_left_only_sound(sound_path)
            CHAN_GLITCH.stop()
            CHAN_GLITCH.play(snd, loops=-1)
    except Exception as e:
        print(f"[glitch] audio skipped: {e}")

    # Notify collage
    payload = {
        "camera": CAM_INDEX,
        "group": "glitch",
        "duration_ms": GLITCH_DURATION_MS,
        "timestamp": int(time.time())
    }
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=0.5)
        print(f"[glitch] sent to collage for {GLITCH_DURATION_MS} ms")
    except Exception:
        pass

    # Force next normal detection to be treated as "fresh"
    current_group = None

def _maybe_end_glitch_and_resume():
    """If glitch window elapsed, stop glitch channel and (optionally) resume current group loop."""
    global glitch_active
    if glitch_active and time.time() >= glitch_ends_at:
        glitch_active = False
        CHAN_GLITCH.stop()
        if current_group:
            _start_normal_loop(current_group)

# ———————————————————————————————————————————————————————————
# Capture + loop
# ———————————————————————————————————————————————————————————
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
if not cap.isOpened():
    print(f"Could not open camera #{CAM_INDEX}")
    exit(1)

window_name = 'Cam1 Preview (ROI)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
print("Starting detection + preview with ROI (Cam1). Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Glitch window handling
        _maybe_end_glitch_and_resume()

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

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        faces = face_detector.process(rgb)
        if faces.detections:
            detections.append(('person', float(faces.detections[0].score[0])))

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

        # During glitch window, skip normal audio/webhook posts
        if glitch_active:
            continue

        # on change → audio + webhook + glitch counting
        if first_group and first_group != current_group:
            label_change_count += 1

            # maybe trigger glitch
            if GLITCH_EVERY_N > 0 and (label_change_count % GLITCH_EVERY_N) == 0:
                _start_glitch()
                continue  # don't start a normal loop this frame

            # start/replace normal loop for this group
            current_group = first_group
            _start_normal_loop(current_group)

            # notify collage with normal group
            payload = {"camera": CAM_INDEX, "group": current_group, "timestamp": int(time.time())}
            try:
                requests.post(WEBHOOK_URL, json=payload, timeout=0.5)
            except Exception:
                pass

except KeyboardInterrupt:
    print("\nStopped by user.")

cap.release()
cv2.destroyAllWindows()
