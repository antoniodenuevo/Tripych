# Triptych 1

Triptych 1 is an audiovisual installation where 3 cameras and screens interact in real time, transforming image recognition into a live collage of sound and images.

## Project structure
- `cam1.py`, `cam2.py`, `cam3.py` – camera scripts running real-time object detection and triggering sound/visual events.  
- `collage-seq.py` – generates visual collages, layering cutouts and backgrounds in real time.
- `img-seq.py` – shows images. use one seq per screen.
- `assets/` – contains images and sounds used by the collage-seq.py.
- `visuals/` – contains images used by img-seq.py.

## Requirements
- Python 3.9+  
- TensorFlow / MediaPipe  
- OpenCV  
- Pygame  
- Flask (for live server)  

## How it works
Each screen plays a looping sequence of images that acts as a visual “score.”  
Cameras detect objects in these images and send labels to sequencing scripts.  
The scripts trigger sounds, mix in glitches, and generate visual collages in real time.  

More info:
https://antoniodenuevo.com/project/triptych
