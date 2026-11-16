# FretVision

**FretVision** is a live audio-visual Windows desktop application built with **PySide6**, **OpenCV**, **MediaPipe HandLandmarker**, and a low-latency **Multi-Resolution YIN + Harmonic NNLS** audio engine.

It combines your webcam or phone camera feed with your microphone or guitar audio interface to accurately display in real time:
- **Pressed & sounding `(string, fret)` positions** (disambiguating duplicate pitches across strings using fingertip location)
- **Live Musical Key & Scale** with top-3 candidates, confidence, and on-fretboard scale highlighting
- **Detected Chord Names**, chromatic tuning/bend cents meter, and **AR camera overlays**
- **Rolling Live Guitar Tablature** with ASCII (`.txt`) and JSON export

## Quick Setup (Windows)

```powershell
cd C:\Users\User\.gemini\antigravity\scratch\fretvision
uv python install 3.12.16
uv venv --python 3.12.16 .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run tests:

```powershell
.venv\Scripts\python.exe -m pytest -v
```

Launch the desktop window:

```powershell
.venv\Scripts\python.exe -m fretvision
```

## Using a Phone Camera

1. **Virtual Webcam (Recommended for lowest latency):** Install DroidCam, Camo, or Iriun on your phone and PC, then select the virtual webcam from the **Camera** dropdown.
2. **Wi-Fi Stream URL:** Paste an MJPEG/HTTP/RTSP URL (e.g. `http://192.168.1.50:8080/video` from IP Webcam) into the **Stream URL** box and click **Connect**.

## Quick Calibration Guide

1. Hold your guitar so the neck (typically frets `0–12` or `1–15`) is clearly visible.
2. Set the **Start Fret** and **End Fret** range, then click **Calibrate Neck** (or try **Auto-Guess** first).
3. Click the 4 corners of that neck region in order:
   1. Start Fret — Top edge of neck
   2. End Fret — Top edge of neck
   3. End Fret — Bottom edge of neck
   4. Start Fret — Bottom edge of neck
4. Optical-flow tracking will automatically lock onto the fretboard texture and follow slight movements while ignoring your moving fingers.
