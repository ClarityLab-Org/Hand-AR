# Hand-ArM2

### Gesture-Based 3D Model Interaction System

Hand-ArM2 is a real-time desktop application for controlling and manipulating 3D models through natural hand gestures captured via webcam — no mouse or keyboard required.

---

## Features

**Gesture Controls**

| Gesture               | Action             |
| --------------------- | ------------------ |
| Index finger movement | Rotate model       |
| Pinch                 | Zoom in / out      |
| Wrist movement        | Pan / move model   |
| Open palm             | Pause interaction  |
| Peace sign            | Capture screenshot |

**3D Model Support**

- Search and download models directly from Sketchfab (free API key required)
- Load local `.glb` and `.obj` files

**Performance Monitoring**

- Real-time FPS, CPU, and RAM stats displayed during interaction

---

## Tech Stack

| Layer          | Technology              |
| -------------- | ----------------------- |
| 3D Rendering   | Ursina, Panda3D         |
| Hand Tracking  | MediaPipe, OpenCV       |
| GUI            | Tkinter                 |
| Model Download | Sketchfab API, Requests |

---

## Project Structure

```
Hand-ArM2/
├── opening.py          # GUI launcher
├── trial2.py           # 3D viewer entry point
├── explore.py          # Standalone point-cloud explorer
├── plotcsv.py          # Point-cloud plotting utility
├── handarm/            # Application package
├── models/             # Local 3D model files
├── screenshots/        # Captured screenshots
├── requirements.txt
└── .env                # API keys (not committed)
```

---

## Installation

### Prerequisites

- Python 3.10
- A webcam

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/Hand-ArM2.git
cd Hand-ArM2
```

### Step 2 — Create the project environment

**Recommended (Linux/macOS):**

```bash
./scripts/setup.sh
source venv/bin/activate
```

**Windows:**

```bash
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Step 3 — Install dependencies

```bash
venv/bin/python -m pip install -r requirements.txt
```

---

## Sketchfab API Setup

To enable online model search and download:

1. Visit [sketchfab.com/settings/password](https://sketchfab.com/settings/password) and generate an API token.
2. Create a `.env` file in the project root:

```env
SKETCHFAB_TOKEN=your_api_token_here
```

> The Sketchfab API is free to use. Without a key, local model loading still works normally.

---

## Dependencies

```
python==3.10
ursina==7.0.0
Panda3D==1.10.16
panda3d-gltf==1.3.0
panda3d-simplepbr==0.13.1
mediapipe==0.10.9
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
numpy==1.26.4
Pillow==10.3.0
psutil>=5.9.0
python-dotenv>=1.0.0
requests>=2.31.0
```



Point-cloud CSV files can be placed in the `models/` directory. The tracked
IIT Jodhpur campus dataset is available as `models/iitj_campus.csv`.
