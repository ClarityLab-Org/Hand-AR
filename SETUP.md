# Hand-AR Setup Guide

This guide helps you set up the Hand-AR project on your machine.

## Prerequisites

- **Python 3.8+** (tested on Python 3.10)
- **Webcam** connected to your computer
- **Git** (for cloning the repository)

### System Requirements

- **GPU** (optional but recommended for better MediaPipe performance)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: ~500MB for dependencies

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repo-url>
cd Hand-AR
```

### 2. Create Virtual Environment

```bash
# On Linux/macOS
./scripts/setup.sh
source venv/bin/activate

# On Windows
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 3. Install Dependencies

```bash
venv/bin/python -m pip install -r requirements.txt
```

**Note**: The first installation may take 10-15 minutes due to large ML models being downloaded.

### 4. Verify Installation

```bash
venv/bin/python -c "import cv2, mediapipe, ursina, panda3d; print('✓ All dependencies installed successfully')"
```

## Running the Application

### Launch GUI Menu (Recommended)

```bash
venv/bin/python opening.py
```

This opens a user-friendly interface to select 3D models.

### Launch with Default Model

```bash
venv/bin/python trial2.py
```

### Launch with Custom Model

```bash
venv/bin/python trial2.py path/to/your/model.glb
```

**Supported formats**: `.glb`, `.obj`, `.gltf`, `.x`, `.egg` (Panda3D compatible formats)

## Troubleshooting

### "No module named 'mediapipe'"

```bash
# Ensure virtual environment is activated
venv/bin/python -m pip install -r requirements.txt
```

### "Could not open camera"

**Causes**:
- Camera not connected
- Another application using the camera (e.g., Zoom, OBS)
- Missing camera permissions

**Solutions**:
1. Check camera is connected: `ls /dev/video*` (Linux) or Device Manager (Windows)
2. Close other applications using camera
3. Grant camera permissions (Settings > Camera > App permissions)
4. Try different camera index: `python3 trial2.py` (app auto-tries indices 0-4)

### "Model file not found"

- Ensure model file exists in the `models/` directory
- Check file path is correct
- Use `opening.py` to browse and select available models

### Poor Hand Detection

- Ensure good lighting
- Keep hands within camera view
- Adjust `MIN_DETECTION_CONFIDENCE` in `.env` if needed

### Low FPS / Performance Issues

**Check system resources**:
- Monitor CPU/RAM usage (shown in UI)
- Close other applications
- Reduce `FEED_WIDTH` and `FEED_HEIGHT` in `.env`

## File Structure

```
Hand-AR/
├── opening.py             # GUI menu for model selection
├── trial2.py              # Interactive 3D viewer
├── explore.py             # Standalone point-cloud explorer
├── plotcsv.py             # Point-cloud plotting utility
├── handarm/               # Application package
├── models/                # 3D model files (.glb, .obj, etc.)
├── screenshots/           # Captured screenshots
├── requirements.txt       # Python dependencies
└── venv/                  # Virtual environment (created by scripts/setup.sh)
```

## Controls

### In trial2.py

**Right Hand**:
- Index finger → Rotate model
- Pinch gesture → Zoom in/out
- Peace sign → Screenshot

**Left Hand**:
- Open palm → Pause all interactions
- Wrist movement → Translate model

**UI Buttons**:
- Model Lock: Toggle model rotation
- View Presets: Switch between Front/Side/Top/ISO views

## Development Notes

### Adding New Dependencies

If you add new packages, update `requirements.txt`:

```bash
venv/bin/python -m pip freeze > requirements.txt
```

### Environment Variables

Create a `.env` file for custom configuration:

```
CAMERA_INDEX=0
MIN_DETECTION_CONFIDENCE=0.8
```

## Performance Tips

1. Close unnecessary background applications
2. Ensure adequate lighting for hand detection
3. Use a USB webcam instead of laptop built-in for better stability

## Reporting Issues

If you encounter problems:

1. Check the troubleshooting section above
2. Verify all dependencies are installed: `pip list`
3. Check Python version: `python3 --version`
4. Test camera independently: `python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`
5. Provide error message and system specs when reporting

---

**Last Updated**: March 2026 | Python 3.10.19 | Panda3D 1.10.16
