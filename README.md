# Face Recognition Attendance - AI Service

## Overview

This service consumes an RTSP video stream and produces attendance events.

Pipeline: **RTSP → frame reader (thread) → YOLOv8-face detector → IoU tracker → InsightFace `buffalo_s` recognizer → embedding match against a local SQLite DB → HTTP POST to a backend webhook**.

It is the AI-only component of a larger attendance system. It runs standalone — no backend, no frontend required for development. A fake backend is included for end-to-end testing.

## Prerequisites

- Python 3.11+
- A working RTSP stream (instructions to set up MediaMTX with a sample video below)
- (Optional) NVIDIA GPU with CUDA 12.1+ for faster inference
- 4GB+ free disk space (models will be downloaded on first run)

## Quick Setup

### 1. Clone and Install

```bash
cd ai-service
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your RTSP URL and backend URL
```

### 3. Set Up Test RTSP Stream (if you don't have a camera)

Download MediaMTX from https://github.com/bluenviron/mediamtx/releases.

Edit `mediamtx.yml` and replace the `paths` section with:

```yaml
paths:
  test:
    runOnInit: ffmpeg -re -stream_loop -1 -i C:/videos/people.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:8554/test
    runOnInitRestart: yes
```

Run `mediamtx.exe` in a separate terminal. Test in VLC: `rtsp://localhost:8554/test`.

## Testing Guide

This section is critical. Test components in order from simplest to most complex.

### Test 1: Camera Stream

Verify RTSP connection works.

```bash
python tests/test_camera.py
```

Expected: a window opens showing the video stream. Console prints frame count after 10 seconds.

If it fails:
- "No frame received" → MediaMTX isn't streaming, check it.
- ImportError → run `pip install -r requirements.txt` again.
- Window won't open → reinstall opencv-python.

### Test 2: Face Detection

Test face detection on a static image.

```bash
# Place a photo with faces as test_face.jpg in the project root
python tests/test_detector.py
```

Expected: a window shows the photo with green boxes around faces. Count printed in console.

If it fails:
- Model download error → check internet connection.
- "CUDA not available" → fine for testing, runs on CPU.
- No faces detected → try a different photo with clearer faces.

### Test 3: Enroll Test Employees

You need at least one enrolled employee to test recognition.

```bash
# Create a folder with 5 photos of yourself
mkdir -p photos/test_user
# Copy 5 clear photos of your face into that folder

python enrollment.py --employee-id TEST001 --name "Test User" --photos ./photos/test_user/
```

Expected:

```
Processing photo1.jpg... face detected, embedding generated
Processing photo2.jpg... face detected, embedding generated
...
Enrolled employee TEST001 with 5 embeddings
Saved to ./data/employees.db
```

Check enrolled employees:

```bash
python enrollment.py --list
```

### Test 4: Face Recognition

Verify recognition works against enrolled embeddings.

```bash
# Place a different photo of the enrolled person as test_recognize.jpg
python tests/test_recognizer.py
```

Expected: prints a match with confidence score above the threshold.

### Test 5: Fake Backend

Start the fake backend in a separate terminal to capture events.

```bash
python tests/fake_backend.py
```

Output: `Fake backend listening on http://localhost:3000`.

Leave this running. It will print every event received.

### Test 6: Full Pipeline (End-to-End)

Now run the main service.

```bash
# Make sure these are running in separate terminals:
# 1. MediaMTX (streaming video)
# 2. tests/fake_backend.py (port 3000)

python main.py
```

Expected console output:

```
INFO - Starting AI service
INFO - Loading models...
INFO - YOLO loaded on cuda (or cpu)
INFO - InsightFace loaded
INFO - Loaded 1 employees from database
INFO - Connecting to rtsp://localhost:8554/test
INFO - Camera stream connected
INFO - Processing started
DEBUG - Frame 30 - 1 face detected, 1 tracked
DEBUG - Track 0 stable for 3 frames, recognizing...
DEBUG - Match: TEST001 confidence 0.78
INFO - Event sent: TEST001
DEBUG - Processing FPS: 28.5
```

In the `fake_backend.py` terminal you should see:

```
==================================================
RECEIVED EVENT
Employee: TEST001
Time: 2026-04-25T12:34:56
Confidence: 0.78
==================================================
```

If `DEBUG=true`, an OpenCV window shows the video with bounding boxes and names.

### Test 7: Edge Cases

Once basic flow works, test these scenarios:

**Multiple faces:** use a video with 2-3 people. You should see multiple tracks and recognitions.

**Unknown person:** show a face that isn't enrolled. The detector should still detect it but logs:

```
DEBUG - Face detected but no match (best similarity: 0.32 for TEST001)
```

**Reconnection:** stop MediaMTX while the service is running. The camera thread logs an error and retries with backoff. Restart MediaMTX, the service should reconnect automatically.

**Cooldown:** the same person walking past repeatedly should only generate one event per minute (configurable via `COOLDOWN_SECONDS`).

**Performance:** watch the FPS counter. On RTX 4050 expect 30-60 fps. On CPU expect 5-10 fps. If under 5 fps with a GPU available, the GPU isn't being used.

Check GPU usage in another terminal:

```bash
nvidia-smi -l 1
```

GPU utilization should be 30-80% during processing.

## Troubleshooting

### "RTSP stream not opening"

- Verify the stream works in VLC first: `rtsp://localhost:8554/test`.
- Check the firewall isn't blocking port 8554.
- Try `127.0.0.1` instead of `localhost`.

### "CUDA not available" but you have an RTX GPU

Reinstall PyTorch with CUDA:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```python
import torch
print(torch.cuda.is_available())  # should print True
```

### "Connection refused" to backend

`tests/fake_backend.py` is not running. Start it first.

### Faces detected but never recognized

1. Lower `RECOGNITION_THRESHOLD` to 0.4 in `.env`.
2. Re-enroll with more/better photos (5-10 photos, different angles, good lighting).
3. Check the person being tested is actually enrolled: `python enrollment.py --list`.

### Same person triggers events repeatedly

Tracker not working properly. Check:

- `COOLDOWN_SECONDS` in `.env` is set (default 60).
- Tracker IoU threshold (in `tracking/tracker.py`, default 0.4).

### Video plays in VLC but Python shows black/no frames

TCP transport setting not applied. Verify in `main.py`:

```python
import config  # MUST be imported before cv2 anywhere
import cv2
```

`config.py` sets `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` at import time.

### Low FPS even with GPU

- Check `nvidia-smi` while running. If GPU usage is 0%, CUDA setup is wrong.
- Try a smaller model (YOLOv8n is already the default).
- Increase `FRAME_SKIP` in `.env`.

### "ImportError: DLL load failed" (Windows)

Install Microsoft Visual C++ Redistributable 2019+:
https://aka.ms/vs/17/release/vc_redist.x64.exe

## Performance Tips

For maximum performance on RTX 4050:

1. Convert YOLO to TensorRT: `yolo export model=data/models/yolov8n-face.pt format=engine half=True`.
2. Point `detection/detector.py` at the resulting `.engine` file.
3. Set `USE_GPU=true` in `.env`.
4. Use an H.264 sub-stream (640x480) instead of the main stream.

Expected performance:

- RTX 4050: 60+ fps full pipeline
- Jetson Orin Nano: 30 fps full pipeline
- CPU only (modern i7): 5-10 fps

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `RTSP_URL` | `rtsp://localhost:8554/test` | Camera stream URL |
| `BACKEND_WEBHOOK_URL` | `http://localhost:3000/api/attendance/events` | Where to send events |
| `BACKEND_API_KEY` | _(empty)_ | API key for backend auth (sent as `X-API-Key`) |
| `DETECTION_CONFIDENCE` | `0.5` | Min confidence for face detection |
| `RECOGNITION_THRESHOLD` | `0.5` | Min similarity for face match |
| `FRAME_SKIP` | `3` | Process every Nth frame |
| `DB_PATH` | `./data/employees.db` | SQLite database location |
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR |
| `DEBUG` | `false` | Show debug window with overlays |
| `COOLDOWN_SECONDS` | `60` | Min seconds between events for same person |
| `USE_GPU` | `auto` | auto, true, false |

## Project Architecture

```
[Cameras] → [RTSP Stream] → [Frame Reader Thread]
                              ↓
                         [Face Detector (YOLO)]
                              ↓
                         [Face Tracker (IoU)]
                              ↓
                    [Face Recognizer (InsightFace)]
                              ↓
                       [Embedding Match (DB)]
                              ↓
                       [Event Publisher (HTTP)]
                              ↓
                          [Backend API]
```

## Next Steps

After verifying this works:

1. Build the backend (NestJS API) — separate project.
2. Build the frontend (React dashboard) — separate project.
3. Connect them via the webhook.
4. Deploy the AI service to a Jetson Orin Nano for production.

## Common Commands

```bash
# Run the service
python main.py

# Run with debug mode
DEBUG=true python main.py

# Enroll new employee
python enrollment.py --employee-id E001 --name "John Doe" --photos ./photos/john/

# List employees
python enrollment.py --list

# Delete employee
python enrollment.py --delete E001

# Run individual tests
python tests/test_camera.py
python tests/test_detector.py
python tests/test_recognizer.py

# Run fake backend for testing
python tests/fake_backend.py
```
