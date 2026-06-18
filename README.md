# birdwatcher
Play with bird detection and recognition

# Tech Stack

Python:
- Python 3.13 or newer
- uv - package manager
- pytest - testing
- ruff - linting and formatting
- pre-commit - run ruff, mypy, tests, etc. on every commit
- just - task running

Computer vision:
- OpenCV 4

ELP High Speed USB Zoom Camera:
- Sensor: Sony IMX577 (12MP)
- Resolutions & Frame Rates:
- 4K @ 60fps
- 1080p @ 230fps
- 12MP @ 30fps
- Interface: USB 3.0 (USB 2.0 compatible)
- Lens: 5–50mm manual zoom (10×), manual focus
- Platform support: Windows, Linux, macOS, Raspberry Pi, Jetson Nano
- Manual: https://manuals.plus/asin/B0CDGX5HYZ

# Development and Deployment

Develop on MacBook (Apple Silicon).
Deploy on Ubuntu Linux.

# Focusing the camera

This camera has manual focus.
Tools:
- On the Mac, use Camera Controller (brew install camera-controller) to set manual focus.
- On Linux, use fswebcam or guvcview — live preview with basic controls; guvcview shows a live histogram which helps
- focus.py is a simple script for this purpose

# Project Structure

Start with just a scripts directory for initial experiments

# TODO

testpaths = ["scripts"] is a placeholder — once you add a real package (e.g. src/bird/), I'd restructure into a proper package with a tests/ dir rather than mixing tests into scripts/.
