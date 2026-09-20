# Meme Mirror

Meme Mirror is a webcam-based pose game that turns your face and hands into a live meme generator. It uses MediaPipe pose detection to recognize a handful of expressive poses, swaps them into meme frames, and adds a chaotic meme-show vibe with overlays, challenge modes, and recording options.

## Features

- Live webcam pose detection
- Meme reactions for multiple poses:
  - staring
  - thinking
  - pointing
  - shocked
- Face crop blending into meme frames
- Smooth transitions between poses
- Fullscreen, vertical mode, PiP and side-by-side layouts
- Watermark, recording, screenshot, and GIF capture tools
- `67` challenge mode: raise both arms to match the meme pattern and count cycles
- Local leaderboard for the `67` challenge
- Debug and help overlays for testing and UX

## Requirements

- Python 3.11+
- Webcam
- OpenCV and MediaPipe dependencies from `requirements.txt`

## Installation

1. Clone or download the project.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt)
.\.venv\Scripts\activate.bat
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

From the project root:

```bash
python main.py
```

On Windows, this also works:

```powershell
py -3.11 main.py
```

## Controls

- `H` — toggle help overlay
- `A` — toggle face crop
- `V` — vertical mode
- `W` — toggle watermark
- `S` — save screenshot
- `R` — start/stop recording
- `F` — toggle fullscreen
- `G` — start the `67` challenge
- `P` — pause/unpause
- `M` — meme-only view
- `Q` — quit
- `Esc` — cancel the `67` challenge

## The `67` Challenge

The `67` game is a quick reaction challenge.

- Press `G` to begin the challenge countdown.
- The game watches your arm motion and counts successful mirrored motions.
- The goal is to complete as many valid cycles as possible before the round ends.
- A leaderboard stores the best scores locally in `leaderboard.json`.

## Project Structure

```text
monkey_meme/
├── main.py
├── leaderboard.json
├── requirements.txt
├── memes/
│   ├── 67-left-up.png
│   ├── 67-right-up.png
│   ├── pointing.jpg
│   ├── shocked.jpg
│   ├── staring.jpg
│   └── thinking.jpg
├── recordings/
├── screenshots/
├── tests/
│   └── test_six_seven.py
└── README.md
```

## Notes

- The app expects a camera to be available at index `0` by default.
- Some controls modify the output layout or visuals and are designed for a fun live-demo experience rather than strict game UX.
- The app writes recordings and screenshots into the project folders automatically.

## License

This project does not currently include a formal license file. If you plan to distribute or reuse it, add a license before publishing.
