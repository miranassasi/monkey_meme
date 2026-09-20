import os
import json
import math
import time
import random
from collections import deque
import cv2
import numpy as np
import mediapipe as mp


WINDOW_NAME = "Meme Mirror"
TOUCH_THRESHOLD = 0.08
MOUTH_OPEN_THRESHOLD = 0.5
POINTING_OFFSET = 0.05
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
ANALYSIS_FRAME_WIDTH = 320
ANALYSIS_FRAME_HEIGHT = 240

SIXTY_SEVEN_MEME_PATHS = {
    "left_up": "memes/67-left-up.png",
    "right_up": "memes/67-right-up.png",
}
SIXTY_SEVEN_ACTIVE_PHASES = {"left_up", "right_up"}
SIXTY_SEVEN_ROUND_SECONDS = 30.0
SIXTY_SEVEN_RESULT_SECONDS = 5.0
SIXTY_SEVEN_COUNTDOWN_STEP_SECONDS = 0.7
SIXTY_SEVEN_COUNTDOWN_SECONDS = SIXTY_SEVEN_COUNTDOWN_STEP_SECONDS * 3
SIXTY_SEVEN_MOVEMENT_THRESHOLD = 0.04
SIXTY_SEVEN_MIN_VISIBILITY = 0.45
SIXTY_SEVEN_RESET_HOLD_SECONDS = 0.5
SIXTY_SEVEN_SEPARATION_THRESHOLD = 0.06
SIXTY_SEVEN_COLOR_START = 1
SIXTY_SEVEN_COLOR_TARGET = 400
SIXTY_SEVEN_RED = (0, 0, 255)
SIXTY_SEVEN_ORANGE = (0, 165, 255)
SIXTY_SEVEN_GREEN = (0, 255, 0)
LEADERBOARD_PATH = "leaderboard.json"
LEADERBOARD_LIMIT = 10
LEADERBOARD_DISPLAY_LIMIT = 5

MEME_PATHS = {
    "thinking": "memes/thinking.jpg",
    "pointing": "memes/pointing.jpg",
    "shocked": "memes/shocked.jpg",
    "staring": "memes/staring.jpg",
}

POSE_COLORS = {
    "staring": (90, 90, 90),
    "thinking": (0, 180, 255),
    "pointing": (0, 255, 120),
    "shocked": (80, 80, 255),
}

ROAST_MESSAGES = [
    "ur not even trying",
    "my grandma poses better",
    "is that a pose or a cry for help",
    "the meme is embarrassed for u",
    "AI judging u rn",
    "touch grass immediately",
    "ur skeleton is disappointed",
    "webcam quality matches ur effort",
    "pose like u mean it coward",
    "even the pixels are cringing",
    "delete this from ur memory",
    "ur aura is beige",
    "main character? more like NPC",
    "the algorithm has seen enough",
    "ur giving nothing",
]


class AppState:
    def __init__(self):
        self.debug_mode = False
        self.fullscreen = False
        self.meme_only = False
        self.paused = False
        self.side_by_side = False
        self.pip_mode = False
        self.zoom_face = False
        self.recording = False
        self.watermark = False
        self.vertical_mode = False
        self.face_crop = False
        self.smooth_transition = True
        self.chaos_mode = False
        self.show_help = False

        self.streak = 0
        self.transition_alpha = 1.0
        self.current_meme = None
        self.manual_pose_index = -1
        self.camera_index = 0
        self.previous_meme = None
        self.blend_alpha = 1.0
        self.flash_intensity = 0
        self.video_writer = None
        self.gif_buffer = deque(maxlen=90)
        self.pose_history = []
        self.last_detected_pose = "staring"
        self.frame_timestamp = time.time()
        self.session_start = time.time()
        self.pose_counts = {"thinking": 0, "pointing": 0, "shocked": 0, "staring": 0}
        self.cringe_score = 0
        self.ego_deaths = 0
        self.last_roast = ""
        self.roast_timer = 0
        self.frame_count = 0
        self.ui_pulse = 0

        self.challenge_status = "idle"
        self.challenge_countdown_started_at = None
        self.challenge_started_at = None
        self.challenge_result_started_at = None
        self.challenge_reset_started_at = None
        self.challenge_start_phase = None
        self.challenge_current_phase = None
        self.challenge_last_relative_position = None
        self.challenge_left_motion_state = "neutral"
        self.challenge_right_motion_state = "neutral"
        self.challenge_count = 0
        self.challenge_final_count = 0
        self.challenge_variant = "left_up"
        self.challenge_leaderboard = []
        self.challenge_leaderboard_rank = None


def calculate_distance(point_a, point_b):
    return np.linalg.norm(np.array(point_a) - np.array(point_b))


def get_finger_position(hand_landmarks, finger_index=8):
    if hand_landmarks is None:
        return None
    landmark = hand_landmarks.landmark[finger_index]
    return [landmark.x, landmark.y, getattr(landmark, "visibility", 1.0)]


def calculate_mouth_openness(face_landmarks):
    if face_landmarks is None:
        return 0
    landmarks = face_landmarks.landmark
    upper_lip = [landmarks[13].x, landmarks[13].y]
    lower_lip = [landmarks[14].x, landmarks[14].y]
    left_corner = [landmarks[78].x, landmarks[78].y]
    right_corner = [landmarks[308].x, landmarks[308].y]
    vertical_distance = calculate_distance(upper_lip, lower_lip)
    horizontal_distance = calculate_distance(left_corner, right_corner)
    if horizontal_distance == 0:
        return 0
    return vertical_distance / horizontal_distance


def apply_glitch_effect(text):
    glitch_chars = "@#$%&*!?"
    result = ""
    for char in text:
        if random.random() > 0.15:
            result += char
        else:
            result += random.choice(glitch_chars)
    return result


def apply_corruption_effect(frame, intensity=0.1):
    if random.random() >= intensity:
        return frame
    height, width = frame.shape[:2]
    start_y = random.randint(0, height - 20)
    end_y = random.randint(start_y, height)
    if start_y < end_y:
        horizontal_shift = random.randint(-30, 30)
        frame[start_y:end_y] = np.roll(frame[start_y:end_y], horizontal_shift, axis=1)
    return frame


def draw_corner_brackets(image, x1, y1, x2, y2, color, corner_length=8, thickness=2):
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)


def draw_scanline_overlay(image, line_spacing=4, overlay_alpha=0.03):
    overlay = image.copy()
    height = image.shape[0]
    for y_position in range(0, height, line_spacing):
        cv2.line(overlay, (0, y_position), (image.shape[1], y_position), (0, 0, 0), 1)
    cv2.addWeighted(overlay, overlay_alpha, image, 1 - overlay_alpha, 0, image)


def get_face_bounding_box(face_landmarks, frame_width, frame_height, padding=0.3):
    if face_landmarks is None:
        return None
    x_coords = [lm.x * frame_width for lm in face_landmarks.landmark]
    y_coords = [lm.y * frame_height for lm in face_landmarks.landmark]
    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)
    face_width = max_x - min_x
    face_height = max_y - min_y
    pad_x = face_width * padding
    pad_y = face_height * padding
    return (
        int(max(0, min_x - pad_x)),
        int(max(0, min_y - pad_y)),
        int(min(frame_width, max_x + pad_x)),
        int(min(frame_height, max_y + pad_y)),
    )


def blend_face_onto_meme(frame, meme_image, face_bbox, target_region):
    if face_bbox is None:
        return meme_image

    face_x1, face_y1, face_x2, face_y2 = face_bbox
    target_x1, target_y1, target_x2, target_y2 = target_region

    face_crop = frame[face_y1:face_y2, face_x1:face_x2]
    if face_crop.size == 0:
        return meme_image

    target_width = target_x2 - target_x1
    target_height = target_y2 - target_y1
    if target_width <= 0 or target_height <= 0:
        return meme_image

    resized_face = cv2.resize(face_crop, (target_width, target_height))
    result = meme_image.copy()

    ellipse_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    center = (target_width // 2, target_height // 2)
    axes = (target_width // 2 - 5, target_height // 2 - 5)
    cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)
    ellipse_mask = cv2.GaussianBlur(ellipse_mask, (21, 21), 0)

    mask_3channel = cv2.merge([ellipse_mask] * 3) / 255.0
    region_of_interest = result[target_y1:target_y2, target_x1:target_x2]
    blended = (resized_face * mask_3channel + region_of_interest * (1 - mask_3channel))
    result[target_y1:target_y2, target_x1:target_x2] = blended.astype(np.uint8)

    return result


def load_image_set(image_paths):
    images = {}
    for image_name, file_path in image_paths.items():
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Cannot load meme image at {file_path}")
            exit(1)
        images[image_name] = image
    return images


def load_meme_images():
    return load_image_set(MEME_PATHS)


def load_challenge_images():
    return load_image_set(SIXTY_SEVEN_MEME_PATHS)


def fit_image_to_region(image, target_width, target_height, background=(8, 8, 8)):
    """Fit an image inside a region without cropping its subject."""
    if image is None or target_width <= 0 or target_height <= 0:
        return image

    image_height, image_width = image.shape[:2]
    scale = min(target_width / image_width, target_height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    result = np.full((target_height, target_width, 3), background, dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    result[offset_y : offset_y + resized_height,
           offset_x : offset_x + resized_width] = resized
    return result


def setup_directories():
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)


def load_leaderboard(path=LEADERBOARD_PATH):
    """Load the local 67 scores, ignoring a missing or damaged score file."""
    try:
        with open(path, "r", encoding="utf-8") as leaderboard_file:
            raw_entries = json.load(leaderboard_file)
    except (OSError, json.JSONDecodeError, TypeError):
        return []

    if not isinstance(raw_entries, list):
        return []

    entries = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        try:
            score = max(0, int(raw_entry.get("score", 0)))
        except (TypeError, ValueError):
            continue
        entries.append({
            "score": score,
            "timestamp": str(raw_entry.get("timestamp", "")),
        })

    entries.sort(key=lambda entry: entry["score"], reverse=True)
    return entries[:LEADERBOARD_LIMIT]


def save_leaderboard(entries, path=LEADERBOARD_PATH):
    """Persist the local leaderboard without requiring an online account."""
    parent_directory = os.path.dirname(path)
    if parent_directory:
        os.makedirs(parent_directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as leaderboard_file:
        json.dump(entries[:LEADERBOARD_LIMIT], leaderboard_file, indent=2)
        leaderboard_file.write("\n")


def record_six_seven_score(score, path=LEADERBOARD_PATH):
    """Add one result and return the sorted board plus this result's rank."""
    entries = load_leaderboard(path)
    new_entry = {
        "score": max(0, int(score)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    entries.append(new_entry)
    entries.sort(key=lambda entry: entry["score"], reverse=True)
    rank = next(index for index, entry in enumerate(entries) if entry is new_entry) + 1
    entries = entries[:LEADERBOARD_LIMIT]
    save_leaderboard(entries, path)
    return entries, rank


def configure_capture(capture):
    """Keep camera frames small and fresh enough for real-time MediaPipe tracking."""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def create_pose_detector():
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        smooth_landmarks=True,
        model_complexity=0,
        refine_face_landmarks=False,
    )


def create_challenge_detector():
    """Create the lightweight pose tracker used by the fast 67 mode."""
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def show_splash_screen(meme_images, challenge_images):
    screen_width, screen_height = 800, 600
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for frame_index in range(45):
        canvas[:] = (8, 8, 8)

        for y in range(0, screen_height, 4):
            fade_amount = 0.015
            row = canvas[y : y + 1, :]
            canvas[y : y + 1, :] = np.clip(row.astype(float) * (1 - fade_amount), 0, 255).astype(np.uint8)

        pulse_value = (math.sin(frame_index * 0.25) + 1) / 2
        red_intensity = int(180 + 75 * pulse_value)

        text = "WARNING"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)
        text_x = screen_width // 2 - text_w // 2
        cv2.putText(canvas, text, (text_x, screen_height // 2 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, red_intensity), 2)

        subtitle = "flashing lights / photosensitive seizure"
        (sub_w, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(canvas, subtitle, (screen_width // 2 - sub_w // 2, screen_height // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

        box_color = (0, 0, red_intensity)
        draw_corner_brackets(canvas, screen_width // 2 - 240, screen_height // 2 - 75,
                             screen_width // 2 + 240, screen_height // 2 + 45,
                             box_color, corner_length=20, thickness=2)

        hint = "[ Q ] quit    [ SPACE ] continue"
        (hint_w, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, hint, (screen_width // 2 - hint_w // 2, screen_height // 2 + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            return False
        if key == ord(" ") and frame_index > 8:
            break

    for frame_index in range(80):
        canvas[:] = (5, 5, 5)
        draw_scanline_overlay(canvas, line_spacing=3, overlay_alpha=0.02)

        if frame_index < 25:
            progress = frame_index / 25
            text_scale = 0.3 + progress * 1.2
            text_alpha = min(1, progress * 2)
            text = "MEME MIRROR"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
            brightness = int(255 * text_alpha)
            cv2.putText(canvas, text, (screen_width // 2 - text_w // 2, screen_height // 2 + text_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (brightness, brightness, brightness), 2)

        elif frame_index < 50:
            phase = (frame_index - 25) / 25
            glitch_offset = int(4 * math.sin(phase * math.pi * 4)) if frame_index < 40 else 0

            (meme_w, _), _ = cv2.getTextSize("MEME", cv2.FONT_HERSHEY_SIMPLEX, 2.2, 3)
            (mirror_w, _), _ = cv2.getTextSize("MIRROR", cv2.FONT_HERSHEY_SIMPLEX, 2.2, 3)

            meme_x = screen_width // 2 - meme_w // 2
            mirror_x = screen_width // 2 - mirror_w // 2

            if frame_index < 38:
                cv2.putText(canvas, "MEME", (meme_x + glitch_offset, screen_height // 2 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 80, 80), 3)
                cv2.putText(canvas, "MEME", (meme_x - glitch_offset, screen_height // 2 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (80, 255, 255), 3)
            cv2.putText(canvas, "MEME", (meme_x, screen_height // 2 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)

            if frame_index < 38:
                cv2.putText(canvas, "MIRROR", (mirror_x + glitch_offset, screen_height // 2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 80, 80), 3)
                cv2.putText(canvas, "MIRROR", (mirror_x - glitch_offset, screen_height // 2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (80, 255, 255), 3)
            cv2.putText(canvas, "MIRROR", (mirror_x, screen_height // 2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)

            draw_corner_brackets(canvas, screen_width // 2 - 180, screen_height // 2 - 70,
                                 screen_width // 2 + 180, screen_height // 2 + 90,
                                 (255, 255, 255), corner_length=15, thickness=2)

        elif frame_index < 70:
            progress = (frame_index - 50) / 20
            bar_y = screen_height // 2 + 30
            bar_x = screen_width // 2 - 150
            bar_width = int(progress * 300)

            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + 3), (255, 255, 255), -1)
            draw_corner_brackets(canvas, bar_x - 5, bar_y - 5, bar_x + 305, bar_y + 8,
                                 (80, 80, 80), corner_length=4, thickness=1)

            loading_messages = ["loading vibes", "calibrating cringe", "summoning memes",
                                "processing ego", "init chaos"]
            loading_text = random.choice(loading_messages)
            (load_w, _), _ = cv2.getTextSize(loading_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(canvas, loading_text, (screen_width // 2 - load_w // 2, bar_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

            percent_text = f"{int(progress * 100)}%"
            cv2.putText(canvas, percent_text, (bar_x + 308, bar_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)

        else:
            flash_value = 255 if frame_index % 2 == 0 else 0
            canvas[:] = (flash_value, flash_value, flash_value)
            go_text = random.choice(["GO", "→", "POSE"])
            (go_w, go_h), _ = cv2.getTextSize(go_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)
            text_color = (0, 0, 0) if flash_value else (255, 255, 255)
            cv2.putText(canvas, go_text, (screen_width // 2 - go_w // 2, screen_height // 2 + go_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, text_color, 4)

        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(45) & 0xFF == ord("q"):
            return False

    tutorial_slides = [
        ("STARING", "just look at camera", "staring", (90, 90, 90)),
        ("THINKING", "finger on chin", "thinking", (0, 180, 255)),
        ("POINTING", "point up above head", "pointing", (0, 255, 120)),
        ("SHOCKED", "open mouth wide", "shocked", (80, 80, 255)),
        ("67", "press G, then pump both arms for 30 seconds", "67", SIXTY_SEVEN_ORANGE),
    ]

    for slide_index, (title, description, pose_key, color) in enumerate(tutorial_slides):
        canvas[:] = (10, 10, 10)
        draw_scanline_overlay(canvas, line_spacing=3, overlay_alpha=0.02)

        (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(canvas, title, (screen_width // 2 - title_w // 2, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        (desc_w, _), _ = cv2.getTextSize(description, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(canvas, description, (screen_width // 2 - desc_w // 2, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        if pose_key in meme_images:
            thumbnail = cv2.resize(meme_images[pose_key], (300, 170))
            thumb_x = screen_width // 2 - 150
            thumb_y = 160
            canvas[thumb_y : thumb_y + 170, thumb_x : thumb_x + 300] = thumbnail
            draw_corner_brackets(canvas, thumb_x - 2, thumb_y - 2, thumb_x + 302, thumb_y + 172,
                                 color, corner_length=12, thickness=2)
        elif pose_key == "67":
            thumb_y = 160
            thumb_width = 145
            thumb_height = 170
            gap = 10
            thumb_x = screen_width // 2 - (thumb_width * 2 + gap) // 2
            for index, variant in enumerate(("left_up", "right_up")):
                thumbnail = fit_image_to_region(
                    challenge_images[variant], thumb_width, thumb_height
                )
                x_position = thumb_x + index * (thumb_width + gap)
                canvas[thumb_y : thumb_y + thumb_height,
                       x_position : x_position + thumb_width] = thumbnail
                draw_corner_brackets(
                    canvas, x_position - 2, thumb_y - 2,
                    x_position + thumb_width + 2, thumb_y + thumb_height + 2,
                    color, corner_length=10, thickness=2
                )

        page_text = f"{slide_index + 1}/{len(tutorial_slides) + 1}"
        cv2.putText(canvas, page_text, (screen_width - 60, screen_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        cv2.putText(canvas, "SPACE to continue", (screen_width // 2 - 80, screen_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        cv2.imshow(WINDOW_NAME, canvas)
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                return False
            if key == ord(" "):
                break

    canvas[:] = (10, 10, 10)
    draw_scanline_overlay(canvas, line_spacing=3, overlay_alpha=0.02)

    (controls_w, _), _ = cv2.getTextSize("CONTROLS", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cv2.putText(canvas, "CONTROLS", (screen_width // 2 - controls_w // 2, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)

    control_list = [
        "H - show/hide help menu",
        "A - toggle face crop",
        "V - vertical mode (9:16)",
        "W - watermark on/off",
        "S - screenshot",
        "R - start/stop recording",
        "F - fullscreen",
        "G - start 67 challenge",
        "Q - quit",
    ]
    for i, control in enumerate(control_list):
        cv2.putText(canvas, control, (screen_width // 2 - 120, 110 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.putText(canvas, "6/6", (screen_width - 60, screen_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(canvas, "SPACE to start", (screen_width // 2 - 70, screen_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    cv2.imshow(WINDOW_NAME, canvas)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            return False
        if key == ord(" "):
            break

    canvas[:] = (0, 0, 0)
    (ready_w, ready_h), _ = cv2.getTextSize("READY", cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
    cv2.putText(canvas, "READY", (screen_width // 2 - ready_w // 2, screen_height // 2 + ready_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
    cv2.imshow(WINDOW_NAME, canvas)
    cv2.waitKey(500)

    return True


def detect_current_pose(pose_results, frame_width, frame_height):
    detected_pose = "staring"
    should_flip = False

    if pose_results.pose_landmarks is None:
        return detected_pose, should_flip

    landmarks = pose_results.pose_landmarks.landmark
    nose_position = [landmarks[0].x, landmarks[0].y]
    left_mouth = [landmarks[9].x, landmarks[9].y]
    right_mouth = [landmarks[10].x, landmarks[10].y]
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2

    right_finger = get_finger_position(pose_results.right_hand_landmarks)
    left_finger = get_finger_position(pose_results.left_hand_landmarks)
    mouth_ratio = calculate_mouth_openness(pose_results.face_landmarks)

    is_thinking = False

    if right_finger is not None:
        distance_to_mouth = calculate_distance(right_finger[:2], right_mouth)
        if distance_to_mouth < TOUCH_THRESHOLD and right_finger[1] < shoulder_y:
            detected_pose = "thinking"
            should_flip = True
            is_thinking = True

    if left_finger is not None and not is_thinking:
        distance_to_mouth = calculate_distance(left_finger[:2], left_mouth)
        if distance_to_mouth < TOUCH_THRESHOLD and left_finger[1] < shoulder_y:
            detected_pose = "thinking"
            should_flip = False
            is_thinking = True

    if not is_thinking:
        if right_finger is not None and right_finger[1] < nose_position[1] - POINTING_OFFSET:
            detected_pose = "pointing"
            should_flip = False
        elif left_finger is not None and left_finger[1] < nose_position[1] - POINTING_OFFSET:
            detected_pose = "pointing"
            should_flip = True

    if detected_pose == "staring" and mouth_ratio > MOUTH_OPEN_THRESHOLD:
        detected_pose = "shocked"

    return detected_pose, should_flip


def get_challenge_hand_points(pose_results):
    """Return the two visible wrists ordered by their mirrored screen position."""
    hand_points = []
    for hand_landmarks in (
        getattr(pose_results, "left_hand_landmarks", None),
        getattr(pose_results, "right_hand_landmarks", None),
    ):
        if hand_landmarks is None or not hand_landmarks.landmark:
            continue
        wrist = hand_landmarks.landmark[0]
        hand_points.append((float(wrist.x), float(wrist.y)))

    if len(hand_points) < 2:
        return None

    hand_points.sort(key=lambda point: point[0])
    return {"left": hand_points[0], "right": hand_points[-1]}


def detect_six_seven_phase(pose_results):
    """Detect which side of the mirrored preview has the higher hand."""
    hand_points = get_challenge_hand_points(pose_results)
    if hand_points is None:
        return None

    left_y = hand_points["left"][1]
    right_y = hand_points["right"][1]
    vertical_difference = left_y - right_y

    if vertical_difference <= -SIXTY_SEVEN_SEPARATION_THRESHOLD:
        return "left_up"
    if vertical_difference >= SIXTY_SEVEN_SEPARATION_THRESHOLD:
        return "right_up"
    return "neutral"


def _landmark_is_visible(landmark, minimum=SIXTY_SEVEN_MIN_VISIBILITY):
    return landmark is not None and getattr(landmark, "visibility", 1.0) >= minimum


def detect_six_seven_motion(pose_results):
    """Read rough arm motion from pose landmarks without requiring a gesture.

    Each wrist is measured relative to its own shoulder. That makes the signal
    work for people at different distances from the camera and lets the game
    react to movement instead of waiting for a precise hand pose.
    """
    pose_landmarks = getattr(pose_results, "pose_landmarks", None)
    if pose_landmarks is None or not pose_landmarks.landmark:
        return None

    landmarks = pose_landmarks.landmark
    landmark_indices = mp.solutions.pose.PoseLandmark
    left_shoulder = landmarks[landmark_indices.LEFT_SHOULDER.value]
    right_shoulder = landmarks[landmark_indices.RIGHT_SHOULDER.value]
    left_wrist = landmarks[landmark_indices.LEFT_WRIST.value]
    right_wrist = landmarks[landmark_indices.RIGHT_WRIST.value]

    required_landmarks = (left_shoulder, right_shoulder, left_wrist, right_wrist)
    if not all(_landmark_is_visible(landmark) for landmark in required_landmarks):
        return None

    screen_wrists = sorted(
        ((float(left_wrist.x), float(left_wrist.y)),
         (float(right_wrist.x), float(right_wrist.y))),
        key=lambda point: point[0],
    )
    vertical_difference = screen_wrists[0][1] - screen_wrists[-1][1]
    if vertical_difference <= -SIXTY_SEVEN_SEPARATION_THRESHOLD:
        phase = "left_up"
    elif vertical_difference >= SIXTY_SEVEN_SEPARATION_THRESHOLD:
        phase = "right_up"
    else:
        phase = "neutral"

    return {
        "left_relative_y": float(left_wrist.y - left_shoulder.y),
        "right_relative_y": float(right_wrist.y - right_shoulder.y),
        "left_wrist": (float(left_wrist.x), float(left_wrist.y)),
        "right_wrist": (float(right_wrist.x), float(right_wrist.y)),
        "phase": phase,
    }


def _reset_six_seven_motion_state(state, relative_position=None):
    state.challenge_last_relative_position = relative_position
    state.challenge_left_motion_state = "neutral"
    state.challenge_right_motion_state = "neutral"


def begin_six_seven_countdown(state, now=None):
    """Start the visible 3-2-1 countdown before a round begins."""
    if now is None:
        now = time.monotonic()

    state.challenge_status = "countdown"
    state.challenge_countdown_started_at = now
    state.challenge_started_at = None
    state.challenge_result_started_at = None
    state.challenge_reset_started_at = None
    state.challenge_start_phase = "left_up"
    state.challenge_current_phase = "left_up"
    state.challenge_count = 0
    state.challenge_final_count = 0
    state.challenge_variant = "left_up"
    state.challenge_leaderboard = []
    state.challenge_leaderboard_rank = None
    _reset_six_seven_motion_state(state)


def start_six_seven_round(state, start_phase, now, motion=None):
    state.challenge_status = "active"
    state.challenge_countdown_started_at = None
    state.challenge_started_at = now
    state.challenge_result_started_at = None
    state.challenge_reset_started_at = None
    state.challenge_start_phase = start_phase
    state.challenge_current_phase = start_phase
    state.challenge_count = 0
    state.challenge_final_count = 0
    state.challenge_variant = start_phase
    state.challenge_leaderboard = []
    state.challenge_leaderboard_rank = None
    if motion is None:
        _reset_six_seven_motion_state(state)
    else:
        _reset_six_seven_motion_state(
            state,
            (motion["left_relative_y"], motion["right_relative_y"]),
        )


def finish_six_seven_round(state, now):
    state.challenge_status = "result"
    state.challenge_result_started_at = now
    state.challenge_final_count = state.challenge_count
    state.challenge_leaderboard, state.challenge_leaderboard_rank = record_six_seven_score(
        state.challenge_final_count,
        LEADERBOARD_PATH,
    )


def cancel_six_seven_challenge(state):
    """Leave 67 mode immediately and clear its round state."""
    state.challenge_status = "idle"
    state.challenge_countdown_started_at = None
    state.challenge_started_at = None
    state.challenge_result_started_at = None
    state.challenge_reset_started_at = None
    state.challenge_start_phase = None
    state.challenge_current_phase = None
    _reset_six_seven_motion_state(state)
    state.challenge_count = 0
    state.challenge_final_count = 0
    state.challenge_variant = "left_up"
    state.challenge_leaderboard = []
    state.challenge_leaderboard_rank = None


def update_six_seven_challenge(state, hand_phase, motion=None, now=None):
    """Advance the 67 state machine using rough frame-to-frame arm motion."""
    if now is None:
        now = time.monotonic()

    if state.challenge_status == "idle":
        # 67 is started with G, so no special activation pose is required.
        return

    if state.challenge_status == "countdown":
        if now - state.challenge_countdown_started_at >= SIXTY_SEVEN_COUNTDOWN_SECONDS:
            start_phase = (
                hand_phase
                if hand_phase in SIXTY_SEVEN_ACTIVE_PHASES
                else state.challenge_start_phase
            )
            start_six_seven_round(state, start_phase, now, motion)
        return

    if state.challenge_status == "active":
        if motion is not None:
            if hand_phase in SIXTY_SEVEN_ACTIVE_PHASES:
                state.challenge_current_phase = hand_phase
                state.challenge_variant = hand_phase

            current_position = (
                motion["left_relative_y"], motion["right_relative_y"]
            )
            previous_position = state.challenge_last_relative_position
            if previous_position is None:
                _reset_six_seven_motion_state(state, current_position)
            else:
                left_delta = current_position[0] - previous_position[0]
                right_delta = current_position[1] - previous_position[1]

                if left_delta < -SIXTY_SEVEN_MOVEMENT_THRESHOLD:
                    state.challenge_left_motion_state = "up"
                elif left_delta > SIXTY_SEVEN_MOVEMENT_THRESHOLD:
                    state.challenge_left_motion_state = "down"

                if right_delta < -SIXTY_SEVEN_MOVEMENT_THRESHOLD:
                    state.challenge_right_motion_state = "up"
                elif right_delta > SIXTY_SEVEN_MOVEMENT_THRESHOLD:
                    state.challenge_right_motion_state = "down"

                state.challenge_last_relative_position = current_position

                if (
                    state.challenge_left_motion_state != "neutral"
                    and state.challenge_right_motion_state != "neutral"
                ):
                    state.challenge_count += 1
                    state.challenge_left_motion_state = "neutral"
                    state.challenge_right_motion_state = "neutral"
        else:
            _reset_six_seven_motion_state(state)

        if now - state.challenge_started_at >= SIXTY_SEVEN_ROUND_SECONDS:
            finish_six_seven_round(state, now)
        return

    if state.challenge_status == "result":
        if hand_phase in (None, "neutral"):
            if state.challenge_reset_started_at is None:
                state.challenge_reset_started_at = now
        else:
            state.challenge_reset_started_at = None

        if now - state.challenge_result_started_at >= SIXTY_SEVEN_RESULT_SECONDS:
            if (
                state.challenge_reset_started_at is not None
                and now - state.challenge_reset_started_at >= SIXTY_SEVEN_RESET_HOLD_SECONDS
            ):
                state.challenge_status = "idle"
                state.challenge_reset_started_at = None
            else:
                state.challenge_status = "reset"
        return

    if state.challenge_status == "reset":
        if hand_phase in (None, "neutral"):
            if state.challenge_reset_started_at is None:
                state.challenge_reset_started_at = now
            elif now - state.challenge_reset_started_at >= SIXTY_SEVEN_RESET_HOLD_SECONDS:
                state.challenge_status = "idle"
                state.challenge_reset_started_at = None
        else:
            state.challenge_reset_started_at = None


def blend_challenge_color(start_color, end_color, progress):
    progress = min(1.0, max(0.0, progress))
    return tuple(
        int(round(start_color[index] + (end_color[index] - start_color[index]) * progress))
        for index in range(3)
    )


def six_seven_color_for_count(count):
    """Blend red -> orange -> green from cycle 1 through cycle 400."""
    progress = (
        (max(0.0, count) - SIXTY_SEVEN_COLOR_START)
        / (SIXTY_SEVEN_COLOR_TARGET - SIXTY_SEVEN_COLOR_START)
    )
    progress = min(1.0, max(0.0, progress))
    if progress <= 0.5:
        return blend_challenge_color(SIXTY_SEVEN_RED, SIXTY_SEVEN_ORANGE, progress * 2)
    return blend_challenge_color(SIXTY_SEVEN_ORANGE, SIXTY_SEVEN_GREEN, (progress - 0.5) * 2)


def draw_debug_overlay(canvas, pose_results, frame_width, frame_height, mouth_ratio):
    cv2.putText(canvas, f"mouth:{mouth_ratio:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if pose_results.face_landmarks is not None:
        face_lm = pose_results.face_landmarks.landmark
        upper_lip_pos = (int(face_lm[13].x * frame_width), int(face_lm[13].y * frame_height))
        lower_lip_pos = (int(face_lm[14].x * frame_width), int(face_lm[14].y * frame_height))
        cv2.circle(canvas, upper_lip_pos, 5, (0, 255, 0), -1)
        cv2.circle(canvas, lower_lip_pos, 5, (0, 0, 255), -1)

    right_finger = get_finger_position(pose_results.right_hand_landmarks)
    left_finger = get_finger_position(pose_results.left_hand_landmarks)

    if right_finger is not None:
        pos = (int(right_finger[0] * frame_width), int(right_finger[1] * frame_height))
        cv2.circle(canvas, pos, 10, (0, 255, 0), -1)

    if left_finger is not None:
        pos = (int(left_finger[0] * frame_width), int(left_finger[1] * frame_height))
        cv2.circle(canvas, pos, 10, (255, 0, 255), -1)


def draw_skeleton_overlay(canvas, pose_results, drawing_utils):
    body_spec = drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2)
    hand_spec = drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

    drawing_utils.draw_landmarks(
        canvas, pose_results.pose_landmarks,
        mp.solutions.holistic.POSE_CONNECTIONS, body_spec, body_spec
    )
    drawing_utils.draw_landmarks(
        canvas, pose_results.right_hand_landmarks,
        mp.solutions.holistic.HAND_CONNECTIONS, hand_spec, hand_spec
    )
    drawing_utils.draw_landmarks(
        canvas, pose_results.left_hand_landmarks,
        mp.solutions.holistic.HAND_CONNECTIONS, hand_spec, hand_spec
    )


def draw_header_ui(canvas, state, frame_width, current_pose,
                   label_override=None, color_override=None):
    header_height = 50
    cv2.rectangle(canvas, (0, 0), (frame_width, header_height), (10, 10, 10), -1)
    cv2.line(canvas, (0, header_height), (frame_width, header_height), (40, 40, 40), 1)

    title = "MEME MIRROR"
    if state.chaos_mode:
        title = apply_glitch_effect(title)

    (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    if state.chaos_mode:
        brightness = random.randint(180, 255)
    else:
        brightness = int(200 + 55 * state.ui_pulse)
    title_color = (brightness, brightness, brightness)

    title_x = frame_width // 2 - title_w // 2
    cv2.putText(canvas, title, (title_x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, title_color, 2)

    cv2.line(canvas, (title_x - 20, 38), (title_x - 5, 38), (60, 60, 60), 1)
    cv2.line(canvas, (title_x + title_w + 5, 38), (title_x + title_w + 20, 38), (60, 60, 60), 1)

    draw_corner_brackets(canvas, 8, 10, 38, 40, (100, 100, 100), corner_length=6, thickness=1)
    cv2.line(canvas, (15, 17), (31, 33), (150, 150, 150), 1)
    cv2.line(canvas, (31, 17), (15, 33), (150, 150, 150), 1)

    pose_text = (label_override or current_pose).upper()
    if state.chaos_mode:
        pose_text = apply_glitch_effect(pose_text)

    (pose_w, _), _ = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pose_x = frame_width - pose_w - 18

    pose_color = color_override or POSE_COLORS[current_pose]
    glow = int(40 * state.ui_pulse)
    bg_color = (pose_color[0] // 4 + glow, pose_color[1] // 4 + glow, pose_color[2] // 4 + glow)

    cv2.rectangle(canvas, (pose_x - 12, 12), (frame_width - 8, 40), bg_color, -1)
    draw_corner_brackets(canvas, pose_x - 12, 12, frame_width - 8, 40,
                         pose_color, corner_length=5, thickness=1)
    cv2.putText(canvas, pose_text, (pose_x - 2, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return header_height


def draw_footer_ui(canvas, state, frame_width, frame_height, current_pose,
                   color_override=None, progress_override=None):
    bar_progress = progress_override
    if bar_progress is None:
        bar_progress = (state.pose_counts[current_pose] % 100) / 100
    bar_width = int(bar_progress * (frame_width - 16))
    cv2.rectangle(canvas, (8, frame_height - 4), (8 + bar_width, frame_height),
                  color_override or POSE_COLORS[current_pose], -1)
    cv2.rectangle(canvas, (8, frame_height - 4), (frame_width - 8, frame_height), (40, 40, 40), 1)

    draw_corner_brackets(canvas, 2, 2, frame_width - 2, frame_height - 2,
                         (50, 50, 50), corner_length=15, thickness=1)


def draw_streak_indicator(canvas, state, frame_width, header_height):
    if state.streak <= 0:
        return

    streak_text = f"x{state.streak}"
    (text_w, _), _ = cv2.getTextSize(streak_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    streak_intensity = min(1.0, state.streak / 10)
    streak_color = (0, int(255 * streak_intensity), int(255 * (1 - streak_intensity * 0.5)))

    box_x1 = frame_width - text_w - 22
    box_y1 = header_height + 5
    box_x2 = frame_width - 8
    box_y2 = header_height + 30

    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1)
    draw_corner_brackets(canvas, box_x1, box_y1, box_x2, box_y2,
                         streak_color, corner_length=4, thickness=1)
    cv2.putText(canvas, streak_text, (frame_width - text_w - 15, header_height + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, streak_color, 1)


def draw_roast_message(canvas, state, frame_width, frame_height):
    time_since_roast = time.time() - state.roast_timer
    if time_since_roast >= 3 or not state.last_roast:
        return

    fade_factor = 1.0 - time_since_roast / 3
    (text_w, _), _ = cv2.getTextSize(state.last_roast, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)

    text_x = frame_width // 2 - text_w // 2
    text_y = frame_height // 2

    box_color = (int(200 * fade_factor), int(50 * fade_factor), int(50 * fade_factor))
    text_brightness = int(255 * fade_factor)

    cv2.rectangle(canvas, (text_x - 15, text_y - 22), (text_x + text_w + 15, text_y + 8),
                  (10, 10, 10), -1)
    draw_corner_brackets(canvas, text_x - 15, text_y - 22, text_x + text_w + 15, text_y + 8,
                         box_color, corner_length=6, thickness=1)
    cv2.putText(canvas, state.last_roast, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (text_brightness,) * 3, 1)


def draw_six_seven_overlay(output, state, now=None):
    if now is None:
        now = time.monotonic()

    if state.challenge_status == "idle":
        return

    output_height, output_width = output.shape[:2]

    if state.challenge_status == "countdown":
        elapsed = max(0.0, now - state.challenge_countdown_started_at)
        countdown_number = max(
            1,
            3 - int(elapsed / SIXTY_SEVEN_COUNTDOWN_STEP_SECONDS),
        )
        center_x = output_width // 2
        center_y = output_height // 2
        panel_width = min(output_width - 24, 500)
        panel_left = (output_width - panel_width) // 2
        panel_top = max(12, center_y - 160)
        panel_bottom = min(output_height - 12, center_y + 160)
        overlay = np.zeros_like(output)
        cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)
        cv2.rectangle(output, (panel_left, panel_top),
                      (panel_left + panel_width, panel_bottom), (10, 10, 10), -1)
        draw_corner_brackets(output, panel_left, panel_top,
                             panel_left + panel_width, panel_bottom,
                             SIXTY_SEVEN_ORANGE, corner_length=18, thickness=2)

        title = "67 GET READY"
        (title_width, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(output, title, (center_x - title_width // 2, panel_top + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2)
        number_text = str(countdown_number)
        (number_width, number_height), _ = cv2.getTextSize(
            number_text, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 5
        )
        cv2.putText(output, number_text,
                    (center_x - number_width // 2, center_y + number_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, SIXTY_SEVEN_ORANGE, 5)
        cv2.putText(output, "move both arms when the timer starts",
                    (center_x - 145, panel_bottom - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        return

    if state.challenge_status == "active":
        accent = six_seven_color_for_count(state.challenge_count)
        seconds_left = max(
            0.0,
            SIXTY_SEVEN_ROUND_SECONDS - (now - state.challenge_started_at),
        )
        panel_y1, panel_y2 = 58, 168
        overlay = output.copy()
        cv2.rectangle(overlay, (12, panel_y1), (output_width - 12, panel_y2), (8, 8, 8), -1)
        cv2.addWeighted(overlay, 0.82, output, 0.18, 0, output)
        draw_corner_brackets(output, 12, panel_y1, output_width - 12, panel_y2,
                             accent, corner_length=10, thickness=2)

        cv2.putText(output, "67 CHALLENGE", (28, panel_y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1)
        timer_text = f"{seconds_left:04.1f}s"
        (timer_width, _), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        cv2.putText(output, timer_text, (output_width - timer_width - 28, panel_y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, accent, 1)
        cv2.putText(output, str(state.challenge_count), (28, panel_y1 + 91),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.65, accent, 3)
        cv2.putText(output, "cycles", (116, panel_y1 + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        bar_x1, bar_x2, bar_y = 28, output_width - 28, panel_y2 - 15
        cv2.rectangle(output, (bar_x1, bar_y), (bar_x2, bar_y + 6), (45, 45, 45), -1)
        count_progress = min(1.0, state.challenge_count / SIXTY_SEVEN_COLOR_TARGET)
        cv2.rectangle(output, (bar_x1, bar_y),
                      (bar_x1 + int((bar_x2 - bar_x1) * count_progress), bar_y + 6),
                      accent, -1)
        return

    if state.challenge_status == "result":
        accent = six_seven_color_for_count(state.challenge_final_count)
        overlay = np.zeros_like(output)
        cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)
        center_x = output_width // 2
        center_y = output_height // 2
        panel_width = min(output_width - 24, 520)
        panel_left = (output_width - panel_width) // 2
        panel_top = max(12, center_y - 185)
        panel_bottom = min(output_height - 12, center_y + 185)
        cv2.rectangle(output, (panel_left, panel_top),
                      (panel_left + panel_width, panel_bottom), (10, 10, 10), -1)
        draw_corner_brackets(output, panel_left, panel_top,
                             panel_left + panel_width, panel_bottom, accent,
                             corner_length=16, thickness=2)

        title = "67 RESULT"
        (title_width, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(output, title, (center_x - title_width // 2, panel_top + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2)
        score_text = f"{state.challenge_final_count} CYCLES"
        (score_width, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 3)
        cv2.putText(output, score_text, (center_x - score_width // 2, panel_top + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, accent, 3)

        rank_text = (
            f"LOCAL RANK #{state.challenge_leaderboard_rank}"
            if state.challenge_leaderboard_rank is not None
            else "LOCAL RANK --"
        )
        (rank_width, _), _ = cv2.getTextSize(rank_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(output, rank_text, (center_x - rank_width // 2, panel_top + 103),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(output, "LOCAL LEADERBOARD", (panel_left + 24, panel_top + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
        leaderboard_y = panel_top + 153
        for index, entry in enumerate(state.challenge_leaderboard[:LEADERBOARD_DISPLAY_LIMIT]):
            row_color = accent if entry["score"] == state.challenge_final_count else (175, 175, 175)
            row_text = f"{index + 1:>2}. {entry['score']:>4} cycles"
            cv2.putText(output, row_text, (panel_left + 32, leaderboard_y + index * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, row_color, 1)

        cv2.putText(output, "lower hands to play again", (center_x - 130, panel_bottom - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def draw_six_seven_motion_points(canvas, state, motion):
    """Show the two wrist points used by the arm-only counter."""
    if state.challenge_status != "active" or motion is None:
        return

    frame_height, frame_width = canvas.shape[:2]
    points = (
        (motion["left_wrist"], state.challenge_left_motion_state),
        (motion["right_wrist"], state.challenge_right_motion_state),
    )
    for (x, y), motion_state in points:
        center = (
            int(min(1.0, max(0.0, x)) * frame_width),
            int(min(1.0, max(0.0, y)) * frame_height),
        )
        if motion_state == "up":
            color = (0, 165, 255)
        elif motion_state == "down":
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)

        cv2.circle(canvas, center, 22, color, 2)
        cv2.circle(canvas, center, 7, color, -1)


def compose_output_frame(camera_view, meme_view, state, frame_width, frame_height):
    if state.vertical_mode:
        vertical_width = min(frame_width, 540)
        vertical_height = int(vertical_width * 16 / 9)
        camera_resized = preserve_aspect_resize(camera_view, vertical_width, vertical_height // 2)
        meme_resized = preserve_aspect_resize(meme_view, vertical_width, vertical_height // 2)
        return np.vstack([camera_resized, meme_resized])

    if state.meme_only:
        return preserve_aspect_resize(meme_view, frame_width, frame_height)

    if state.side_by_side:
        camera_half = preserve_aspect_resize(camera_view, frame_width // 2, frame_height)
        meme_half = preserve_aspect_resize(meme_view, frame_width // 2, frame_height)
        return np.hstack([camera_half, meme_half])

    if state.pip_mode:
        output = preserve_aspect_resize(meme_view, frame_width, frame_height)
        pip_width = frame_width // 4
        pip_height = frame_height // 4
        pip_view = preserve_aspect_resize(camera_view, pip_width, pip_height)
        pip_x = frame_width - pip_width - 10
        output[10 : 10 + pip_height, pip_x : pip_x + pip_width] = pip_view
        cv2.rectangle(output, (pip_x, 10), (frame_width - 10, 10 + pip_height), (255, 255, 255), 2)
        return output

    # โหมดเริ่มต้น (บน-ล่าง) แก้ให้ทั้งภาพกล้องและมีมตัดขอบแทนการบีบอัดแบน
    camera_half_height = preserve_aspect_resize(camera_view, frame_width, frame_height // 2)
    meme_half_height = preserve_aspect_resize(meme_view, frame_width, frame_height // 2)
    return np.vstack([camera_half_height, meme_half_height])

def preserve_aspect_resize(image, target_width, target_height):
    """ปรับขนาดภาพโดยรักษาสัดส่วนเดิม และตัดขอบตรงกลาง (Center Crop) ไม่ให้ภาพแบน"""
    if image is None:
        return None
    if target_width <= 0 or target_height <= 0:
        return image

    image_h, image_w = image.shape[:2]

    scale = max(target_width / image_w, target_height / image_h)
    resized_w = max(1, int(round(image_w * scale)))
    resized_h = max(1, int(round(image_h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

    left = max(0, (resized_w - target_width) // 2)
    top = max(0, (resized_h - target_height) // 2)
    right = left + target_width
    bottom = top + target_height

    return resized[top:bottom, left:right]


def compose_output_frame(camera_view, meme_view, state, frame_width, frame_height):
    if state.vertical_mode:
        vertical_width = min(frame_width, 540)
        vertical_height = int(vertical_width * 16 / 9)
        camera_resized = preserve_aspect_resize(camera_view, vertical_width, vertical_height // 2)
        meme_resized = preserve_aspect_resize(meme_view, vertical_width, vertical_height // 2)
        return np.vstack([camera_resized, meme_resized])

    if state.meme_only:
        return preserve_aspect_resize(meme_view, frame_width, frame_height)

    if state.side_by_side:
        camera_half = preserve_aspect_resize(camera_view, frame_width // 2, frame_height)
        meme_half = preserve_aspect_resize(meme_view, frame_width // 2, frame_height)
        return np.hstack([camera_half, meme_half])

    if state.pip_mode:
        output = preserve_aspect_resize(meme_view, frame_width, frame_height)
        pip_width = frame_width // 4
        pip_height = frame_height // 4
        pip_view = preserve_aspect_resize(camera_view, pip_width, pip_height)
        pip_x = frame_width - pip_width - 10
        output[10 : 10 + pip_height, pip_x : pip_x + pip_width] = pip_view
        cv2.rectangle(output, (pip_x, 10), (frame_width - 10, 10 + pip_height), (255, 255, 255), 2)
        return output

    # โหมดเริ่มต้น (บน-ล่าง สมมาตร ไม่บวม)
    camera_half_height = preserve_aspect_resize(camera_view, frame_width, frame_height // 2)
    meme_half_height = preserve_aspect_resize(meme_view, frame_width, frame_height // 2)
    return np.vstack([camera_half_height, meme_half_height])

def draw_status_indicators(output, state):
    output_height, output_width = output.shape[:2]

    fps_value = 1 / (time.time() - state.frame_timestamp + 0.001)
    fps_text = f"{int(fps_value)}"
    cv2.putText(output, fps_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    cv2.putText(output, "fps", (10 + len(fps_text) * 9, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (35, 35, 35), 1)

    indicator_y = 28

    if state.manual_pose_index >= 0:
        cv2.putText(output, "MANUAL", (10, indicator_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 140, 50), 1)
        indicator_y += 16

    if state.chaos_mode:
        chaos_text = apply_glitch_effect("CHAOS") if random.random() > 0.5 else "CHAOS"
        cv2.putText(output, chaos_text, (10, indicator_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 50, 200), 1)
        indicator_y += 16

    if state.ego_deaths > 0:
        ego_text = f"{state.ego_deaths} ego deaths"
        cv2.putText(output, ego_text, (10, indicator_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 40, 100), 1)

    session_minutes = int((time.time() - state.session_start) / 60)
    if session_minutes >= 5:
        shame_messages = ["wasting time", "go outside", "intervention needed"]
        shame_index = min(2, (session_minutes - 5) // 5)
        shame_text = f"{shame_messages[shame_index]} ({session_minutes}m)"
        cv2.putText(output, shame_text, (output_width - 130, output_height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 40, 40), 1)


def draw_mode_badges(output, state):
    output_height, output_width = output.shape[:2]
    badge_y = 45

    if state.vertical_mode:
        cv2.rectangle(output, (output_width - 85, badge_y - 14), (output_width - 8, badge_y + 4),
                      (80, 40, 120), -1)
        cv2.putText(output, "VERTICAL", (output_width - 80, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 255), 1)
        badge_y += 22

    if state.face_crop:
        cv2.rectangle(output, (output_width - 95, badge_y - 14), (output_width - 8, badge_y + 4),
                      (40, 100, 80), -1)
        cv2.putText(output, "FACE CROP", (output_width - 90, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 200), 1)
        badge_y += 22

    if state.watermark:
        cv2.rectangle(output, (output_width - 55, badge_y - 14), (output_width - 8, badge_y + 4),
                      (60, 60, 60), -1)
        cv2.putText(output, "WM", (output_width - 45, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        badge_y += 22

    if state.smooth_transition:
        cv2.putText(output, "~smooth", (output_width - 65, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)


def draw_watermark(output, state):
    if not state.watermark:
        return

    output_height, output_width = output.shape[:2]
    watermark_text = "MEME MIRROR"
    (text_w, _), _ = cv2.getTextSize(watermark_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    text_x = output_width - text_w - 15
    text_y = output_height - 35

    cv2.putText(output, watermark_text, (text_x + 1, text_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(output, watermark_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_recording_indicator(output, state):
    if not state.recording:
        return

    output_height, output_width = output.shape[:2]
    pulse_value = int(180 + 75 * math.sin(state.frame_count * 0.15))

    cv2.circle(output, (output_width - 20, 20), 6, (0, 0, pulse_value), -1)
    cv2.putText(output, "rec", (output_width - 50, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, pulse_value), 1)


def draw_help_button(output, state):
    output_height, output_width = output.shape[:2]

    cv2.rectangle(output, (12, output_height - 24), (55, output_height - 6), (0, 0, 0), -1)
    cv2.rectangle(output, (12, output_height - 24), (55, output_height - 6), (0, 200, 255), 1)
    cv2.putText(output, "HELP", (17, output_height - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)


def draw_help_menu(output, state):
    if not state.show_help:
        return

    output_height, output_width = output.shape[:2]

    commands = [
        "Q - quit", "H - toggle help", "A - face crop", "V - vertical mode",
        "W - watermark", "N - smooth transition", "S - screenshot", "R - record",
        "Y - save gif", "F - fullscreen", "P - pause", "M - meme only",
        "B - side by side", "I - pip mode", "Z - zoom face",
        "X - chaos mode", "D - debug", "C - switch camera",
        "G - start 67 challenge", "ESC - exit 67 mode",
    ]

    menu_width = 180
    menu_height = len(commands) * 18 + 20
    menu_x = 12
    menu_y = output_height - 30 - menu_height

    cv2.rectangle(output, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height),
                  (0, 0, 0), -1)
    cv2.rectangle(output, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height),
                  (0, 200, 255), 1)

    for i, command in enumerate(commands):
        cv2.putText(output, command, (menu_x + 10, menu_y + 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def handle_keypress(key, state, output, capture, pose_list):
    if key == ord("q"):
        return False
    if key == 27:
        cancel_six_seven_challenge(state)
        return True
    if key == ord("g"):
        if state.challenge_status not in ("countdown", "active"):
            begin_six_seven_countdown(state, time.monotonic())
        return True

    if key == ord("h"):
        state.show_help = not state.show_help
    elif key == ord("d"):
        state.debug_mode = not state.debug_mode
    elif key == ord("s"):
        filename = f"screenshots/meme_{int(time.time())}.png"
        cv2.imwrite(filename, output)
        state.flash_intensity = 255
    elif key == ord("f"):
        state.fullscreen = not state.fullscreen
    elif key == ord("p"):
        state.paused = not state.paused
    elif key == ord("m"):
        state.meme_only = not state.meme_only
    elif key == ord("b"):
        state.side_by_side = not state.side_by_side
        state.pip_mode = False
    elif key == ord("i"):
        state.pip_mode = not state.pip_mode
        state.side_by_side = False
    elif key == ord("z"):
        state.zoom_face = not state.zoom_face
    elif key == ord("x"):
        state.chaos_mode = not state.chaos_mode
    elif key == ord("w"):
        state.watermark = not state.watermark
    elif key == ord("v"):
        state.vertical_mode = not state.vertical_mode
    elif key == ord("n"):
        state.smooth_transition = not state.smooth_transition
    elif key == ord("a"):
        state.face_crop = not state.face_crop
    elif key == ord("c"):
        state.camera_index = (state.camera_index + 1) % 3
        capture.release()
        capture.open(state.camera_index)
        configure_capture(capture)
    elif key == ord("r"):
        state.recording = not state.recording
        if state.recording:
            output_height, output_width = output.shape[:2]
            filename = f"recordings/meme_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            state.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (output_width, output_height))
        elif state.video_writer is not None:
            state.video_writer.release()
            state.video_writer = None
    elif key in (ord("y"), ord("G")):
        for i, frame in enumerate(state.gif_buffer):
            cv2.imwrite(f"screenshots/gif_{i:03d}.png", frame)
        state.flash_intensity = 255
    elif key == ord("t"):
        state.last_roast = random.choice(ROAST_MESSAGES)
        state.roast_timer = time.time()
    elif key in [81, 2]:
        if state.manual_pose_index >= 0:
            state.manual_pose_index = (state.manual_pose_index - 1) % len(pose_list)
        else:
            state.manual_pose_index = len(pose_list) - 1
    elif key in [83, 3]:
        state.manual_pose_index = (state.manual_pose_index + 1) % len(pose_list)
    elif key in [82, 0]:
        state.manual_pose_index = -1

    return True


def main():
    setup_directories()
    meme_images = load_meme_images()
    challenge_images = load_challenge_images()
    pose_list = list(MEME_PATHS.keys())

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Cannot open camera")
        return
    configure_capture(capture)

    if not show_splash_screen(meme_images, challenge_images):
        capture.release()
        cv2.destroyAllWindows()
        return

    pose_detector = create_pose_detector()
    challenge_detector = create_challenge_detector()
    drawing_utils = mp.solutions.drawing_utils
    state = AppState()
    cached_frame_size = None
    sized_memes = None
    sized_challenge_memes = None

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        if state.paused:
            cv2.imshow(WINDOW_NAME, state.last_output if hasattr(state, "last_output") else frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("p"):
                state.paused = False
            if key == ord("q"):
                break
            continue

        frame_size = (frame_width, frame_height)
        if frame_size != cached_frame_size:
            sized_memes = {}
            for pose_name, image in meme_images.items():
                resized = cv2.resize(
                    image, (frame_width, frame_height // 2), interpolation=cv2.INTER_AREA
                )
                sized_memes[pose_name] = {
                    "normal": resized,
                    "flipped": cv2.flip(resized, 1),
                }

            sized_challenge_memes = {
                variant: fit_image_to_region(image, frame_width, frame_height // 2)
                for variant, image in challenge_images.items()
            }
            cached_frame_size = frame_size

        canvas = frame.copy()
        analysis_frame = cv2.resize(
            frame,
            (ANALYSIS_FRAME_WIDTH, ANALYSIS_FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)

        fast_challenge_tracking = state.challenge_status in (
            "countdown", "active", "result"
        )
        if fast_challenge_tracking:
            challenge_pose_results = challenge_detector.process(rgb_frame)
            pose_results = None
            challenge_motion = detect_six_seven_motion(challenge_pose_results)
        else:
            pose_results = pose_detector.process(rgb_frame)
            challenge_motion = detect_six_seven_motion(pose_results)

        challenge_hand_phase = (
            challenge_motion["phase"] if challenge_motion is not None else None
        )

        frame_now = time.monotonic()
        update_six_seven_challenge(
            state, challenge_hand_phase, motion=challenge_motion, now=frame_now
        )
        challenge_visible = state.challenge_status in (
            "countdown", "active", "result"
        )
        draw_six_seven_motion_points(canvas, state, challenge_motion)

        if not challenge_visible and pose_results is None:
            pose_results = pose_detector.process(rgb_frame)

        if challenge_visible:
            detected_pose = "staring"
            should_flip = False
            stable_pose = "staring"
            state.streak = 0
        else:
            detected_pose, should_flip = detect_current_pose(
                pose_results, frame_width, frame_height
            )

            if state.debug_mode and pose_results.pose_landmarks is not None:
                mouth_ratio = calculate_mouth_openness(pose_results.face_landmarks)
                draw_debug_overlay(canvas, pose_results, frame_width, frame_height, mouth_ratio)

            if detected_pose == "staring" and pose_results.pose_landmarks is not None:
                draw_skeleton_overlay(canvas, pose_results, drawing_utils)

            state.pose_history.append(detected_pose)
            if len(state.pose_history) > 5:
                state.pose_history.pop(0)

            stable_pose = detected_pose
            if len(state.pose_history) >= 2:
                if state.pose_history.count(state.pose_history[-1]) >= 2:
                    stable_pose = state.pose_history[-1]

            state.pose_counts[stable_pose] += 1
            total_poses = sum(state.pose_counts.values())

            if stable_pose == "staring" and total_poses > 100:
                state.cringe_score += 0.1
            elif stable_pose != "staring":
                state.cringe_score = max(0, state.cringe_score - 0.5)

            if state.cringe_score > 50 and random.random() < 0.02:
                state.ego_deaths += 1
                state.cringe_score = 0
                state.last_roast = random.choice(ROAST_MESSAGES)
                state.roast_timer = time.time()

        if challenge_visible:
            display_pose = "staring"
            challenge_variant = state.challenge_variant
            current_meme = sized_challenge_memes.get(
                challenge_variant, sized_challenge_memes["left_up"]
            )
            state.streak = 0
            state.last_detected_pose = "staring"
            state.current_meme = current_meme.copy()
            state.previous_meme = current_meme.copy()
            state.transition_alpha = 1.0
            state.blend_alpha = 1.0
        else:
            if state.manual_pose_index >= 0:
                display_pose = pose_list[state.manual_pose_index]
            else:
                display_pose = stable_pose

            meme_variant = "flipped" if should_flip else "normal"
            current_meme = sized_memes[display_pose][meme_variant]

            if state.chaos_mode:
                current_meme = apply_corruption_effect(current_meme.copy(), 0.3)
                if random.random() < 0.1:
                    current_meme = cv2.flip(current_meme, random.choice([-1, 0, 1]))

            if stable_pose != "staring":
                if stable_pose == state.last_detected_pose:
                    state.streak += 1
                else:
                    state.streak = 1
            else:
                state.streak = 0
            state.last_detected_pose = stable_pose

            if state.smooth_transition:
                if state.current_meme is None or state.current_meme.shape != current_meme.shape:
                    state.current_meme = current_meme.copy()
                    state.transition_alpha = 1.0
                elif not np.array_equal(state.current_meme, sized_memes[display_pose][meme_variant]):
                    state.transition_alpha = max(0.0, state.transition_alpha - 0.08)
                    if state.transition_alpha <= 0:
                        state.current_meme = sized_memes[display_pose][meme_variant].copy()
                        state.transition_alpha = 1.0
                    else:
                        current_meme = cv2.addWeighted(
                            state.current_meme, state.transition_alpha,
                            current_meme, 1 - state.transition_alpha, 0
                        )
            else:
                if state.previous_meme is not None and state.previous_meme.shape == current_meme.shape:
                    if state.blend_alpha < 1.0:
                        state.blend_alpha = min(1.0, state.blend_alpha + 0.12)
                        current_meme = cv2.addWeighted(
                            state.previous_meme, 1 - state.blend_alpha,
                            current_meme, state.blend_alpha, 0
                        )
                    elif not np.array_equal(state.previous_meme, current_meme):
                        state.blend_alpha = 0.0
                        state.flash_intensity = 180
            state.previous_meme = sized_memes[display_pose][meme_variant].copy()

        if not challenge_visible and state.face_crop and pose_results.face_landmarks is not None:
            face_bbox = get_face_bounding_box(pose_results.face_landmarks, frame_width, frame_height)
            meme_height, meme_width = current_meme.shape[:2]
            face_target_regions = {
                "thinking": (int(meme_width * 0.25), int(meme_height * 0.08),
                             int(meme_width * 0.75), int(meme_height * 0.58)),
                "pointing": (int(meme_width * 0.2), int(meme_height * 0.08),
                             int(meme_width * 0.65), int(meme_height * 0.55)),
                "shocked": (int(meme_width * 0.25), int(meme_height * 0.05),
                            int(meme_width * 0.75), int(meme_height * 0.55)),
                "staring": (int(meme_width * 0.25), int(meme_height * 0.08),
                            int(meme_width * 0.75), int(meme_height * 0.58)),
            }
            if display_pose in face_target_regions:
                current_meme = blend_face_onto_meme(
                    frame, current_meme, face_bbox, face_target_regions[display_pose]
                )

        if state.flash_intensity > 0:
            flash_overlay = np.full(canvas.shape, state.flash_intensity, dtype=np.uint8)
            canvas = cv2.addWeighted(canvas, 1, flash_overlay, 0.3, 0)
            state.flash_intensity = max(0, state.flash_intensity - 40)

        if state.chaos_mode:
            canvas = apply_corruption_effect(canvas, 0.2)

        state.frame_count += 1
        state.ui_pulse = math.sin(state.frame_count * 0.05) * 0.5 + 0.5

        camera_view = canvas.copy()

        if not challenge_visible and state.zoom_face and pose_results.pose_landmarks is not None:
            nose = pose_results.pose_landmarks.landmark[0]
            center_x = int(nose.x * frame_width)
            center_y = int(nose.y * frame_height)
            zoom_size = min(frame_width, frame_height) // 2

            crop_x1 = max(0, center_x - zoom_size // 2)
            crop_y1 = max(0, center_y - zoom_size // 2)
            crop_x2 = min(frame_width, crop_x1 + zoom_size)
            crop_y2 = min(frame_height, crop_y1 + zoom_size)

            if crop_x2 - crop_x1 > 50 and crop_y2 - crop_y1 > 50:
                camera_view = cv2.resize(canvas[crop_y1:crop_y2, crop_x1:crop_x2],
                                         (frame_width, frame_height))

        challenge_accent = None
        challenge_label = None
        challenge_progress = None
        if challenge_visible:
            challenge_score = (
                state.challenge_final_count
                if state.challenge_status == "result"
                else state.challenge_count
            )
            challenge_accent = six_seven_color_for_count(challenge_score)
            challenge_label = "67"
            challenge_progress = min(1.0, challenge_score / SIXTY_SEVEN_COLOR_TARGET)

        header_height = draw_header_ui(
            camera_view, state, frame_width, display_pose,
            label_override=challenge_label,
            color_override=challenge_accent,
        )
        draw_footer_ui(
            camera_view, state, frame_width, frame_height, display_pose,
            color_override=challenge_accent,
            progress_override=challenge_progress,
        )
        draw_streak_indicator(camera_view, state, frame_width, header_height)
        draw_roast_message(camera_view, state, frame_width, frame_height)

        output = compose_output_frame(camera_view, current_meme, state, frame_width, frame_height)

        state.frame_timestamp = time.time()

        if not state.meme_only:
            draw_scanline_overlay(output, line_spacing=3, overlay_alpha=0.02)

        draw_six_seven_overlay(output, state, frame_now)
        draw_status_indicators(output, state)
        draw_mode_badges(output, state)
        draw_watermark(output, state)

        output_height, output_width = output.shape[:2]
        draw_corner_brackets(output, 4, output_height - 28, output_width - 4, output_height - 4,
                             (40, 40, 40), corner_length=8, thickness=1)

        draw_recording_indicator(output, state)
        draw_help_button(output, state)
        draw_help_menu(output, state)

        if state.recording and state.video_writer is not None:
            state.video_writer.write(output)

        state.gif_buffer.append(output.copy())
        state.last_output = output

        if state.fullscreen:
            cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        cv2.imshow(WINDOW_NAME, output)

        key = cv2.waitKey(5) & 0xFF
        if not handle_keypress(key, state, output, capture, pose_list):
            break

    capture.release()
    if state.video_writer is not None:
        state.video_writer.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    challenge_detector.close()


if __name__ == "__main__":
    main()
