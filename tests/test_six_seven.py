import os
import tempfile
import unittest

import main


class Landmark:
    def __init__(self, x, y, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


class HandLandmarks:
    def __init__(self, x, y):
        self.landmark = [Landmark(x, y)]


class PoseResults:
    def __init__(self, left_x=0.3, left_y=0.3, right_x=0.7, right_y=0.6):
        self.left_hand_landmarks = HandLandmarks(left_x, left_y)
        self.right_hand_landmarks = HandLandmarks(right_x, right_y)


class BodyPoseResults:
    def __init__(self, left_shoulder_y=0.5, right_shoulder_y=0.5,
                 left_wrist_y=0.3, right_wrist_y=0.7):
        landmarks = [Landmark(0.5, 0.5) for _ in range(33)]
        indices = main.mp.solutions.pose.PoseLandmark
        landmarks[indices.LEFT_SHOULDER.value] = Landmark(0.3, left_shoulder_y)
        landmarks[indices.RIGHT_SHOULDER.value] = Landmark(0.7, right_shoulder_y)
        landmarks[indices.LEFT_WRIST.value] = Landmark(0.3, left_wrist_y)
        landmarks[indices.RIGHT_WRIST.value] = Landmark(0.7, right_wrist_y)
        self.pose_landmarks = type("PoseLandmarks", (), {"landmark": landmarks})()


def motion(left_relative_y, right_relative_y, phase="neutral"):
    return {
        "left_relative_y": left_relative_y,
        "right_relative_y": right_relative_y,
        "left_wrist": (0.3, left_relative_y + 0.5),
        "right_wrist": (0.7, right_relative_y + 0.5),
        "phase": phase,
    }


class SixSevenChallengeTests(unittest.TestCase):
    def test_phase_uses_mirrored_screen_side(self):
        self.assertEqual(main.detect_six_seven_phase(PoseResults()), "left_up")
        self.assertEqual(
            main.detect_six_seven_phase(PoseResults(left_y=0.6, right_y=0.3)),
            "right_up",
        )
        self.assertEqual(
            main.detect_six_seven_phase(PoseResults(left_y=0.45, right_y=0.48)),
            "neutral",
        )

    def test_motion_uses_wrist_relative_to_each_shoulder(self):
        result = main.detect_six_seven_motion(
            BodyPoseResults(
                left_shoulder_y=0.5,
                right_shoulder_y=0.5,
                left_wrist_y=0.2,
                right_wrist_y=0.8,
            )
        )
        self.assertAlmostEqual(result["left_relative_y"], -0.3)
        self.assertAlmostEqual(result["right_relative_y"], 0.3)
        self.assertEqual(result["phase"], "left_up")

    def test_full_motion_cycle_counts_once(self):
        state = main.AppState()
        initial = motion(-0.1, 0.1, "left_up")

        main.update_six_seven_challenge(state, "left_up", motion=initial, now=0.0)
        main.start_six_seven_round(state, "left_up", now=0.3, motion=initial)
        main.update_six_seven_challenge(
            state,
            "neutral",
            motion=motion(-0.16, 0.16),
            now=0.4,
        )

        self.assertEqual(state.challenge_count, 1)

    def test_g_starts_countdown_then_round(self):
        state = main.AppState()
        initial = motion(-0.1, 0.1)

        main.begin_six_seven_countdown(state, now=0.0)
        main.update_six_seven_challenge(state, None, motion=initial, now=1.0)
        self.assertEqual(state.challenge_status, "countdown")

        main.update_six_seven_challenge(state, None, motion=initial, now=2.2)
        self.assertEqual(state.challenge_status, "active")
        self.assertEqual(state.challenge_count, 0)
        self.assertEqual(state.challenge_started_at, 2.2)

    def test_single_arm_motion_does_not_count(self):
        state = main.AppState()
        main.start_six_seven_round(state, "left_up", now=0.0, motion=motion(-0.1, 0.1))
        main.update_six_seven_challenge(
            state,
            "neutral",
            motion=motion(-0.16, 0.1),
            now=0.1,
        )

        self.assertEqual(state.challenge_count, 0)

    def test_round_result_and_reset_gate(self):
        state = main.AppState()
        initial = motion(-0.1, 0.1)
        main.start_six_seven_round(state, "left_up", now=0.0, motion=initial)
        main.update_six_seven_challenge(state, None, motion=initial, now=30.1)
        self.assertEqual(state.challenge_status, "result")
        self.assertEqual(state.challenge_final_count, 0)

        main.update_six_seven_challenge(state, "left_up", now=35.2)
        self.assertEqual(state.challenge_status, "reset")
        main.update_six_seven_challenge(state, "neutral", now=35.3)
        main.update_six_seven_challenge(state, "neutral", now=35.9)
        self.assertEqual(state.challenge_status, "idle")

    def test_color_gradient_endpoints(self):
        self.assertEqual(main.six_seven_color_for_count(0), main.SIXTY_SEVEN_RED)
        self.assertEqual(main.six_seven_color_for_count(1), main.SIXTY_SEVEN_RED)
        self.assertEqual(main.six_seven_color_for_count(200), main.SIXTY_SEVEN_ORANGE)
        self.assertEqual(main.six_seven_color_for_count(400), main.SIXTY_SEVEN_GREEN)
        self.assertEqual(main.six_seven_color_for_count(401), main.SIXTY_SEVEN_GREEN)

    def test_local_leaderboard_sorts_and_keeps_top_ten(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            leaderboard_path = os.path.join(temp_directory, "leaderboard.json")
            entries, rank = main.record_six_seven_score(12, leaderboard_path)
            self.assertEqual(entries[0]["score"], 12)
            self.assertEqual(rank, 1)

            entries, rank = main.record_six_seven_score(20, leaderboard_path)
            self.assertEqual([entry["score"] for entry in entries], [20, 12])
            self.assertEqual(rank, 1)

            for score in range(11):
                main.record_six_seven_score(score, leaderboard_path)
            self.assertEqual(len(main.load_leaderboard(leaderboard_path)), 10)
            self.assertEqual(main.load_leaderboard(leaderboard_path)[0]["score"], 20)

    def test_g_starts_countdown_and_escape_cancels(self):
        state = main.AppState()

        self.assertTrue(main.handle_keypress(ord("g"), state, None, None, []))
        self.assertEqual(state.challenge_status, "countdown")

        state.challenge_count = 7
        self.assertTrue(main.handle_keypress(27, state, None, None, []))
        self.assertEqual(state.challenge_status, "idle")
        self.assertEqual(state.challenge_count, 0)
        self.assertIsNone(state.challenge_started_at)


if __name__ == "__main__":
    unittest.main()
