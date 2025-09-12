import os
import cv2
from src.Huehistogram.CountColors import count_pixels
import unittest


class TestCountColors(unittest.TestCase):
    def _test_single_color_image(self, image_path, expected_color, expected_count):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        df = count_pixels(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        row = df[
            (df["Color_R"] == expected_color[0])
            & (df["Color_G"] == expected_color[1])
            & (df["Color_B"] == expected_color[2])
        ]
        self.assertTrue(not row.empty, f"色 {expected_color} の行がありません")
        self.assertEqual(
            row["Count"].values[0],
            expected_count,
            f"色 {expected_color} のカウントが {expected_count} ではありません",
        )
        self.assertTrue(len(row) == 1, f"期待した色以外が含まれています: {row}")

    def test_red_only_image(self):
        self._test_single_color_image(
            "tests/files/r255_g0_b0_h36_w36.png", (255, 0, 0), 36 * 36
        )

    def test_green_only_image(self):
        self._test_single_color_image(
            "tests/files/r0_g255_b0_h36_w36.png", (0, 255, 0), 36 * 36
        )

    def test_blue_only_image(self):
        self._test_single_color_image(
            "tests/files/r0_g0_b255_h36_w36.png", (0, 0, 255), 36 * 36
        )

    def test_black_only_image(self):
        self._test_single_color_image(
            "tests/files/r0_g0_b0_h36_w36.png", (0, 0, 0), 36 * 36
        )

    def test_white_only_image(self):
        self._test_single_color_image(
            "tests/files/r255_g255_b255_h36_w36.png", (255, 255, 255), 36 * 36
        )
