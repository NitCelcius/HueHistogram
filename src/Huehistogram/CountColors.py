import os

import cv2
import pandas as pd
import numpy as np

image_path = r"docs\images\pexels-nietjuh-1883385.jpg"
out_path = r"docs\images"

if not os.path.exists(out_path):
    os.makedirs(out_path)


def count_pixels(
    image: np.ndarray, ignore_transparent_pixels: bool = True
) -> pd.DataFrame:
    """
    画像内のピクセルの色をカウントし、DataFrameにまとめる
    ignore_transparent_pixels を True にすると完全に透過しているピクセル（alpha=0）を無視する

    :param image: 画像データ (numpy.ndarray)
    :param ignore_transparent_pixels: 完全に透過しているピクセルを無視するかどうか (True で無視)
    :return: 色ごとのカウントを含むDataFrame
    """
    # アルファチャンネルがある場合 A=0 のやつは除外
    if image.shape[-1] == 4:
        # RGBA
        if ignore_transparent_pixels:
            mask = image[..., 3] != 0
        else:
            mask = np.ones(image.shape[:2], dtype=bool)  # ぜんぶ対象
        pixels = image[mask]
        pixels = pixels[:, :3]  # RGBのみ
    else:
        pixels = image.reshape(-1, 3)
    pixels = pixels.astype(int)
    color_int = pixels[:, 0] + pixels[:, 1] * 256 + pixels[:, 2] * 256 * 256
    unique_colors, counts = np.unique(color_int, return_counts=True)

    # 必要なら元のRGBに戻す
    R = unique_colors % 256
    G = (unique_colors // 256) % 256
    B = (unique_colors // (256 * 256)) % 256

    df = pd.DataFrame({"Color_R": R, "Color_G": G, "Color_B": B, "Count": counts})
    df = df.sort_values(by="Count", ascending=False)
    return df.reset_index(drop=True)


def count_colors_from_file(
    image_path: str, ignore_transparent_pixels: bool = True
) -> pd.DataFrame:
    """
    画像ファイルパスから画像を読み込み、色カウントDataFrameを返す
    :param image_path: 画像ファイルパス
    :param ignore_transparent_pixels: 完全に透過しているピクセルを無視するかどうか
    :return: 色ごとのカウントを含むDataFrame
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image file: {image_path}")
    # BGR→RGB変換（アルファチャネルがある場合はBGRA→RGBA）
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return count_pixels(img, ignore_transparent_pixels=ignore_transparent_pixels)


if __name__ == "__main__":
    # 画像を読み込み、色をカウント
    df_desc_index = count_colors_from_file(image_path)

    print(df_desc_index.head(10))  # 上位10色とそのカウントを表示

    base_name = os.path.basename(image_path)
    file_name = os.path.splitext(base_name)[0]
    save_path = os.path.join(out_path, f"{file_name}_color_counts.csv")
    df_desc_index.to_csv(save_path, index=False)

    print(f"Color counts saved to {save_path}")
