import os

import cv2
import pandas as pd
import numpy as np

image_path = (
    r"docs\images\pexels-nietjuh-1883385.jpg"
)
out_path = r"docs\images"

if not os.path.exists(out_path):
    os.makedirs(out_path)


def count_pixels(image: np.ndarray) -> pd.DataFrame:
    """
    画像内のピクセルの色をカウントし、DataFrameにまとめる

    :param image: 画像データ (numpy.ndarray)
    :return: 色ごとのカウントを含むDataFrame
    """
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


if __name__ == "__main__":
    img = cv2.imread(image_path)

    # 画像を読み込む
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # 画像を読み込み、色をカウント
    df_desc_index = count_pixels(
        cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    )

    print(df_desc_index.head(10))  # 上位10色とそのカウントを表示

    base_name = os.path.basename(image_path)
    file_name = os.path.splitext(base_name)[0]
    save_path = os.path.join(out_path, f"{file_name}_color_counts.csv")
    df_desc_index.to_csv(save_path, index=False)

    print(f"Color counts saved to {save_path}")