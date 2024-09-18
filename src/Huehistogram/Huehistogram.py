import math
import os

import cv2
import line_profiler
import memory_profiler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from seaborn import barplot
from matplotlib.pyplot import savefig, show
from concurrent.futures import ProcessPoolExecutor

Uint8Info = np.iinfo(np.uint8)
UINT8_MAX = Uint8Info.max
UINT8_MIN = Uint8Info.min

# フィルター設定
SATURATION_MAX = UINT8_MAX - 60  # 彩度の値の上限 (これより大きいのは数えない)
SATURATION_MIN = UINT8_MIN + 30  # 彩度の値の下限 (これより小さいのは数えない)
BRIGHTNESS_MAX = UINT8_MAX - 0  # 明るさの値の上限 (これより大きいのは数えない)
BRIGHTNESS_MIN = UINT8_MIN + 20  # 明るさの値の下限 (これより小さいのは数えない)

# ヒストグラムの設定
PLOT_SATURATION = 255  # ヒストグラムに使う色の彩度
PLOT_BRIGHTNESS = 255  # ヒストグラムに使う色の明るさ


@memory_profiler.profile
@line_profiler.profile
def generate_hue_histogram(
    image_path: str, out_path: str | None = None, out_dpi: int | None = None
) -> Axes:
    """
    画像ファイルから彩度ヒストグラムを生成して、表示・保存する

    引数・戻り値
    ----------
    :param image_path: str
        生成対象にしたい画像。OpenCVがサポートしていればなんでも使えます。
        例: "/path/to/image.jpg"
    :param out_path: str | None
        生成後に画像を出力するパスとファイル名。
        None を指定するか未指定だと、かわりに plt.show() を使って表示します。
        拡張子は matplotlib がサポートしていればなんでも使えます。確認しないので、上書き注意！
        例: "/path/to/histogram.png"
    :param out_dpi: int | None
        生成するグラフのdpi。
        None を指定すると matplotlib に任せます。
        例: 600 (600dpi: 1920 x 1440 くらい)
    :return: matplotlib.axes._axe.Axes
        描画したヒストグラムの Axes オブジェクト。凡例は削除され、タテ・ヨコ軸、タイトルは設定された状態で返ります
    """

    img = cv2.imread(image_path)
    hsvf = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL).astype(
        np.int32
    )  # この HSVの H は0~255まである

    # [H,S,V], [H, S, V] ... に変換
    hsvf = hsvf.reshape(hsvf.shape[0] * hsvf.shape[1], 3)

    # V が BrightnessMin <= V <= BrightnessMax のものだけ抜き出す
    # S が SaturationMin <= S <= SaturationMax のものだけを抜き出す
    cutter = np.where(
        (BRIGHTNESS_MIN <= hsvf[:, 2])
        & (hsvf[:, 2] <= BRIGHTNESS_MAX)
        & (SATURATION_MIN <= hsvf[:, 1])
        & (hsvf[:, 1] <= SATURATION_MAX)
    )
    hsvf = hsvf[cutter]

    img_hue_values, img_hue_counts = np.unique(hsvf[:, 0], return_counts=True)
    # Hue 飛ばすとまずいので 0 ~ 255 までの Hue の入れ口を作っておく
    hue_range = range(0, UINT8_MAX + 1)
    hue_counts = np.zeros(
        [UINT8_MAX + 1]
    )  # index を hue とした、同じ hue を持つ画素の数

    for i, v in enumerate(img_hue_values):
        hue_counts[v] = img_hue_counts[i]
    # で counts に代入

    # Hue -> BGR -> RGBA に変換して seaborn が読めるパレットを作る
    palette = []
    CV2_HUE_MAX = 180
    PLT_HUE_MAX = 255
    for Hue in hue_range:
        # OpenCV が扱える Hue に変換しつつ HSV の色を指定
        hsv_pixel = np.array(
            [Hue / PLT_HUE_MAX * CV2_HUE_MAX, PLOT_SATURATION, PLOT_BRIGHTNESS],
            dtype=np.uint8,
        )
        hsv_image = hsv_pixel.reshape(1, 1, 3)  # 1x1 の画像を作る
        # HSV -> BGR
        bgr_pixel = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)[0][0]
        # BGR -> RGBA (A = 1.0)
        palette.append(
            (
                bgr_pixel[2] / UINT8_MAX,
                bgr_pixel[1] / UINT8_MAX,
                bgr_pixel[0] / UINT8_MAX,
                1.0,
            )
        )

    # Plot する
    # ついでに凡例を追加したり x label を減らしたり
    if out_path is not None:
        plt.figure(dpi=out_dpi)  # dpi設定
    g = barplot(x=hue_range, y=hue_counts, hue=hue_range, palette=palette, lw=0.0)
    g.legend().remove()
    g.set_title(f"Hue Histogram of {image_path}")
    g.axes.set_ylabel("Pixels")
    g.axes.set_xlabel("Hue")
    g.axes.set_xticks(hue_range[:: math.floor(len(hue_range) / 10)])

    # 保存か表示する
    if out_path is not None:
        if out_dpi is not None:
            savefig(out_path)
        else:
            savefig(out_path, dpi=out_dpi)
    else:
        show()

    return g


async def generate_hue_histograms(
    input_files: list[str], output_dir: str, output_dpi: int | None = None
):
    with ProcessPoolExecutor(max_workers=8) as executor:
        for input_file in input_files:
            output_path = os.path.join(output_dir, os.path.basename(input_file))
            print(f"submit {input_file} ...")
            executor.submit(generate_hue_histogram, input_file, output_path, output_dpi)
            print(f"submit {input_file} done")
    return


# generate_hue_histogram(
#     image_path="tests/files/StickiesOnPaper0.JPG",
#     out_path="tests/files/out/StickiesOnPaper0.PNG",
#     out_dpi=300
# )
