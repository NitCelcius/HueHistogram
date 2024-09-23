import math
import os
import warnings
from asyncio import Future
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.pyplot import savefig, show
from seaborn import barplot

Uint8Info = np.iinfo(np.uint8)
UINT8_MAX = Uint8Info.max
UINT8_MIN = Uint8Info.min


class HueHistogramOptionFilter:
    """
    ヒストグラム生成時の画素計算に対するオプション

    引数・戻り値
    ----------
    :param saturation_upper_limit: int
        彩度の上限。これより彩度の大きい画素は数えません。0 <= x <= 255 で指定します。
        指定しない場合上限を設けず OpenCV が HSV で扱う範囲ですべて使用します。
        (196 がデフォルトでした)
    :param saturation_lower_limit: int
        彩度の下限。これより彩度の小さい画素は数えません。0 <= x <= 255 で指定します。
        指定しない場合下限を設けず OpenCV が HSV で扱う範囲ですべて使用します。
        (30 がデフォルトでした)
    :param brightness_upper_limit: int
        明度の上限。これより明度の大きい画素は数えません。0 <= x <= 255 で指定します。
        指定しない場合上限を設けず OpenCV が HSV で扱う範囲ですべて使用します。
        (255 がデフォルトでした)
    :param brightness_lower_limit: int
        明度の下限。これより明度の小さい画素は数えません。0 <= x <= 255 で指定します。
        指定しない場合下限を設けず OpenCV が HSV で扱う範囲ですべて使用します。
        (20 がデフォルトでした)
    """

    saturation_upper_limit: int = UINT8_MAX
    saturation_lower_limit: int = UINT8_MIN
    brightness_upper_limit: int = UINT8_MAX
    brightness_lower_limit: int = UINT8_MIN

    def __init__(
        self,
        saturation_upper_limit: int | None = None,
        saturation_lower_limit: int | None = None,
        brightness_upper_limit: int | None = None,
        brightness_lower_limit: int | None = None,
    ):
        if saturation_upper_limit is not None:
            self.saturation_upper_limit = saturation_upper_limit
        if saturation_lower_limit is not None:
            self.saturation_lower_limit = saturation_lower_limit
        if brightness_upper_limit is not None:
            self.brightness_upper_limit = brightness_upper_limit
        if brightness_lower_limit is not None:
            self.brightness_lower_limit = brightness_lower_limit


class HueHistogramOptions:
    """
    ヒストグラム生成時のオプション

    プロパティ
    ----------
    :param pixel_filter: HueHistogramOptionFilter | None
        ヒストグラムを描画する前、画素を数えるときのフィルタ。
        None や指定しない場合 HueHistogramOptionFilter のデフォルトを使用します。
    :param bar_saturation: int
        ヒストグラムを生成するときに使う、棒グラフの色の彩度。
    :param bar_brightness: int
        ヒストグラムを生成するときに使う、棒グラフの色の明度。
    :param output_dpi: int
        ヒストグラムを生成するときの画素密度(dpi)。
        None や指定しない場合 matplotlib の挙動に任せます。
        例: 600 (600dpi: 1920 x 1440 くらい)
    """

    pixel_filter: HueHistogramOptionFilter = HueHistogramOptionFilter()
    bar_saturation: int = 255
    bar_brightness: int = 255
    output_dpi: int | None = None

    def __init__(
        self,
        pixel_filter: HueHistogramOptionFilter | None = None,
        bar_saturation: int | None = None,
        bar_brightness: int | None = None,
        output_dpi: int | None = None,
    ):
        if pixel_filter is not None:
            self.pixel_filter = pixel_filter
        else:
            self.pixel_filter = HueHistogramOptionFilter()
        if bar_saturation is not None:
            self.bar_saturation = bar_saturation
        if bar_brightness is not None:
            self.bar_brightness = bar_brightness
        if output_dpi is not None:
            self.output_dpi = output_dpi


# @memory_profiler.profile
# @line_profiler.profile
def generate_hue_histogram(
    image_path: str,
    out_path: str | None = None,
    out_dpi: int | None = None,
    verbose: bool = False,
    allow_overwrite: bool = False,
    options: HueHistogramOptions | None = None,
) -> Axes | None:
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
    :param verbose: bool
        ログを出力するかどうか。
        True であれば進捗などいっぱいメッセージを出します。
        False であれば警告以外何も出力しません。
    :param allow_overwrite: bool
        出力先の画像が存在する場合に上書きするかどうか。
        True だと確認せず上書きします。
        False だと上書きせず警告します。
    :param options: HueHistogramOptions | None
        ヒストグラム生成時に使うオプション。
        指定しない場合はデフォルトの設定を使います。
    :return: matplotlib.axes._axe.Axes
        描画したヒストグラムの Axes オブジェクト。凡例は削除され、タテ・ヨコ軸、タイトルは設定された状態で返ります
    """

    if options is None:
        options = HueHistogramOptions()

    BRIGHTNESS_MIN = options.pixel_filter.brightness_lower_limit
    BRIGHTNESS_MAX = options.pixel_filter.brightness_upper_limit
    SATURATION_MAX = options.pixel_filter.saturation_upper_limit
    SATURATION_MIN = options.pixel_filter.saturation_lower_limit
    PLOT_SATURATION = options.bar_saturation
    PLOT_BRIGHTNESS = options.bar_brightness

    # 上書きを防止する
    if os.path.exists(out_path) and not allow_overwrite:
        warnings.warn(
            f"The file {out_path} already exists, skipping! Specify allow_overwrite=True to allow overwrite."
        )
        return None

    if verbose:
        print(f"Processing {image_path}...")
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
            if os.path.exists(out_path) and allow_overwrite is False:
                warnings.warn(
                    f"The file {out_path} already exists so unable to save the histogram! Specify allow_overwrite=True to allow overwrite."
                )
                return
            savefig(out_path)
        else:
            if os.path.exists(out_path) and allow_overwrite is False:
                warnings.warn(
                    f"The file {out_path} already exists so unable to save the histogram! Specify allow_overwrite=True to allow overwrite."
                )
                return
            savefig(out_path, dpi=out_dpi)
    else:
        show()

    if verbose:
        print(f"Histogram saved to {out_path}")

    return g


async def generate_hue_histograms(
    input_files: list[str],
    output_paths: list[str],
    **kwargs,
    # out_dpi: int | None = None,
    # verbose: bool | None = None,
    # allow_overwrite: bool | None = None,
) -> bool:
    failed_files: list[str] = list()
    path_and_future_mapping: dict[str, Future] = dict()
    if len(input_files) != len(output_paths):
        raise ValueError("input_files and output_paths must have the same length")

    with ProcessPoolExecutor(max_workers=8) as executor:
        for input_file, output_path in zip(input_files, output_paths):
            future = executor.submit(
                generate_hue_histogram, input_file, output_path, **kwargs
            )
            path_and_future_mapping[output_path] = future
    for output_path in path_and_future_mapping:
        if path_and_future_mapping[output_path].result() is None:
            failed_files.append(input_file)
    if kwargs["verbose"] and failed_files:
        print("Failed to generate the histograms for following files: ")
        print(failed_files)
    return not failed_files  # 空っぽだと True になる


# generate_hue_histogram(
#     image_path="tests/files/StickiesOnPaper0.JPG",
#     out_path="tests/files/out/StickiesOnPaper0.PNG",
#     out_dpi=300
# )
