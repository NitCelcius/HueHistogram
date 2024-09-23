# Run me from console!
import asyncio
import os
import sys
import argparse
from typing import Tuple

from Huehistogram import Huehistogram

SIG_INVALID_ARGS = 1


def input_and_output_extract_args(
    input_arg: str | list[str], output_arg: str
) -> Tuple[list[str], list[str]]:
    if isinstance(input_arg, str):
        # input_arg: list[str] を保証
        return input_and_output_extract_args([input_arg], output_arg)

    if os.path.isdir(output_arg):  # 存在するディレクトリならここに入れれば OK
        output_paths = [
            os.path.join(output_arg, os.path.basename(input_file))
            for input_file in input_arg
        ]
        return input_arg, output_paths

    # output_arg は存在するディレクトリではないので、ファイルか存在しないディレクトリ
    # つまりここに書き込むことはできない
    output_file_name = os.path.basename(output_arg)
    if output_file_name == "":  # output_arg がファイルじゃないなら
        if (
            len(input_arg) >= 2
        ):  # のわりに入力ファイルが2つ以上あるとどの名前を使えばいいか不明
            # これは input_arg が str の場合も含むけど list で包括的に処理したかった
            print(
                "If you are trying to input two or more files, "
                "use --output-dir option or specify a path to "
                "existing directory on --output option."
            )
            exit(SIG_INVALID_ARGS)
        # 入力ファイルが1つなので、出力ファイルも同じ名前を使わせる
        output_file_name = os.path.basename(input_arg[0])

    # output 先のディレクトリに output_file_name なるファイルを作れということと解釈
    output_paths = [os.path.join(os.path.dirname(output_arg), output_file_name)]
    return input_arg, output_paths


if __name__ == "__main__":
    # TODO: これ外に出す
    def show_help():
        print("Usage:")
        print("  To show color histogram:")
        print("    python huehistogram_cli.py [path/to/image.jpg]")
        print("  To save color histogram:")
        print(
            "    python huehistogram_cli.py -i [path/to/image.jpg] [image_2.jpg] -o [path/to/histograms/]"
        )
        return

    if len(sys.argv) <= 1:
        show_help()
        exit(SIG_INVALID_ARGS)

    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*")
    arg_input_group = parser.add_mutually_exclusive_group()
    arg_input_group.add_argument(
        "-i", "--input", help="path(s) to input image files", nargs="+"
    )
    arg_input_group.add_argument(
        "-dir",
        "--directory-input",
        help="the path input image files are located",
        nargs="+",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path(s) to the directory to output histogram(s). Specify a path to image file to input just one image, otherwise specify a path to existing directory.",
    )
    parser.add_argument(
        "-v", "--verbose", help="display verbose messages", action="store_true"
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        help="Do overwrite files if a file exists on destination",
        action="store_true",
    )
    parser.add_argument("-d", "--dpi", help="the dpi to output histograms")
    args = parser.parse_args()

    input_files = []
    is_verbose = args.verbose
    input_arg = args.input
    output_arg = args.output

    if args.directory_input:
        # directory_input, input は mutually exclusive なので input_arg = None であることが保証される
        input_path = args.directory_input[0]
        input_arg = [  # 子要素ぜんぶ input に指定したことにする
            os.path.join(input_path, file_name) for file_name in os.listdir(input_path)
        ]

    if input_arg is not None and output_arg is not None and args.args:
        print(
            "Extra arguments are supplied! "
            "If you intend to use --input and --output options, "
            "please remove extra arguments other than -i and -o options."
        )
        print("Use:")
        print(
            "  python huehistogram_cli.py -i [path/to/image.jpg] [image_2.jpg] -o [path/to/histograms/]"
        )
        exit(SIG_INVALID_ARGS)

    if input_arg is None and output_arg is None:
        # Without explicit input or output files
        input_arg = sys.argv[
            1:-1
        ]  # 実行するプログラム名と最後を除いて input だと思い込む
        output_arg = sys.argv[-1]  # 最後は output だと思い込む

    if output_arg is None:
        print("Please specify a path to output histograms using --output option!")
        exit(SIG_INVALID_ARGS)

    input_files, output_paths = input_and_output_extract_args(input_arg, output_arg)

    if not input_files or not output_paths:
        print("Please specify both path(s) to input and output files!")
        exit(SIG_INVALID_ARGS)

    if any(
        (
            not os.path.isdir(nonexistent_output_dir := os.path.dirname(output_file))
            for output_file in output_paths
        )
    ):
        print(f"Output directory {nonexistent_output_dir} does not exist!")
        exit(SIG_INVALID_ARGS)

    if any(
        (
            not os.path.isdir(os.path.dirname(nonexistent_input_file := input_file))
            for input_file in input_files
        )
    ):
        print(f"Input file {nonexistent_input_file} does not exist!")
        exit(SIG_INVALID_ARGS)

    options = Huehistogram.HueHistogramOptions()
    if args.dpi:
        options.output_dpi = args.dpi

    res = asyncio.run(
        Huehistogram.generate_hue_histograms(
            input_files=input_files,
            output_paths=output_paths,
            verbose=is_verbose,
            allow_overwrite=args.force_overwrite,
            options=options,
        )
    )
    if res:
        exit(0)
    else:
        print(
            "Tip: If you want to overwrite files on destination, use -f (--force-overwrite) option."
        )
        exit(1)
