# Run me from console!
import os
import argparse
from typing import Tuple
from Huehistogram.CountColors import count_colors_from_file
import asyncio
from concurrent.futures import ProcessPoolExecutor

SIG_INVALID_ARGS = 1


def input_and_output_extract_args(
    input_arg: str | list[str], output_arg: str
) -> Tuple[list[str], list[str]]:
    if isinstance(input_arg, str):
        return input_and_output_extract_args([input_arg], output_arg)
    if os.path.isdir(output_arg):
        output_paths = [
            os.path.join(
                output_arg,
                os.path.splitext(os.path.basename(input_file))[0] + "_color_counts.csv",
            )
            for input_file in input_arg
        ]
        return input_arg, output_paths
    output_file_name = os.path.basename(output_arg)
    if output_file_name == "":
        if len(input_arg) >= 2:
            print(
                "If you are trying to input two or more files, "
                "use --output-dir option or specify a path to "
                "existing directory on --output option."
            )
            exit(SIG_INVALID_ARGS)
        base_name = os.path.splitext(os.path.basename(input_arg[0]))[0]
        output_file_name = f"{base_name}_color_counts.csv"
    output_paths = [os.path.join(os.path.dirname(output_arg), output_file_name)]
    return input_arg, output_paths


def run_color_counting(
    input_file: str, output_path: str, ignore_transparent: bool, verbose: bool = False
) -> bool:
    try:
        if verbose:
            print(f"Counting colors in: {input_file}")
        df = count_colors_from_file(
            input_file, ignore_transparent_pixels=ignore_transparent
        )
        if verbose:
            print(f"Saving results to: {output_path}")
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False


async def process_image_async(
    input_file: str,
    output_path: str,
    allow_overwrite: bool,
    ignore_transparent: bool,
    verbose: bool = False,
) -> bool:
    if os.path.exists(output_path) and not allow_overwrite:
        print(f"Output file already exists: {output_path}")
        print("Use --force-overwrite (-f) option to overwrite existing files.")
        return False
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, run_color_counting, input_file, output_path, ignore_transparent, verbose
    )


async def generate_color_counts(
    input_files: list[str],
    output_paths: list[str],
    allow_overwrite: bool,
    ignore_transparent: bool,
    verbose: bool = False,
    max_workers: int = 8,
) -> bool:
    if len(input_files) != len(output_paths):
        raise ValueError("input_files and output_paths must have the same length")

    success = True
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for input_file, output_path in zip(input_files, output_paths):
            futures.append(
                executor.submit(
                    run_color_counting,
                    input_file,
                    output_path,
                    ignore_transparent,
                    verbose,
                )
            )
        for future, input_file, output_path in zip(futures, input_files, output_paths):
            if os.path.exists(output_path) and not allow_overwrite:
                print(f"Output file already exists: {output_path}")
                print("Use --force-overwrite (-f) option to overwrite existing files.")
                success = False
                continue
            result = future.result()
            success = success and result

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count colors in images")
    parser.add_argument(
        "args",
        nargs="*",
        help="path(s) to some input files and output directory/file. The last argument will be used as output path if you do not specify -i and -o explicitly.",
    )
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
        help="path(s) to the directory to output color counts. Specify a path to CSV file to input just one image, otherwise specify a path to existing directory.",
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
    parser.add_argument(
        "--ignore-transparent",
        help="Ignore fully transparent pixels (alpha=0)",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--include-transparent",
        help="Include fully transparent pixels in the count",
        action="store_false",
        dest="ignore_transparent",
    )
    args = parser.parse_args()

    input_files = []
    is_verbose = args.verbose
    input_arg = args.input
    output_arg = args.output

    if args.directory_input:
        input_path = args.directory_input[0]
        input_arg = [
            os.path.join(input_path, file_name)
            for file_name in os.listdir(input_path)
            if file_name.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
            )
        ]
        if not input_arg:
            print(f"No image files found in directory: {input_path}")
            exit(SIG_INVALID_ARGS)

    if input_arg is not None and output_arg is not None and args.args:
        print(
            "Extra arguments are supplied! "
            "If you intend to use --input and --output options, "
            "please remove extra arguments other than -i and -o options."
        )
        print("Use:")
        print(
            "  python countcolors_cli.py -i [path/to/image.jpg] [image_2.jpg] -o [path/to/output/]"
        )
        exit(SIG_INVALID_ARGS)

    if input_arg is None and output_arg is None:
        if len(args.args) < 2:
            print("Please specify both input and output paths!")
            exit(SIG_INVALID_ARGS)
        input_arg = args.args[:-1]
        output_arg = args.args[-1]

    if output_arg is None:
        print("Please specify a path to output color counts using --output option!")
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

    res = asyncio.run(
        generate_color_counts(
            input_files,
            output_paths,
            args.force_overwrite,
            args.ignore_transparent,
            is_verbose,
        )
    )

    if not res:
        exit(SIG_INVALID_ARGS)
