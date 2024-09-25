from src.huehistogram_cli import input_and_output_extract_args


def test_ioargs_with_ext():
    input_paths, output_paths = input_and_output_extract_args(
        input_arg=["resources/hello.jpg"], output_arg="out/", output_extension="png"
    )
    assert output_paths == ["out\\hello.png"]

    input_paths, output_paths = input_and_output_extract_args(
        input_arg=["resources/hello.jpg", "hello.png", "hello.pdf.jpg"],
        output_arg="resources/",
        output_extension="png",
    )
    assert output_paths == [
        "resources/hello.png",
        "resources/hello.png",
        "resources/hello.pdf.png",
    ]

    input_paths, output_paths = input_and_output_extract_args(
        input_arg=["resources/hello.jpg", "hello.png", "hello.pdf.jpg"],
        output_arg="resources/",
        output_extension="_hist.png",
    )
    assert output_paths == [
        "resources/hello_hist.png",
        "resources/hello_hist.png",
        "resources/hello.pdf_hist.png",
    ]