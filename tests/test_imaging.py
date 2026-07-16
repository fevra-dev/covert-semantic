import io

from PIL import Image

from csd.imaging import render_text_to_png


def test_render_returns_valid_png():
    png = render_text_to_png("The azure report noted a bright signal.")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"          # PNG signature
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG" and img.width == 760 and img.height > 40


def test_render_grows_with_more_text():
    short = render_text_to_png("one line")
    long = render_text_to_png("\n".join(f"sentence number {i} here" for i in range(12)))
    assert Image.open(io.BytesIO(long)).height > Image.open(io.BytesIO(short)).height


def test_render_handles_empty():
    png = render_text_to_png("")
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and Image.open(io.BytesIO(png)).height > 0
