"""Render text as a PNG image — the modality-laundering carrier (FigStep/pxpipe class).

Used to test whether rendering a semantic-stego payload as an image defeats a text-oriented
informed monitor (the covert-semantic x modality-laundering bridge experiment). Pure/offline;
no model, no network."""
from __future__ import annotations
import io
import textwrap

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()  # portable fallback (tests / non-macOS)


def render_text_to_png(text: str, width: int = 760, font_size: int = 28,
                       line_chars: int = 46, pad: int = 20) -> bytes:
    """Render `text` as black-on-white PNG, wrapped. Returns PNG bytes.
    Legible, ordinary text-as-image — the point is that the words survive OCR while a
    text-only monitor never sees them."""
    font = _load_font(font_size)
    lines: list[str] = []
    for para in (text or " ").split("\n"):
        lines.extend(textwrap.wrap(para, line_chars) or [""])
    line_h = font_size + 10
    height = pad * 2 + line_h * max(1, len(lines))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
