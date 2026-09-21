"""
LocalOCR - High performance, local and private OCR toolkit for Python and CLI.
"""

from .engine import LocalOCR, OCREngine
from .models import OCRResult, OCRPage, OCRLine, BoundingBox
from .language import detect_language, detect_language_detailed
from .formatters import to_text, to_markdown, to_sidecar

__version__ = "1.0.0"
__all__ = [
    "LocalOCR",
    "OCREngine",
    "OCRResult",
    "OCRPage",
    "OCRLine",
    "BoundingBox",
    "detect_language",
    "detect_language_detailed",
    "to_text",
    "to_markdown",
    "to_sidecar",
]
