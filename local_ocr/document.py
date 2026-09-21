import os
import io
from typing import List, Tuple, Generator, Optional, Union, Any

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    from PIL import Image
except ImportError:
    Image = None

from .models import BoundingBox

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp', '.gif'}
PDF_EXTENSIONS = {'.pdf'}

class PageFrame(tuple):
    """
    Subclass of 4-tuple (page_num, img_source, width, height) supporting
    tuple unpacking while providing offset_x and offset_y attributes.
    """
    offset_x: float
    offset_y: float

    def __new__(
        cls,
        page_num: int,
        img_source: Union[str, bytes],
        width: Optional[float],
        height: Optional[float],
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ):
        obj = super().__new__(cls, (page_num, img_source, width, height))
        obj.offset_x = float(offset_x)
        obj.offset_y = float(offset_y)
        return obj


class DocumentProcessor:
    """
    Handles loading, rendering, and cropping documents (images and multi-page PDFs)
    directly in-memory without creating temporary files on disk.
    """

    @staticmethod
    def is_pdf(source: Union[str, bytes]) -> bool:
        if isinstance(source, str):
            ext = os.path.splitext(source)[1].lower()
            return ext in PDF_EXTENSIONS
        elif isinstance(source, (bytes, bytearray)):
            return source.startswith(b"%PDF-")
        return False

    @staticmethod
    def is_image_file(file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in IMAGE_EXTENSIONS

    @classmethod
    def get_pages(
        cls,
        source: Union[str, bytes],
        dpi: int = 150,
        page_indices: Optional[List[int]] = None,
        rect: Optional[Any] = None
    ) -> Generator[PageFrame, None, None]:
        """
        Yields PageFrame(page_num, img_source, width, height, offset_x, offset_y) for each page.
        Supports 4-tuple unpacking (page_num, img_source, w, h) for backward compatibility.
        If rect is specified (x1, y1, x2, y2), crops the page directly in-memory before yielding.
        """
        target_rect = BoundingBox.parse(rect) if rect is not None else None

        if cls.is_pdf(source):
            if pymupdf is None:
                raise RuntimeError("PyMuPDF is required for PDF processing. Run: pip install pymupdf")

            if isinstance(source, str):
                if not os.path.exists(source):
                    raise FileNotFoundError(f"PDF file not found: {source}")
                doc = pymupdf.open(source)
            else:
                doc = pymupdf.open(stream=source, filetype="pdf")

            try:
                total_pages = len(doc)
                indices = page_indices if page_indices is not None else range(total_pages)

                for idx in indices:
                    if 0 <= idx < total_pages:
                        page = doc[idx]
                        page_rect = page.rect

                        if target_rect is not None:
                            # Convert pixel coordinates to PDF point coordinates at specified DPI
                            scale = 72.0 / dpi
                            pdf_w = float(page_rect.width)
                            pdf_h = float(page_rect.height)
                            pix_w = pdf_w * dpi / 72.0
                            pix_h = pdf_h * dpi / 72.0

                            # Clamp rectangle to rendered pixel bounds
                            rx1 = max(0.0, min(pix_w, target_rect.x1))
                            ry1 = max(0.0, min(pix_h, target_rect.y1))
                            rx2 = max(0.0, min(pix_w, target_rect.x2))
                            ry2 = max(0.0, min(pix_h, target_rect.y2))

                            if rx2 <= rx1 or ry2 <= ry1:
                                continue

                            clip_box = pymupdf.Rect(rx1 * scale, ry1 * scale, rx2 * scale, ry2 * scale)
                            pix = page.get_pixmap(dpi=dpi, clip=clip_box)
                            img_bytes = pix.tobytes("png")
                            yield PageFrame(idx + 1, img_bytes, float(pix.width), float(pix.height), offset_x=rx1, offset_y=ry1)
                        else:
                            pix = page.get_pixmap(dpi=dpi)
                            img_bytes = pix.tobytes("png")
                            yield PageFrame(idx + 1, img_bytes, float(pix.width), float(pix.height), offset_x=0.0, offset_y=0.0)
            finally:
                doc.close()
        else:
            # Single image source
            if target_rect is not None:
                if Image is None:
                    raise RuntimeError("Pillow (PIL) is required for image cropping. Run: pip install Pillow")

                if isinstance(source, str):
                    if not os.path.exists(source):
                        raise FileNotFoundError(f"Image file not found: {source}")
                    img = Image.open(source)
                else:
                    img = Image.open(io.BytesIO(source))

                with img:
                    w, h = img.width, img.height
                    rx1 = max(0.0, min(float(w), target_rect.x1))
                    ry1 = max(0.0, min(float(h), target_rect.y1))
                    rx2 = max(0.0, min(float(w), target_rect.x2))
                    ry2 = max(0.0, min(float(h), target_rect.y2))

                    if rx2 <= rx1 or ry2 <= ry1:
                        return

                    cropped = img.crop((int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2))))
                    if cropped.mode not in ("RGB", "L"):
                        cropped = cropped.convert("RGB")

                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                    yield PageFrame(1, img_bytes, float(cropped.width), float(cropped.height), offset_x=rx1, offset_y=ry1)
            else:
                if isinstance(source, str):
                    if not os.path.exists(source):
                        raise FileNotFoundError(f"Image file not found: {source}")
                    yield PageFrame(1, source, None, None, offset_x=0.0, offset_y=0.0)
                else:
                    yield PageFrame(1, source, None, None, offset_x=0.0, offset_y=0.0)
