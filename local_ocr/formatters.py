import os
from typing import Optional
from .models import OCRResult

def to_text(result: OCRResult) -> str:
    """Format OCR result as plain text."""
    return result.full_text

def to_markdown(result: OCRResult) -> str:
    """Format OCR result as Markdown with page sections."""
    md_lines = []
    if result.file_path:
        md_lines.append(f"# OCR Result: {os.path.basename(result.file_path)}\n")
    md_lines.append(f"**Language**: `{result.language}` | **Pages**: `{result.page_count}`\n")
    md_lines.append("---\n")

    if result.pages:
        for page in result.pages:
            md_lines.append(f"## Page {page.page_num}\n")
            md_lines.append(page.text)
            md_lines.append("\n")
    else:
        md_lines.append(result.full_text)

    return "\n".join(md_lines)

def to_sidecar(result: OCRResult, destination_path: Optional[str] = None) -> str:
    """
    Write full OCR text to a .ocr.txt sidecar file.
    If destination_path is not specified, uses <source_file>.ocr.txt.
    Returns the path to the written sidecar file.
    """
    if destination_path is None:
        if not result.file_path:
            raise ValueError("Cannot deduce sidecar destination without a source file_path.")
        destination_path = f"{result.file_path}.ocr.txt"

    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
    with open(destination_path, "w", encoding="utf-8") as f:
        f.write(result.full_text)

    return destination_path
