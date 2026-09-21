---
name: local-ocr
description: Fast, private, and 100% local OCR engine for extracting text, bounding boxes, and languages from images and multi-page PDFs. Use this skill whenever a task requires OCR (receipts, invoices, scanned documents), extracting text from a specific rectangle/ROI (x1 y1 x2 y2), or installing and calling the LocalOCR library and CLI in any project.
---

# LocalOCR Skill

LocalOCR is a completely offline, high-performance Python library and CLI tool powered by RapidOCR (ONNX Runtime) and PyMuPDF. It operates with **zero cloud dependencies, zero network requests, and zero intermediate disk files**.

---

## 🔒 Security & Privacy Guarantee

- **100% Offline**: Never sends any document, image, coordinates, or text to any external or cloud service.
- **Air-Gapped & Safe**: Safe for sensitive financial, personal, medical, or legal documents.
- **In-Memory Buffers**: PyMuPDF renders pages and clips directly in memory without writing temporary unencrypted images to disk.

---

## Installation & Setup Across Projects

To use LocalOCR across projects, install it in editable mode:

```bash
# From within the LocalOCR repository directory:
pip install -e .

# Or from an external project environment:
pip install -e /path/to/LocalOCR   # e.g., pip install -e ../LocalOCR
```

Or if importing directly without `pip install`:
```python
import sys, os
# Direct import from repository root or sibling directory:
local_ocr_path = os.environ.get("LOCAL_OCR_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LocalOCR")))
if os.path.exists(local_ocr_path) and local_ocr_path not in sys.path:
    sys.path.insert(0, local_ocr_path)

from local_ocr import LocalOCR, OCREngine, BoundingBox, detect_language
```

Dependencies:
- `rapidocr-onnxruntime>=1.2.0`
- `pymupdf>=1.20.0`
- `Pillow`

---

## Python API Usage

### 1. Basic Document OCR

```python
from local_ocr import LocalOCR

# Default model: Chinese/English PP-OCRv3
ocr = LocalOCR()
result = ocr.process("path/to/document.pdf")  # or .png, .jpg, .tiff, etc.

print("Language:", result.language)
print("Pages:", result.page_count)
print("Execution time:", result.duration_seconds, "s")
print("Full Text:\n", result.full_text)
```

### 2. Japanese OCR (`lang="ja"`)

RapidOCR's default dictionary contains 0 Hiragana characters. LocalOCR bundles a dedicated, high-accuracy Japanese PP-OCRv4 model (`japan_PP-OCRv4_rec_infer.onnx` with full 4,399 character dictionary covering Hiragana, Katakana, and Kanji):

```python
ocr_ja = LocalOCR(lang="ja")
result = ocr_ja.process("japanese_card.png")
print(result.full_text)  # Exact Hiragana/Kanji recognition: e.g. "穴がある", "生きている", "木を含む"
```

### 3. Multi-Angle Orientation Verification & Fast-Path

Documents and game cards may be scanned upside down (180°) or sideways (90° / 270°). LocalOCR includes automatic multi-angle orientation verification with smart candidate deduplication:

- **`compare_orthogonal_base: bool = True`** (evaluates 0°, 90°, 180°, 270°)
- **`use_angle_cls: bool = True`** (runs line classifier)
- **`compare_orthogonal_angle_cls: bool = True`** (evaluates candidate angles relative to classifier angle)

Candidates are deduplicated (evaluating each unique angle at most once) and scored by character script quality.

> [!TIP]
> **Fast-Path Recommendation for Confident Scans**:
> If you are confident that the source document is already clean and correctly oriented (upright), pass `compare_orthogonal_base=False, compare_orthogonal_angle_cls=False` (or `--no-orthogonal` in the CLI) to skip multi-angle checks and run at maximum speed (~10 ms).
>
> ```python
> fast_ocr = LocalOCR(lang="ja", compare_orthogonal_base=False, compare_orthogonal_angle_cls=False)
> ```

> [!WARNING]
> **Angle Classifier Advisory for Japanese**:
> The standard line orientation classifier (`ch_ppocr_mobile_v2.0_cls_infer.onnx`) is trained on Chinese text and can misclassify curved Hiragana characters (such as `生きている`) as 180° upside-down.
> With default multi-angle verification enabled, LocalOCR automatically scores and rejects the false 180° flip.
> When using the fast path (`--no-orthogonal`) on upright Japanese documents, pass `use_angle_cls=False` (`--no-angle-cls`) to prevent false inversion.

### 4. OCR a Specific Rectangle (ROI / Bounding Box)

Extract text exclusively from a sub-region `(x1, y1, x2, y2)` where coordinates represent `(left, top, right, bottom)` in pixels:

```python
# Method A: Dedicated process_rect helper
result = ocr.process_rect("invoice.png", rect=(100, 200, 500, 400))

# Method B: Using process with rect argument
result = ocr.process("invoice.pdf", rect="100 200 500 400", dpi=150)

# Line coordinates are preserved in the original document space:
for line in result.lines:
    print(f"Text: {line.text} (conf: {line.confidence:.2f})")
    print(f"  Document Box: ({line.box.x1}, {line.box.y1}) -> ({line.box.x2}, {line.box.y2})")
    if line.relative_box:
        print(f"  Crop-Relative: ({line.relative_box.x1}, {line.relative_box.y1})")
```

> [!NOTE]
> **ROI / Cropping Best Practice**:
> Always keep crop rectangles focused on textual areas with a 10–15px padding margin. Avoid including high-contrast jagged card borders, starburst frames, or geometric divider artwork, as text detectors may interpret repetitive border spikes as character glyphs.

### 5. Spatial Query on Existing Results

If you have already processed an entire document and want to find text inside a specific region:

```python
header_lines = result.get_lines_in_rect((0, 0, 800, 150))
header_text = "\n".join(line.text for line in header_lines)
```

### 6. Multi-Page PDFs & DPI Setting

```python
# High-DPI scanning for fine receipt text on specific pages:
result = ocr.process("scans.pdf", dpi=200, page_indices=[0, 1])

for page in result.pages:
    print(f"--- Page {page.page_num} ---")
    print(page.text)
```

### 7. Language Detection

```python
from local_ocr import detect_language, detect_language_detailed

lang = detect_language("2020年11月20日 領収書")  # returns 'ja', 'en', 'fr', etc.
code, stats = detect_language_detailed("Facture d'électricité")
```

### 8. Backward Compatibility with UltimateSorter

LocalOCR provides `OCREngine` as an alias and dictionary-style result access:

```python
from local_ocr import OCREngine

engine = OCREngine()
res = engine.process_file("scan.png")
text = res["full_text"]
lang = res["language"]
lines = res["lines"]
```

---

## CLI Usage (`local-ocr`)

The CLI is available directly as `local-ocr` (or via `python -m local_ocr.cli`):

```bash
# 1. Process document and print text
local-ocr process document.pdf

# 2. Process Japanese document / card
local-ocr process japanese_card.png --lang ja

# 3. Fast single-pass OCR on clean upright scans (skips multi-angle checks)
local-ocr process document.png --no-orthogonal

# 4. OCR a specific bounding box (x1 y1 x2 y2)
local-ocr process receipt.png --rect 50 100 400 300

# 5. Save structured JSON output
local-ocr process document.pdf -f json -o result.json

# 6. Save sidecar .ocr.txt next to file
local-ocr process receipt.png -f sidecar

# 7. Batch process a directory
local-ocr batch ./Scans --lang ja --sidecars

# 8. Batch process a specific region across all documents
local-ocr batch ./Invoices --rect 100 50 500 150 --output-json headers.json

# 9. Language detection
local-ocr detect-lang "Sample text to identify"
```
