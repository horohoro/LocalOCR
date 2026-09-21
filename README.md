# LocalOCR 🔍

A fast, lightweight, and completely private local OCR library and CLI tool for Python. Powered by [RapidOCR](https://github.com/RapidAI/RapidOCR) (ONNX runtime) and [PyMuPDF](https://pymupdf.readthedocs.io/).

LocalOCR extracts text, line bounding boxes, and detects languages across single images and multi-page PDFs with **zero cloud dependencies** and **zero temporary disk files**.

---

## 🔒 100% Fully Local & Private Guarantee (Zero Remote Transmission)

> [!IMPORTANT]
> **LocalOCR is strictly 100% local and offline.**
> - **Zero Remote Calls**: **NOTHING is ever sent to any other service or remote server for OCR.** LocalOCR makes zero network requests and does not communicate with external APIs, cloud OCR services, or telemetry endpoints.
> - **Air-Gapped & Confidential**: All inference runs locally on your machine via ONNX Runtime and PyMuPDF. It is entirely safe for sensitive, confidential, financial, legal, and medical documents.
> - **Zero Temporary Disk Files**: PDF rendering and image crops take place directly in-memory buffers without writing unencrypted intermediate images to disk.

---

## Features

- ⚡ **Blazing Fast Local Execution**: Runs locally on CPU via optimized ONNX models without needing external Tesseract binaries or GPU setup.
- 🇯🇵 **Dedicated Japanese Model Support**: Bundles `japan_PP-OCRv4_rec_infer.onnx` with full 4,399 character dictionary including all Hiragana, Katakana, and Kanji.
- 📐 **Multi-Angle Orientation Verification**: Automatic candidate orientation verification across 0°, 90°, 180°, and 270° with smart deduplication and composite quality scoring.
- 🚀 **Fast-Path Configurable**: Skip multi-angle checks (`--no-orthogonal` / `compare_orthogonal_base=False`) on clean upright documents for ~10 ms single-pass OCR.
- 🎯 **Targeted Rectangle (ROI) OCR**: Extract text exclusively from a specific bounding box `(x1, y1, x2, y2)` with automatic coordinate preservation (`line.box` in document coordinates, `line.relative_box` in crop coordinates).
- 📄 **Multi-Page PDF & Image Support**: In-memory rendering of multi-page PDFs (PNG, JPG, TIFF, WebP, BMP, etc.) with custom DPI control.
- 🌐 **Multilingual Heuristics**: Automatic language detection with character statistics (Japanese, French, English, Chinese, German, Spanish).
- 📦 **Structured Results**: Access plain text, per-line bounding boxes (`[x1, y1, x2, y2]` and 4-point polygons), confidence scores, and per-page breakdowns.
- 💻 **Dual Interface**:
  - **Python Library**: Clean, typed, object-oriented API with dictionary backward-compatibility.
  - **CLI Command**: Standalone `local-ocr` terminal command for scripting, pipeline integration, and batch directory processing.

---

## Installation

Install in editable mode for local development or direct use across your projects:

```bash
cd local-ocr
pip install -e .
```

Dependencies:
- `rapidocr-onnxruntime>=1.2.0`
- `pymupdf>=1.20.0`
- `Pillow`

---

## Python Library Usage

### 1. Basic Text Extraction

```python
from local_ocr import LocalOCR

# Default model: Chinese/English PP-OCRv3
ocr = LocalOCR()

# Process an image or multi-page PDF
result = ocr.process("path/to/invoice.pdf")

print("Language:", result.language)
print("Total Pages:", result.page_count)
print("Duration:", result.duration_seconds, "s")
print("\nExtracted Text:\n", result.full_text)
```

### 2. Japanese OCR (`lang="ja"`)

```python
# Initialize with bundled Japanese PP-OCRv4 model:
ocr_ja = LocalOCR(lang="ja")

result = ocr_ja.process("japanese_card.png")
print(result.full_text)  # Exact Hiragana/Katakana/Kanji: e.g. "穴がある", "生きている", "木を含む"
```

### 3. Orientation Verification & Fast-Path

By default, LocalOCR automatically checks candidate orthogonal orientations (0°, 90°, 180°, 270°) to guarantee that inverted cards and sideways documents are correctly recognized:

```python
# Default: full multi-angle orientation check with smart deduplication
ocr = LocalOCR(
    lang="ja",
    compare_orthogonal_base=True,
    use_angle_cls=True,
    compare_orthogonal_angle_cls=True
)

# Fast path for clean, upright scans (~10 ms):
fast_ocr = LocalOCR(
    lang="ja",
    compare_orthogonal_base=False,
    compare_orthogonal_angle_cls=False
)
```

> [!WARNING]
> **Japanese Angle Classifier Note**:
> The standard RapidOCR angle classifier (`ch_ppocr_mobile_v2.0_cls_infer.onnx`) is trained on Chinese text and can misclassify curved Hiragana characters (e.g. `生きている`) as 180° upside-down.
> With default multi-angle verification, LocalOCR automatically scores candidates and rejects the false inversion.
> If using the fast path (`compare_orthogonal_base=False`) on upright Japanese documents, pass `use_angle_cls=False` to prevent misclassification.

### 4. Structured Line Details & Bounding Boxes

Each line contains text, recognition confidence, page index, and bounding box coordinates:

```python
for line in result.lines:
    print(f"Page {line.page} | [{line.confidence:.2f}] {line.text}")
    if line.box:
        print(f"  Coordinates: ({line.box.x1}, {line.box.y1}) -> ({line.box.x2}, {line.box.y2})")
        print(f"  Width: {line.box.width}, Height: {line.box.height}")
```

### 5. OCR a Specific Rectangle (ROI / Bounding Box)

Extract text exclusively from a sub-region `(x1, y1, x2, y2)` of an image or PDF page without running OCR on the rest of the document:

```python
# Pass rect as a tuple, list, BoundingBox, or string:
result = ocr.process_rect("invoice.png", rect=(100, 200, 500, 400))
# Or pass rect directly to process():
# result = ocr.process("invoice.png", rect="100 200 500 400")

print(result.full_text)

# Line bounding boxes preserve global document coordinates for easy mapping:
for line in result.lines:
    print(f"Global Document Box: ({line.box.x1}, {line.box.y1}) -> ({line.box.x2}, {line.box.y2})")
    if line.relative_box:
        print(f"Crop-Relative Box: ({line.relative_box.x1}, {line.relative_box.y1})")

# You can also filter lines from an already-processed full document:
header_lines = result.get_lines_in_rect((0, 0, 800, 150))
```

> [!NOTE]
> **ROI / Cropping Best Practice**:
> Always keep crop rectangles focused on textual areas with a 10–15px padding margin. Avoid including high-contrast jagged outer card frames or decorative borders, as text detectors may mistake repetitive border teeth for glyphs.

### 6. Processing Multi-Page PDFs with Page Selection & Custom DPI

```python
# Process pages 1 and 2 at 200 DPI for high-resolution receipt OCR
result = ocr.process("receipts.pdf", dpi=200, page_indices=[0, 1])

for page in result.pages:
    print(f"--- Page {page.page_num} ---")
    print(page.text)
```

### 7. Language Detection

```python
from local_ocr import detect_language, detect_language_detailed

text = "2020年11月20日（金）領収書"
lang = detect_language(text)
print(lang)  # 'ja'

code, stats = detect_language_detailed(text)
print(f"Code: {code}, Stats: {stats}")
```

### 8. Export Formats (JSON, Markdown, Sidecars)

```python
# Export to JSON string
json_str = result.to_json(indent=2)

# Export as formatted Markdown
from local_ocr import to_markdown, to_sidecar
markdown_content = to_markdown(result)

# Write a .ocr.txt sidecar next to the file
sidecar_file = to_sidecar(result)
```

### 9. Backward Compatibility with UltimateSorter

`LocalOCR` provides `OCREngine` as a drop-in alias and supports dictionary access on results (`result["full_text"]`, `result["language"]`, `result["lines"]`):

```python
from local_ocr import OCREngine

engine = OCREngine()
res = engine.process_file("document.png")
print(res["full_text"])
print(res["language"])
```

---

## Command-Line Interface (CLI)

After installation, the `local-ocr` command is available in your shell or terminal.

### 1. Process Single File & Rectangle Crops

```bash
# Print OCR text to terminal
local-ocr process document.pdf

# Process Japanese document / card
local-ocr process card.png --lang ja

# Fast single-pass OCR on clean upright scans
local-ocr process document.png --no-orthogonal

# OCR a specific rectangle (x1 y1 x2 y2)
local-ocr process invoice.png --rect 100 200 500 400

# Save structured JSON output with bounding boxes
local-ocr process invoice.png -f json -o output.json

# Save as formatted Markdown
local-ocr process scan.pdf -f markdown -o output.md

# Generate a .ocr.txt sidecar file next to the input file
local-ocr process receipt.png -f sidecar
```

### 2. Batch Process a Directory

Process an entire directory of mixed images and PDFs:

```bash
# Generate .ocr.txt sidecar files for every image and PDF in the folder
local-ocr batch ./Scans_Folder --sidecars

# Batch process Japanese documents
local-ocr batch ./Japanese_Scans --lang ja --sidecars

# OCR a specific region (e.g. receipt header) across all files in a folder
local-ocr batch ./Scans_Folder --rect 50 20 600 200 --output-json headers.json

# Recursively scan subdirectories and save a combined JSON summary report
local-ocr batch ./Scans_Folder --recursive --output-json summary.json

# Filter specific extensions
local-ocr batch ./Scans_Folder --extensions pdf,png
```

### 3. Detect Language

```bash
# Detect language from text argument
local-ocr detect-lang "Facture d'électricité pour le mois de mai"

# Detect language from existing document or text file
local-ocr detect-lang ./receipt.png
```

---

## Architecture & Design

- **Zero-disk In-Memory Buffers**: PyMuPDF renders PDF pixmaps directly into PNG bytes in memory, feeding RapidOCR without writing intermediate files to disk.
- **Orientation Normalization**: Candidate angles (0°, 90°, 180°, 270°) are evaluated with smart deduplication and composite language scoring to ensure upside-down or sideways scans are properly recognized.
- **Typed & Extensible**: Modular structure (`models.py`, `document.py`, `language.py`, `engine.py`, `cli.py`) makes it simple to extend to additional OCR backends or formats in the future.

---

## 📄 License & Commercial Rights

Copyright (c) 2026 **horohoro** ([github.com/horohoro](https://github.com/horohoro)).

This repository is licensed under **[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE)**:
- **Personal / Non-Commercial Use**: You are free to use, copy, modify, and study this project for personal, academic, or non-commercial purposes.
- **Commercial Use**: Any commercial use, commercial distribution, or incorporation into a paid product/service is strictly prohibited without explicit permission or a commercial license agreement.

📩 **For commercial licensing inquiries**: Please contact **horohoro** directly on GitHub (https://github.com/horohoro).
