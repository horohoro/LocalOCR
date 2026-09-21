import sys
import os
import argparse
import json
from typing import List

from .engine import LocalOCR
from .formatters import to_text, to_markdown, to_sidecar
from .language import detect_language_detailed
from .models import BoundingBox

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-ocr",
        description="LocalOCR - Fast, local, private OCR engine for images and PDFs"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: process
    proc_parser = subparsers.add_parser("process", help="Extract text from an image or PDF file")
    proc_parser.add_argument("file", help="Path to input image or PDF file")
    proc_parser.add_argument("-o", "--output", help="Write result to file instead of stdout")
    proc_parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "markdown", "sidecar"],
        default="text",
        help="Output format (default: text)"
    )
    proc_parser.add_argument("--dpi", type=int, default=150, help="Rendering DPI for PDF pages (default: 150)")
    proc_parser.add_argument("-l", "--lang", default="ch", choices=["ch", "ja"], help="Language model: 'ch' (Chinese/English, default) or 'ja' (Japanese)")
    proc_parser.add_argument("--no-angle-cls", action="store_true", help="Disable orientation angle classification")
    proc_parser.add_argument("--no-orthogonal", action="store_true", help="Disable multi-angle orthogonal verification for speed on clean upright documents")
    proc_parser.add_argument(
        "--rect",
        nargs="+",
        help="OCR a specific rectangle (e.g. --rect x1 y1 x2 y2 or --rect x1,y1,x2,y2)"
    )

    # Command: batch
    batch_parser = subparsers.add_parser("batch", help="Batch process a directory of files")
    batch_parser.add_argument("directory", help="Directory containing images and PDFs")
    batch_parser.add_argument("-r", "--recursive", action="store_true", help="Scan directory recursively")
    batch_parser.add_argument("-l", "--lang", default="ch", choices=["ch", "ja"], help="Language model: 'ch' or 'ja' (default: ch)")
    batch_parser.add_argument("--no-angle-cls", action="store_true", help="Disable orientation angle classification")
    batch_parser.add_argument("--no-orthogonal", action="store_true", help="Disable multi-angle orthogonal verification")
    batch_parser.add_argument("--sidecars", action="store_true", help="Write .ocr.txt sidecar files next to each input file")
    batch_parser.add_argument("-o", "--output-json", help="Save combined results to a single JSON file")
    batch_parser.add_argument(
        "--extensions",
        default="pdf,png,jpg,jpeg,tiff,tif,bmp,webp",
        help="Comma-separated file extensions to process (default: pdf,png,jpg,jpeg,tiff,tif,bmp,webp)"
    )
    batch_parser.add_argument("--dpi", type=int, default=150, help="Rendering DPI for PDF pages (default: 150)")
    batch_parser.add_argument(
        "--rect",
        nargs="+",
        help="OCR a specific rectangle across all files (e.g. --rect x1 y1 x2 y2)"
    )

    # Command: detect-lang
    lang_parser = subparsers.add_parser("detect-lang", help="Detect language of given text or file")
    lang_parser.add_argument("input", help="Text string or path to text/image/pdf file")

    return parser

def cmd_process(args) -> int:
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    target_rect = None
    if getattr(args, "rect", None):
        try:
            target_rect = BoundingBox.parse(args.rect)
        except Exception as e:
            print(f"Error: Invalid --rect argument: {e}", file=sys.stderr)
            return 1

    engine = LocalOCR(
        lang=getattr(args, "lang", "ch"),
        use_angle_cls=not getattr(args, "no_angle_cls", False),
        compare_orthogonal_base=not getattr(args, "no_orthogonal", False),
        compare_orthogonal_angle_cls=not getattr(args, "no_orthogonal", False)
    )
    res = engine.process(args.file, dpi=args.dpi, rect=target_rect)

    if args.format == "text":
        out_content = to_text(res)
    elif args.format == "json":
        out_content = res.to_json(indent=2)
    elif args.format == "markdown":
        out_content = to_markdown(res)
    elif args.format == "sidecar":
        sidecar_path = to_sidecar(res, args.output)
        print(f"Sidecar created: {sidecar_path}")
        return 0

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_content)
        print(f"Output saved to: {args.output}")
    else:
        print(out_content)

    return 0

def cmd_batch(args) -> int:
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
        return 1

    target_rect = None
    if getattr(args, "rect", None):
        try:
            target_rect = BoundingBox.parse(args.rect)
        except Exception as e:
            print(f"Error: Invalid --rect argument: {e}", file=sys.stderr)
            return 1

    valid_exts = {f".{ext.strip().lower().lstrip('.')}" for ext in args.extensions.split(",")}
    matched_files: List[str] = []

    if args.recursive:
        for root, _, files in os.walk(args.directory):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts and not f.endswith(".ocr.txt"):
                    matched_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(args.directory):
            full_p = os.path.join(args.directory, f)
            if os.path.isfile(full_p) and os.path.splitext(f)[1].lower() in valid_exts and not f.endswith(".ocr.txt"):
                matched_files.append(full_p)

    if not matched_files:
        print(f"No matching files found in {args.directory} with extensions {args.extensions}")
        return 0

    print(f"Processing {len(matched_files)} file(s)...")
    engine = LocalOCR(
        lang=getattr(args, "lang", "ch"),
        use_angle_cls=not getattr(args, "no_angle_cls", False),
        compare_orthogonal_base=not getattr(args, "no_orthogonal", False),
        compare_orthogonal_angle_cls=not getattr(args, "no_orthogonal", False)
    )
    results = []

    for i, file_path in enumerate(matched_files, 1):
        print(f"[{i}/{len(matched_files)}] {os.path.basename(file_path)} ...", end="", flush=True)
        try:
            res = engine.process(file_path, dpi=args.dpi, rect=target_rect)
            results.append(res.to_dict())
            if args.sidecars:
                to_sidecar(res)
            print(f" OK ({res.language}, {res.duration_seconds:.2f}s)")
        except Exception as e:
            print(f" ERROR: {e}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nBatch results saved to: {args.output_json}")

    print(f"Batch processing completed: {len(results)}/{len(matched_files)} succeeded.")
    return 0

def cmd_detect_lang(args) -> int:
    text = args.input
    if os.path.isfile(args.input):
        ext = os.path.splitext(args.input)[1].lower()
        if ext in {".txt", ".ocr.txt", ".md"}:
            with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            engine = LocalOCR()
            res = engine.process(args.input)
            text = res.full_text

    lang, stats = detect_language_detailed(text)
    print(f"Language: {lang} (confidence: {stats.get('confidence', 0.0):.2f})")
    print("Character breakdown:")
    for k, v in stats.items():
        if k != "confidence":
            print(f"  - {k}: {v}")
    return 0

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "process":
        sys.exit(cmd_process(args))
    elif args.command == "batch":
        sys.exit(cmd_batch(args))
    elif args.command == "detect-lang":
        sys.exit(cmd_detect_lang(args))

if __name__ == "__main__":
    main()
