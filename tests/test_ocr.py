import unittest
import os
import sys
import tempfile
import subprocess
import json

# Ensure local_ocr is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_ocr import LocalOCR, OCREngine, detect_language, detect_language_detailed, to_text, to_markdown, to_sidecar
from local_ocr.models import OCRResult, OCRLine, OCRPage, BoundingBox

class TestLanguageDetection(unittest.TestCase):
    def test_japanese_detection(self):
        text = "2020年11月20日（金）領収書 芝浦店 お買い上げありがとうございます"
        self.assertEqual(detect_language(text), "ja")

    def test_french_detection(self):
        text = "Facture d'électricité et attestation de contrat pour le mois de décembre."
        self.assertEqual(detect_language(text), "fr")

    def test_english_detection(self):
        text = "Invoice and bank statement summary for account number 123456789."
        self.assertEqual(detect_language(text), "en")

    def test_empty_detection(self):
        self.assertEqual(detect_language(""), "unknown")

    def test_detailed_detection(self):
        code, stats = detect_language_detailed("住民票の写し")
        self.assertEqual(code, "ja")
        self.assertGreater(stats.get("kana_count", 0) + stats.get("cjk_count", 0), 0)

class TestDataModels(unittest.TestCase):
    def test_bounding_box(self):
        poly = [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]
        box = BoundingBox.from_polygon(poly)
        self.assertEqual(box.x1, 10.0)
        self.assertEqual(box.y1, 20.0)
        self.assertEqual(box.x2, 50.0)
        self.assertEqual(box.y2, 40.0)
        self.assertEqual(box.width, 40.0)
        self.assertEqual(box.height, 20.0)

    def test_dict_compatibility(self):
        res = OCRResult(
            file_path="test.pdf",
            full_text="Sample text",
            language="en",
            page_count=1,
            lines=[OCRLine(text="Sample text", confidence=0.95, page=1)]
        )
        # Test backward-compatible dict access
        self.assertEqual(res["full_text"], "Sample text")
        self.assertEqual(res["language"], "en")
        self.assertEqual(res["page_count"], 1)
        self.assertEqual(res["file_path"], "test.pdf")
        self.assertEqual(len(res["lines"]), 1)
        self.assertEqual(res["lines"][0]["text"], "Sample text")
        self.assertEqual(res.get("language"), "en")
        self.assertIn("full_text", res)

class TestOCREngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LocalOCR()

    def test_in_memory_pdf_ocr(self):
        import pymupdf
        # Create an in-memory 2-page PDF
        doc = pymupdf.open()
        p1 = doc.new_page(width=300, height=100)
        p1.insert_text((20, 50), "First Page Test", fontsize=20)
        p2 = doc.new_page(width=300, height=100)
        p2.insert_text((20, 50), "Second Page Receipt", fontsize=20)
        pdf_bytes = doc.tobytes()
        doc.close()

        # Run OCR on in-memory PDF bytes
        result = self.engine.process(pdf_bytes)
        self.assertEqual(result.page_count, 2)
        self.assertIn("First Page", result.full_text)
        self.assertIn("Second Page", result.full_text)

    def test_sample_image_ocr(self):
        from PIL import Image, ImageDraw
        import io
        img = Image.new("RGB", (300, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 40), "Sample Text Line", fill="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = self.engine.process(buf.getvalue())
        self.assertIsNotNone(result.full_text)
        self.assertIn("Sample", result.full_text)
        self.assertGreater(len(result.lines), 0)
        self.assertIsNotNone(result.lines[0].box)

class TestFormatters(unittest.TestCase):
    def test_markdown_and_sidecar(self):
        res = OCRResult(
            file_path="dummy.png",
            full_text="Line one\nLine two",
            language="en",
            page_count=1,
            pages=[OCRPage(page_num=1, text="Line one\nLine two")]
        )
        md = to_markdown(res)
        self.assertIn("## Page 1", md)
        self.assertIn("Line one", md)

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "test.ocr.txt")
            sidecar = to_sidecar(res, destination_path=dest)
            self.assertTrue(os.path.exists(sidecar))
            with open(sidecar, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "Line one\nLine two")

class TestCLIExecution(unittest.TestCase):
    def test_cli_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "local_ocr.cli", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("local-ocr", proc.stdout)
        self.assertIn("process", proc.stdout)
        self.assertIn("batch", proc.stdout)

    def test_cli_detect_lang(self):
        proc = subprocess.run(
            [sys.executable, "-m", "local_ocr.cli", "detect-lang", "Bonjour le monde"],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Language: fr", proc.stdout)

    def test_cli_process_image(self):
        sample_path = os.path.join(os.path.dirname(__file__), "assets", "card3_ja.png")
        if os.path.exists(sample_path):
            proc = subprocess.run(
                [sys.executable, "-m", "local_ocr.cli", "process", sample_path, "--lang", "ja", "-f", "json"],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            self.assertEqual(proc.returncode, 0)
            data = json.loads(proc.stdout)
            self.assertEqual(data["language"], "ja")
            self.assertIn("lines", data)

    def test_direct_cli_executable(self):
        import shutil
        import site
        exe_path = shutil.which("local-ocr") or shutil.which("local-ocr.exe")
        if not exe_path:
            candidate = os.path.join(
                site.USER_BASE,
                f"Python{sys.version_info.major}{sys.version_info.minor}",
                "Scripts",
                "local-ocr.exe"
            )
            if os.path.exists(candidate):
                exe_path = candidate

        if exe_path and os.path.exists(exe_path):
            proc = subprocess.run([exe_path, "--help"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("local-ocr", proc.stdout)


class TestRectangleOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LocalOCR()

    def test_bounding_box_parsing_and_spatial(self):
        # 1. From tuple
        b1 = BoundingBox.parse((10, 20, 100, 200))
        self.assertEqual(b1.to_tuple(), (10.0, 20.0, 100.0, 200.0))
        self.assertEqual(b1.to_list(), [10.0, 20.0, 100.0, 200.0])

        # 2. Inverted coordinates normalization
        b_inv = BoundingBox.parse((100, 200, 10, 20))
        self.assertEqual(b_inv.to_tuple(), (10.0, 20.0, 100.0, 200.0))

        # 3. From dict
        b2 = BoundingBox.parse({"x1": 10, "y1": 20, "x2": 100, "y2": 200})
        self.assertEqual(b2.to_tuple(), (10.0, 20.0, 100.0, 200.0))

        # 4. From string (space and comma separated)
        b3 = BoundingBox.parse("10 20 100 200")
        self.assertEqual(b3.to_tuple(), (10.0, 20.0, 100.0, 200.0))
        b4 = BoundingBox.parse("10, 20, 100, 200")
        self.assertEqual(b4.to_tuple(), (10.0, 20.0, 100.0, 200.0))

        # 5. Spatial query methods
        self.assertTrue(b1.contains_point(50, 50))
        self.assertFalse(b1.contains_point(5, 50))
        self.assertFalse(b1.contains_point(50, 250))

        self.assertTrue(b1.intersects(BoundingBox(50, 50, 150, 250)))
        self.assertFalse(b1.intersects(BoundingBox(300, 300, 400, 400)))

    def test_image_rectangle_ocr(self):
        from PIL import Image, ImageDraw
        import io

        # Create synthetic image with 2 separate lines of text
        img = Image.new("RGB", (400, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((30, 30), "UPPER REGION TEXT", fill="black")
        draw.text((30, 130), "LOWER REGION TEXT", fill="black")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # 1. OCR only the upper rectangle: (10, 10, 380, 80)
        res_upper = self.engine.process_rect(img_bytes, rect=(10, 10, 380, 80))
        self.assertIn("UPPER", res_upper.full_text)
        self.assertNotIn("LOWER", res_upper.full_text)
        self.assertIn("crop_rect", res_upper.metadata)
        self.assertGreater(len(res_upper.lines), 0)
        self.assertIsNotNone(res_upper.lines[0].relative_box)
        # Verify line.box coordinates reflect absolute source image coordinates
        self.assertGreaterEqual(res_upper.lines[0].box.y1, 10.0)

        # 2. OCR only the lower rectangle: (10, 100, 380, 180)
        res_lower = self.engine.process(img_bytes, rect="10 100 380 180")
        self.assertIn("LOWER", res_lower.full_text)
        self.assertNotIn("UPPER", res_lower.full_text)

    def test_pdf_rectangle_ocr(self):
        import pymupdf
        # Create an in-memory PDF with header and footer
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=200)
        page.insert_text((30, 40), "CONFIDENTIAL INVOICE 999", fontsize=18)
        page.insert_text((30, 160), "PAYMENT DUE IN 30 DAYS", fontsize=18)
        pdf_bytes = doc.tobytes()
        doc.close()

        # Render DPI is 150. (72 pt -> 150 px).
        # Header is at y=40 pt -> ~83 px.
        # Footer is at y=160 pt -> ~333 px.
        # Crop header: y from 0 to 200 px
        res_header = self.engine.process_rect(pdf_bytes, rect=(0, 0, 800, 200), dpi=150)
        self.assertIn("CONFIDENTIAL", res_header.full_text)
        self.assertNotIn("PAYMENT", res_header.full_text)
        self.assertGreater(len(res_header.lines), 0)
        # Check that line.box preserves absolute coordinates
        self.assertIsNotNone(res_header.lines[0].box)

    def test_get_lines_in_rect_filtering(self):
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=200)
        page.insert_text((30, 40), "Top Header", fontsize=18)
        page.insert_text((30, 160), "Bottom Details", fontsize=18)
        pdf_bytes = doc.tobytes()
        doc.close()

        # Run full OCR
        full_res = self.engine.process(pdf_bytes, dpi=150)
        self.assertIn("Top", full_res.full_text)
        self.assertIn("Bottom", full_res.full_text)

        # Filter lines post-OCR using get_lines_in_rect
        top_lines = full_res.get_lines_in_rect((0, 0, 800, 200))
        top_texts = " ".join(ln.text for ln in top_lines)
        self.assertIn("Top", top_texts)
        self.assertNotIn("Bottom", top_texts)

    def test_cli_rect_argument(self):
        from PIL import Image, ImageDraw
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "crop_test.png")
            img = Image.new("RGB", (400, 200), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((30, 30), "CLI_TOP_SECRET", fill="black")
            draw.text((30, 130), "CLI_BOTTOM_NOTE", fill="black")
            img.save(img_path)

            proc = subprocess.run(
                [sys.executable, "-m", "local_ocr.cli", "process", img_path, "--rect", "10", "10", "380", "80", "-f", "json"],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            self.assertEqual(proc.returncode, 0)
            data = json.loads(proc.stdout)
            self.assertIn("CLI_TOP", data["full_text"])
            self.assertNotIn("CLI_BOTTOM", data["full_text"])
            self.assertIn("crop_rect", data["metadata"])
            self.assertEqual(data["metadata"]["crop_rect"], [10.0, 10.0, 380.0, 80.0])


class TestJapaneseAndOrientation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine_ja = LocalOCR(lang="ja")
        cls.sample = os.path.join(os.path.dirname(__file__), "assets", "card3_ja.png")

    def test_japanese_recognition(self):
        if os.path.exists(self.sample):
            res = self.engine_ja.process(self.sample)
            self.assertEqual(res.language, "ja")
            self.assertIn("生きている", res.full_text)
            self.assertGreater(len(res.lines), 0)

    def test_upside_down_recognition(self):
        import cv2
        if os.path.exists(self.sample):
            img = cv2.imread(self.sample)
            img_180 = cv2.rotate(img, cv2.ROTATE_180)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                cv2.imwrite(tmp_path, img_180)
                res = self.engine_ja.process(tmp_path)
                self.assertIn("生きている", res.full_text)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def test_fast_path_bypass(self):
        if os.path.exists(self.sample):
            fast_engine = LocalOCR(
                lang="ja",
                compare_orthogonal_base=False,
                compare_orthogonal_angle_cls=False,
                use_angle_cls=False
            )
            res = fast_engine.process(self.sample)
            self.assertIn("生きている", res.full_text)

    def test_cli_japanese_and_no_orthogonal(self):
        if os.path.exists(self.sample):
            proc = subprocess.run(
                [sys.executable, "-m", "local_ocr.cli", "process", self.sample, "--lang", "ja", "--no-orthogonal", "-f", "json"],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            self.assertEqual(proc.returncode, 0)
            data = json.loads(proc.stdout)
            self.assertEqual(data["language"], "ja")
            self.assertIn("生きている", data["full_text"])


if __name__ == "__main__":
    unittest.main()


