import os
import re
import time
import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

try:
    from rapidocr_onnxruntime import RapidOCR
    from rapidocr_onnxruntime.utils import UpdateParameters

    # Patch RapidOCR's update_rec_params bug where rec_keys_path prefix is not stripped
    _orig_update_rec_params = UpdateParameters.update_rec_params
    def _patched_update_rec_params(self, config, rec_dict):
        if rec_dict and "rec_keys_path" in rec_dict:
            rec_dict = dict(rec_dict)
            rec_dict["keys_path"] = rec_dict.pop("rec_keys_path")
        return _orig_update_rec_params(self, config, rec_dict)
    UpdateParameters.update_rec_params = _patched_update_rec_params
except ImportError:
    RapidOCR = None

from .models import OCRResult, OCRPage, OCRLine, BoundingBox
from .document import DocumentProcessor
from .language import detect_language, detect_language_detailed

logger = logging.getLogger("local_ocr")

class LocalOCR:
    """
    Main Local OCR Engine powered by RapidOCR ONNX runtime.
    100% offline, local-only OCR supporting in-memory images, multi-page PDFs,
    arbitrary ROI rectangles, multiple language models (ch/en, ja), and
    multi-angle orthogonal orientation verification.
    """

    def __init__(
        self,
        lang: str = "ch",
        use_angle_cls: bool = True,
        compare_orthogonal_base: bool = True,
        compare_orthogonal_angle_cls: bool = True,
        **rapidocr_kwargs
    ):
        if RapidOCR is None:
            raise RuntimeError(
                "rapidocr-onnxruntime is not installed. Please install via: pip install rapidocr-onnxruntime"
            )
        self.lang = lang.lower()
        self.use_angle_cls = use_angle_cls
        self.compare_orthogonal_base = compare_orthogonal_base
        self.compare_orthogonal_angle_cls = compare_orthogonal_angle_cls
        self.rapidocr_kwargs = rapidocr_kwargs

        self.engine = self._create_rapidocr(self.lang, use_angle_cls=use_angle_cls, **rapidocr_kwargs)

    @classmethod
    def _create_rapidocr(cls, lang: str, use_angle_cls: bool = True, **kwargs) -> Any:
        opts = dict(kwargs)
        if lang == "ja":
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            model_path = os.path.join(models_dir, "japan_PP-OCRv4_rec_infer.onnx")
            keys_path = os.path.join(models_dir, "japan_dict.txt")
            if "rec_model_path" not in opts and os.path.exists(model_path):
                opts["rec_model_path"] = model_path
            if "rec_keys_path" not in opts and os.path.exists(keys_path):
                opts["rec_keys_path"] = keys_path
        return RapidOCR(use_angle_cls=use_angle_cls, **opts)

    @staticmethod
    def _rotate_crop(crop: np.ndarray, angle: int) -> np.ndarray:
        angle = int(angle) % 360
        if angle == 0:
            return crop
        elif angle == 90:
            return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(crop, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            h, w = crop.shape[:2]
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, -float(angle), 1.0)
            return cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _unrotate_polygon(polygon: List[List[float]], angle: int, orig_w: int, orig_h: int) -> List[List[float]]:
        unrot = []
        angle = int(angle) % 360
        for pt in polygon:
            x, y = float(pt[0]), float(pt[1])
            if angle == 90:
                unrot.append([y, float(orig_h - 1 - x)])
            elif angle == 180:
                unrot.append([float(orig_w - 1 - x), float(orig_h - 1 - y)])
            elif angle == 270:
                unrot.append([float(orig_w - 1 - y), x])
            else:
                unrot.append([x, y])
        return unrot

    def _score_candidate(self, text: str, conf: float) -> float:
        if not text or conf <= 0:
            return 0.0
        total_len = len(text)
        if total_len == 0:
            return 0.0

        if self.lang == "ja":
            kana_count = len(re.findall(r'[\u3040-\u30ff]', text))
            kanji_count = len(re.findall(r'[\u4e00-\u9fff]', text))
            latin_count = len(re.findall(r'[A-Za-z]', text))
            bonus = 1.5 * (kana_count / total_len) + 0.8 * (kanji_count / total_len) + 0.2 * (latin_count / total_len)
        else:
            words = len(re.findall(r'[A-Za-z0-9\u4e00-\u9fff]', text))
            bonus = 1.0 * (words / total_len)

        return conf * (1.0 + bonus)

    def _recognize_boxes(self, img: np.ndarray, dt_boxes: Any) -> Optional[List[List[Any]]]:
        img_crop_list = self.engine.get_crop_img_list(img, dt_boxes)
        if not img_crop_list:
            return None

        base_angles = set()
        if self.compare_orthogonal_base:
            base_angles.update([0, 90, 180, 270])

        if self.use_angle_cls and hasattr(self.engine, "text_cls") and self.engine.text_cls is not None:
            _, cls_res, _ = self.engine.text_cls(img_crop_list)
        else:
            cls_res = [["0", 1.0]] * len(img_crop_list)

        results = []
        for i, (box, crop) in enumerate(zip(dt_boxes, img_crop_list)):
            angles = set(base_angles)
            if self.use_angle_cls:
                label = str(cls_res[i][0])
                det_angle = 180 if "180" in label else 0
                if self.compare_orthogonal_angle_cls:
                    angles.update([(det_angle + off) % 360 for off in [0, 90, 180, 270]])
                else:
                    angles.add(det_angle)

            if not angles:
                angles = {0}

            unique_angles = sorted(angles)
            if len(unique_angles) == 1 and unique_angles[0] == 0:
                rec_res, _ = self.engine.text_recognizer([crop])
                best_txt, best_conf = rec_res[0][0], float(rec_res[0][1])
            else:
                cand_crops = [self._rotate_crop(crop, ang) for ang in unique_angles]
                cand_rec, _ = self.engine.text_recognizer(cand_crops)

                best_txt = ""
                best_conf = 0.0
                best_score = -1.0
                for ang, (txt, conf) in zip(unique_angles, cand_rec):
                    c_conf = float(conf)
                    s = self._score_candidate(txt, c_conf)
                    if s > best_score:
                        best_score = s
                        best_txt = txt
                        best_conf = c_conf

            if best_conf >= self.engine.text_score and best_txt:
                box_data = box.tolist() if hasattr(box, "tolist") else box
                results.append([box_data, best_txt, str(best_conf)])

        return results if results else None

    def _run_ocr(self, img_input: Any) -> Optional[List[List[Any]]]:
        # Fast path: when multi-angle orthogonal comparisons are disabled
        if not self.compare_orthogonal_base and not self.compare_orthogonal_angle_cls:
            raw_results, _ = self.engine(img_input)
            return raw_results

        # Multi-angle verification path
        img = self.engine.load_img(img_input)
        h_orig, w_orig = img.shape[:2]
        dt_boxes, _ = self.engine.text_detector(img)

        # Fallback 1: if detector found 0 boxes and image looks like a single text strip/crop
        if (dt_boxes is None or len(dt_boxes) < 1) and (h_orig <= 120 or (w_orig / max(1, h_orig)) >= 1.8):
            dt_boxes, _ = self.engine.get_boxes_img_without_det(img, h_orig, w_orig)

        # Fallback 2: if detector found 0 boxes and orthogonal comparison is enabled, check 90/180/270 image rotation
        if (dt_boxes is None or len(dt_boxes) < 1) and self.compare_orthogonal_base:
            for rot_deg, rot_flag in [
                (90, cv2.ROTATE_90_CLOCKWISE),
                (180, cv2.ROTATE_180),
                (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ]:
                rot_img = cv2.rotate(img, rot_flag)
                dt_boxes_rot, _ = self.engine.text_detector(rot_img)
                if dt_boxes_rot is not None and len(dt_boxes_rot) >= 1:
                    rot_results = self._recognize_boxes(rot_img, self.engine.sorted_boxes(dt_boxes_rot))
                    if rot_results:
                        unrot_results = []
                        for item in rot_results:
                            unrot_poly = self._unrotate_polygon(item[0], rot_deg, w_orig, h_orig)
                            unrot_results.append([unrot_poly, item[1], item[2]])
                        return unrot_results

        if dt_boxes is None or len(dt_boxes) < 1:
            return None

        dt_boxes = self.engine.sorted_boxes(dt_boxes)
        return self._recognize_boxes(img, dt_boxes)

    def process(
        self,
        source: Union[str, bytes],
        dpi: int = 150,
        page_indices: Optional[List[int]] = None,
        rect: Optional[Union[Tuple[float, float, float, float], List[float], BoundingBox, str, Dict[str, float]]] = None
    ) -> OCRResult:
        """
        Process an image or PDF (path or raw bytes) and return an OCRResult.
        If rect is specified (x1, y1, x2, y2), extracts text only within that bounding box.
        """
        start_time = time.perf_counter()
        file_path = source if isinstance(source, str) else None

        pages: List[OCRPage] = []
        all_lines: List[OCRLine] = []
        all_text_parts: List[str] = []

        for frame in DocumentProcessor.get_pages(source, dpi=dpi, page_indices=page_indices, rect=rect):
            page_num, img_input, w, h = frame[0], frame[1], frame[2], frame[3]
            offset_x = getattr(frame, "offset_x", 0.0)
            offset_y = getattr(frame, "offset_y", 0.0)

            raw_results = self._run_ocr(img_input)

            page_lines: List[OCRLine] = []
            page_text_parts: List[str] = []

            if raw_results:
                for item in raw_results:
                    # RapidOCR returns [polygon_pts, text_str, confidence_val]
                    polygon = item[0]
                    txt = str(item[1])
                    try:
                        conf = float(item[2])
                    except (ValueError, TypeError):
                        conf = 0.0

                    rel_bbox = BoundingBox.from_polygon(polygon)
                    if offset_x != 0.0 or offset_y != 0.0:
                        abs_polygon = [[pt[0] + offset_x, pt[1] + offset_y] for pt in polygon]
                        abs_bbox = BoundingBox.from_polygon(abs_polygon)
                    else:
                        abs_bbox = rel_bbox

                    line_obj = OCRLine(
                        text=txt,
                        confidence=conf,
                        page=page_num,
                        box=abs_bbox,
                        relative_box=rel_bbox if (offset_x != 0.0 or offset_y != 0.0) else None
                    )
                    page_lines.append(line_obj)
                    all_lines.append(line_obj)

                    page_text_parts.append(txt)
                    all_text_parts.append(txt)

            page_text = "\n".join(page_text_parts)
            pages.append(
                OCRPage(
                    page_num=page_num,
                    text=page_text,
                    lines=page_lines,
                    width=w,
                    height=h
                )
            )

        full_text = "\n".join(all_text_parts)
        lang, lang_stats = detect_language_detailed(full_text)
        duration = time.perf_counter() - start_time

        metadata: Dict[str, Any] = {"lang_stats": lang_stats, "dpi": dpi}
        if rect is not None:
            parsed_rect = BoundingBox.parse(rect)
            metadata["crop_rect"] = parsed_rect.to_list()

        return OCRResult(
            file_path=file_path,
            full_text=full_text,
            language=lang,
            language_confidence=lang_stats.get("confidence", 1.0),
            pages=pages,
            lines=all_lines,
            page_count=len(pages),
            duration_seconds=duration,
            metadata=metadata
        )

    def process_rect(
        self,
        source: Union[str, bytes],
        rect: Union[Tuple[float, float, float, float], List[float], BoundingBox, str, Dict[str, float]],
        dpi: int = 150,
        page_indices: Optional[List[int]] = None
    ) -> OCRResult:
        """
        Convenience method to OCR a specific rectangle (x1, y1, x2, y2) in an image or PDF.
        Coordinates are in pixel values (at the specified DPI for PDFs).
        """
        return self.process(source, dpi=dpi, page_indices=page_indices, rect=rect)

    def process_file(self, file_path: str) -> OCRResult:
        """
        Backward-compatible method matching UltimateSorter's original OCREngine API.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return self.process(file_path)

    @staticmethod
    def detect_language(text: str) -> str:
        """Static helper to detect language of text."""
        return detect_language(text)

# Direct alias for seamless backward compatibility
OCREngine = LocalOCR
