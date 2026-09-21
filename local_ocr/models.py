from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

@dataclass
class BoundingBox:
    """Bounding box coordinates for a detected line or word."""
    x1: float
    y1: float
    x2: float
    y2: float
    polygon: List[List[float]] = field(default_factory=list)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is within this bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bounding box intersects with another."""
        return not (
            self.x2 < other.x1 or
            self.x1 > other.x2 or
            self.y2 < other.y1 or
            self.y1 > other.y2
        )

    @classmethod
    def from_polygon(cls, polygon: List[List[float]]) -> "BoundingBox":
        """Create BoundingBox from 4 corner points [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]."""
        if not polygon:
            return cls(0.0, 0.0, 0.0, 0.0, [])
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        return cls(
            x1=float(min(xs)),
            y1=float(min(ys)),
            x2=float(max(xs)),
            y2=float(max(ys)),
            polygon=polygon
        )

    @classmethod
    def parse(cls, val: Any) -> "BoundingBox":
        """
        Parse various rectangle representations into a BoundingBox.
        Supported inputs:
        - BoundingBox instance
        - Tuple or list of 4 numbers: (x1, y1, x2, y2)
        - Dict with keys 'x1', 'y1', 'x2', 'y2'
        - String: "x1, y1, x2, y2" or "x1 y1 x2 y2"
        """
        if isinstance(val, BoundingBox):
            return cls(val.x1, val.y1, val.x2, val.y2, list(val.polygon))

        if isinstance(val, dict):
            raw = [val["x1"], val["y1"], val["x2"], val["y2"]]
        elif isinstance(val, str):
            cleaned = val.strip().strip("()[]{}")
            parts = [p.strip() for p in cleaned.replace(",", " ").split() if p.strip()]
            if len(parts) != 4:
                raise ValueError(f"Cannot parse rectangle from string '{val}'. Expected 4 coordinates (x1 y1 x2 y2).")
            raw = [float(p) for p in parts]
        elif hasattr(val, "__iter__"):
            raw = list(val)
            # Flatten if nested e.g. from argparse [['10', '20', '30', '40']] or single string inside list
            if len(raw) == 1 and isinstance(raw[0], str):
                return cls.parse(raw[0])
            if len(raw) != 4:
                raise ValueError(f"Expected 4 rectangle coordinates, got {len(raw)}: {val}")
            raw = [float(x) for x in raw]
        else:
            raise TypeError(f"Unsupported rectangle type: {type(val)}. Expected tuple, list, dict, str, or BoundingBox.")

        x1, y1, x2, y2 = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
        # Normalize in case coordinates were inverted
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        polygon = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]
        return cls(x1=min_x, y1=min_y, x2=max_x, y2=max_y, polygon=polygon)

    @classmethod
    def from_rect(cls, rect: Any) -> "BoundingBox":
        """Alias for parse()."""
        return cls.parse(rect)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "polygon": self.polygon
        }


@dataclass
class OCRLine:
    """Represents a recognized line of text."""
    text: str
    confidence: float
    page: int = 1
    box: Optional[BoundingBox] = None
    relative_box: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "page": self.page,
        }
        if self.box:
            d["box"] = self.box.to_dict()
        if self.relative_box:
            d["relative_box"] = self.relative_box.to_dict()
        return d

    # Allow dict-like access for backward compatibility: line["text"], line["confidence"]
    def __getitem__(self, item: str) -> Any:
        if item == "text":
            return self.text
        elif item == "confidence":
            return self.confidence
        elif item == "page":
            return self.page
        elif item == "box":
            return self.box.to_dict() if self.box else None
        elif item == "relative_box":
            return self.relative_box.to_dict() if self.relative_box else None
        raise KeyError(item)


@dataclass
class OCRPage:
    """Represents OCR results for an individual page."""
    page_num: int
    text: str = ""
    lines: List[OCRLine] = field(default_factory=list)
    width: Optional[float] = None
    height: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_num": self.page_num,
            "text": self.text,
            "lines": [ln.to_dict() for ln in self.lines],
            "width": self.width,
            "height": self.height,
        }


@dataclass
class OCRResult:
    """
    Comprehensive result of an OCR extraction.
    Implements dict interface (__getitem__, get) for full backward compatibility
    with callers expecting a dict return value.
    """
    file_path: Optional[str] = None
    full_text: str = ""
    language: str = "unknown"
    language_confidence: float = 1.0
    pages: List[OCRPage] = field(default_factory=list)
    lines: List[OCRLine] = field(default_factory=list)
    page_count: int = 1
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Dictionary compatibility methods
    def __getitem__(self, key: str) -> Any:
        if key == "full_text":
            return self.full_text
        elif key == "language":
            return self.language
        elif key == "lines":
            return [ln.to_dict() for ln in self.lines]
        elif key == "file_path":
            return self.file_path
        elif key == "page_count":
            return self.page_count
        elif key == "pages":
            return [p.to_dict() for p in self.pages]
        elif key == "duration_seconds":
            return self.duration_seconds
        elif key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in ("full_text", "language", "lines", "file_path", "page_count", "pages", "duration_seconds") or key in self.metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "full_text": self.full_text,
            "language": self.language,
            "language_confidence": round(self.language_confidence, 4),
            "page_count": self.page_count,
            "duration_seconds": round(self.duration_seconds, 4),
            "lines": [ln.to_dict() for ln in self.lines],
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_text(self) -> str:
        return self.full_text

    def get_lines_in_rect(self, rect: Any, page: Optional[int] = None) -> List[OCRLine]:
        """
        Filter lines that intersect or fall inside the specified rectangle.
        :param rect: BoundingBox, (x1, y1, x2, y2), dict, or string "x1 y1 x2 y2"
        :param page: Optional 1-based page number to filter by
        """
        target_box = BoundingBox.parse(rect)
        matching: List[OCRLine] = []
        for line in self.lines:
            if page is not None and line.page != page:
                continue
            if line.box and line.box.intersects(target_box):
                matching.append(line)
        return matching

