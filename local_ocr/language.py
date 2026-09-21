from typing import Dict, Any, Tuple

# Pre-defined character sets for linguistic heuristics
FRENCH_ACCENTS = set("éèêëàâäôöûüçÉÈÊËÀÂÄÔÖÛÜÇœæ")
GERMAN_CHARS = set("äöüßÄÖÜ")
SPANISH_CHARS = set("áéíóúñÁÉÍÓÚÑ¿¡")

def detect_language(text: str) -> str:
    """
    Fast heuristic language detector returning standard ISO 639-1 code:
    - 'ja': Japanese (Hiragana, Katakana, Kanji)
    - 'zh': Chinese (Kanji / Hanzi without Kana)
    - 'fr': French (French accented characters or high accent frequency)
    - 'de': German (German umlauts and eszett)
    - 'es': Spanish (Spanish accents, inverted punctuation, ñ)
    - 'en': English (predominantly ASCII Latin)
    - 'unknown': Empty or unidentifiable
    """
    code, _ = detect_language_detailed(text)
    return code

def detect_language_detailed(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Detailed heuristic language detection returning (lang_code, stats).
    """
    if not text or len(text.strip()) == 0:
        return "unknown", {"total_chars": 0, "confidence": 0.0}

    text_stripped = text.strip()
    total_chars = len(text_stripped)

    kana_count = 0
    cjk_count = 0
    fr_count = 0
    de_count = 0
    es_count = 0
    ascii_alpha_count = 0

    for char in text_stripped:
        code = ord(char)
        # Hiragana (0x3040-0x309F) or Katakana (0x30A0-0x30FF)
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
            kana_count += 1
        # CJK Unified Ideographs (0x4E00-0x9FFF)
        elif 0x4E00 <= code <= 0x9FFF:
            cjk_count += 1
        elif char in FRENCH_ACCENTS:
            fr_count += 1
        elif char in GERMAN_CHARS:
            de_count += 1
        elif char in SPANISH_CHARS:
            es_count += 1
        elif ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
            ascii_alpha_count += 1

    stats = {
        "total_chars": total_chars,
        "kana_count": kana_count,
        "cjk_count": cjk_count,
        "fr_count": fr_count,
        "de_count": de_count,
        "es_count": es_count,
        "ascii_alpha_count": ascii_alpha_count,
    }

    # Japanese check: Any Kana present, or CJK + Kana
    if kana_count > 0 or (cjk_count > 0 and (kana_count + cjk_count) / total_chars > 0.05):
        confidence = min(1.0, (kana_count * 2 + cjk_count) / max(1, total_chars))
        stats["confidence"] = confidence
        return "ja", stats

    # Chinese check: CJK characters without Kana
    if cjk_count > 0 and (cjk_count / total_chars > 0.15):
        stats["confidence"] = min(1.0, cjk_count / total_chars)
        return "zh", stats

    # French check
    if fr_count > 0:
        confidence = min(1.0, fr_count / max(1, total_chars) * 10)
        stats["confidence"] = confidence
        return "fr", stats

    # German check
    if de_count > 0:
        confidence = min(1.0, de_count / max(1, total_chars) * 10)
        stats["confidence"] = confidence
        return "de", stats

    # Spanish check
    if es_count > 0:
        confidence = min(1.0, es_count / max(1, total_chars) * 10)
        stats["confidence"] = confidence
        return "es", stats

    # Word-level heuristic check if no special accents are present
    words = {w.strip(".,;:!?\"'()[]{}«»").lower() for w in text_stripped.split()}
    fr_common = {"bonjour", "merci", "le", "la", "les", "des", "du", "dans", "pour", "avec", "est", "une", "facture", "attestation"}
    de_common = {"hallo", "danke", "der", "die", "das", "und", "nicht", "mit", "ist", "rechnung"}
    es_common = {"hola", "gracias", "por", "para", "con", "una", "los", "las", "factura"}

    fr_word_matches = len(words & fr_common)
    de_word_matches = len(words & de_common)
    es_word_matches = len(words & es_common)

    if fr_word_matches > 0 and fr_word_matches >= max(de_word_matches, es_word_matches):
        stats["confidence"] = min(1.0, fr_word_matches * 0.3)
        return "fr", stats
    elif de_word_matches > 0 and de_word_matches >= es_word_matches:
        stats["confidence"] = min(1.0, de_word_matches * 0.3)
        return "de", stats
    elif es_word_matches > 0:
        stats["confidence"] = min(1.0, es_word_matches * 0.3)
        return "es", stats

    # Default to English if predominantly Latin/ASCII characters
    if ascii_alpha_count > 0:
        stats["confidence"] = min(1.0, ascii_alpha_count / total_chars)
        return "en", stats

    stats["confidence"] = 0.5
    return "unknown", stats
