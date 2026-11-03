"""
Vision and LLM client using ai.dury.dev API for score metadata extraction and lyrics verification.
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pymupdf
import requests
from lxml import etree


DEFAULT_API_BASE = "https://ai.dury.dev/v1"
DEFAULT_MODEL = os.environ.get("DIGISCORE_AI_MODEL", "opllm/combos/vision-top")
# Tried in order when the current model is out of quota or unavailable
FALLBACK_MODELS = ["antigravity/gemini-3-flash"]
MUSICXML_DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)


def get_api_key() -> Optional[str]:
    """Retrieves the API key from environment or .env file."""
    key = os.environ.get("DURY_AI_API_KEY")
    if key:
        return key

    # Look in current working directory .env or parent
    env_paths = [Path.cwd() / ".env", Path(__file__).resolve().parent.parent.parent / ".env"]
    for env_path in env_paths:
        if env_path.is_file():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DURY_AI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class VisionClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        timeout: int = 25,
    ):
        self.api_key = api_key or get_api_key()
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _chat_completion(self, messages: list, max_tokens: int = 800) -> str:
        if not self.api_key:
            raise RuntimeError("DURY_AI_API_KEY is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }

        url = f"{self.api_base}/chat/completions"
        # The endpoint rate-limits and is sometimes unavailable: retry with backoff
        import time

        exhausted = getattr(self, "_exhausted", set())
        self._exhausted = exhausted
        models = [m for m in [self.model] + FALLBACK_MODELS if m not in exhausted]
        if not models:
            raise RuntimeError("AI quota exhausted for this run")
        resp = None
        for model in dict.fromkeys(models):
            payload["model"] = model
            delay = 4.0
            for attempt in range(4):
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in (404, 429) and (resp.status_code == 404 or "quota" in resp.text.lower()):
                    exhausted.add(model)
                    break
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(delay)
                    delay *= 2.5
                    continue
                break
            if resp.ok:
                break
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extracts score metadata (title, subtitle, composer, lyricist, arranger, tempo, parts)
        from the first page of the score using vision.
        """
        if not self.is_available:
            return {}

        try:
            doc = pymupdf.open(pdf_path)
            if len(doc) == 0:
                return {}
            page = doc[0]
            # Render first page at 120 DPI
            pix = page.get_pixmap(dpi=120)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            doc.close()

            prompt = (
                "You are an expert musicologist and score editor. Examine this first page of a music score.\n"
                "Extract the following metadata in clean JSON format:\n"
                "{\n"
                '  "title": "Main title of the work",\n'
                '  "subtitle": "Subtitle or movement title if any",\n'
                '  "composer": "Composer name",\n'
                '  "lyricist": "Lyricist / author of text",\n'
                '  "arranger": "Arranger if mentioned",\n'
                '  "tempo_marking": "Tempo text or metronome indication (e.g. Allegro, Moderato, quarter=80)",\n'
                '  "parts": ["List", "of", "vocal/instrumental", "parts", "e.g.", "Soprano", "Alto", "Tenor", "Bass"]\n'
                "}\n"
                "Return ONLY valid JSON. If a field is not found or ambiguous, leave it as an empty string or empty list."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ]

            raw_response = self._chat_completion(messages, max_tokens=600)
            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {}
        except Exception as e:
            # Fallback gracefully if AI is unavailable or fails
            return {"error": str(e)}

    def read_lyric_line(self, image_png: bytes, syllables: List[str]) -> Optional[List[str]]:
        """
        Read one line of sung lyrics from its image, helped by the OCR's guess.

        Returns the syllables as printed, in order (one per sung note: words
        hyphenated as printed are separate items), or None when unavailable.
        The caller aligns them with the notes, so the count may differ from the
        OCR's (merged or missed syllables are the OCR's usual mistakes).
        """
        if not self.is_available or not syllables:
            return None
        prompt = (
            "This image is one line of lyrics printed under a music staff. Each syllable "
            "sung on its own note is printed separately; syllables of one word are joined "
            "by hyphens. An OCR read the line as:\n"
            f"{json.dumps(syllables, ensure_ascii=False)}\n\n"
            "The OCR often merges two neighbouring syllables into one item, misses some, "
            "and misreads letters and accents. Using the image, return every syllable of "
            "the line in order, one item per syllable as printed, with its accents and "
            "attached punctuation, without hyphens. Leave out verse numbers.\n"
            "Return ONLY a JSON array of strings."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_png).decode()}},
            ],
        }]
        items = None
        for _ in range(3):
            try:
                raw = self._chat_completion(messages, max_tokens=80 + 14 * len(syllables))
            except Exception:
                continue
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                continue
            try:
                items = json.loads(match.group(0))
            except Exception:
                continue
            if isinstance(items, list):
                break
            items = None
        if not items:
            return None
        out = [str(x).strip().strip("-").strip() for x in items]
        out = [x for x in out if x]
        # Where the reading only cuts the OCR's letters differently, without
        # an OCR syllable that ran two together, keep the OCR's syllables
        out = _keep_ocr_cuts(syllables, out)
        # A reading far from the OCR's is a misread line (or the model's own text)
        joined_a = "".join(syllables).lower()
        joined_b = "".join(out).lower()
        if not out or not _close_text(joined_a, joined_b):
            return None
        return out

    def correct_lyric_line(self, image_png: bytes, syllables: List[str]) -> Optional[List[str]]:
        """
        Correct OCR mistakes in one line of sung lyrics, one string per given
        syllable (kept for callers that need the same count).
        """
        items = self.read_lyric_line(image_png, syllables)
        if not items or len(items) != len(syllables):
            return None
        return [new if _close(old, new) else old for old, new in zip(syllables, items)]

    def verify_and_clean_lyrics(self, lyrics_text: str, language: str = "French") -> str:
        """
        Proofreads and cleans lyric text or verses, fixing OCR spelling or hyphenation errors.
        """
        if not self.is_available or not lyrics_text.strip():
            return lyrics_text

        try:
            prompt = (
                f"You are an expert vocal score proofreader. The following text contains singing lyrics "
                f"extracted via OCR or text analysis ({language}).\n"
                f"Please fix any obvious OCR typos or accent errors while preserving the poetic line structure "
                f"and syllabification.\n\n"
                f"Text:\n{lyrics_text}\n\n"
                f"Return ONLY the corrected text without any commentary."
            )
            messages = [{"role": "user", "content": prompt}]
            return self._chat_completion(messages, max_tokens=1000).strip()
        except Exception:
            return lyrics_text


def _keep_ocr_cuts(ocr: List[str], read: List[str]) -> List[str]:
    """
    The reading may split an OCR syllable (two syllables merged by the OCR)
    and fix letters, but not move a boundary the OCR saw: where the reading
    joins letters across an OCR boundary ("luctus in" read as "luc tu sin"),
    that stretch keeps the OCR's syllables.
    """
    import difflib

    a = "".join(ocr).lower()
    b = "".join(read).lower()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    a_of_b = {}
    for blk in sm.get_matching_blocks():
        for t in range(blk.size):
            a_of_b[blk.b + t] = blk.a + t
    bounds_a = set()
    k = 0
    for x in ocr:
        k += len(x)
        bounds_a.add(k)
    # Reading items with their letter span in the OCR text (None when unmatched)
    spans = []
    k = 0
    for x in read:
        idx = [a_of_b.get(k + t) for t in range(len(x))]
        k += len(x)
        got = [i for i in idx if i is not None]
        spans.append((min(got), max(got) + 1) if got else None)
    ocr_spans = []
    k = 0
    for x in ocr:
        ocr_spans.append((k, k + len(x)))
        k += len(x)
    # Group reading items and OCR syllables that share letters; a group of
    # one OCR syllable takes the reading (merged syllables split, letters
    # fixed); a group of several keeps the OCR's cut (the reading moved a
    # boundary the OCR saw, or joined pieces the note alignment will handle)
    out: List[str] = []
    i = 0
    n = len(read)
    while i < n:
        if spans[i] is None:
            out.append(read[i])
            i += 1
            continue
        lo, hi = spans[i]
        j = i + 1
        group_ocr = {q for q, (s0, s1) in enumerate(ocr_spans) if s0 < hi and lo < s1}
        while True:
            if group_ocr:
                lo = min(lo, min(ocr_spans[q][0] for q in group_ocr))
                hi = max(hi, max(ocr_spans[q][1] for q in group_ocr))
            if j < n and spans[j] is not None and spans[j][0] < hi:
                hi = max(hi, spans[j][1])
                group_ocr |= {q for q, (s0, s1) in enumerate(ocr_spans) if s0 < hi and lo < s1}
                j += 1
                continue
            break
        if len(group_ocr) <= 1:
            out.extend(read[i:j])
        else:
            out.extend(ocr[q] for q in sorted(group_ocr))
        i = j
    return out


def _close_text(a: str, b: str) -> bool:
    """Two readings of a whole line agree on most of their letters."""
    import difflib
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.6


def _close(a: str, b: str) -> bool:
    """Small edit: accents, a letter or two, punctuation."""
    import difflib
    import unicodedata

    def base(s):
        s = unicodedata.normalize("NFKD", s)
        return "".join(c for c in s if not unicodedata.combining(c)).lower()

    if base(a) == base(b):
        return True
    ratio = difflib.SequenceMatcher(None, base(a), base(b)).ratio()
    return ratio >= 0.6 and abs(len(a) - len(b)) <= 2


def _new_creator(ident, kind: str):
    """Creators must come first in <identification>, before rights/encoding/source."""
    elem = etree.Element("creator", type=kind)
    existing = ident.findall("creator")
    if existing:
        existing[-1].addnext(elem)
    else:
        ident.insert(0, elem)
    return elem


def enrich_musicxml_with_metadata(xml_content: str, metadata: Dict[str, Any]) -> str:
    """
    Injects or updates metadata into an existing MusicXML string.
    """
    if not metadata:
        return xml_content

    try:
        parser = etree.XMLParser(remove_blank_text=False)
        root = etree.fromstring(xml_content.encode("utf-8"), parser=parser)

        title = metadata.get("title")
        if title:
            # Update or create <work><work-title>
            work = root.find("work")
            if work is None:
                work = etree.Element("work")
                root.insert(0, work)
            wt = work.find("work-title")
            if wt is None:
                wt = etree.SubElement(work, "work-title")
            wt.text = title

            # Also update <movement-title> if present or needed
            mt = root.find("movement-title")
            if mt is not None and not mt.text:
                mt.text = title

        # Identification & Creators
        ident = root.find("identification")
        if ident is None:
            # Goes after <work>, <movement-number> and <movement-title>
            ident = etree.Element("identification")
            anchor = None
            for tag in ("work", "movement-number", "movement-title"):
                if root.find(tag) is not None:
                    anchor = root.find(tag)
            if anchor is not None:
                anchor.addnext(ident)
            else:
                root.insert(0, ident)

        composer = metadata.get("composer")
        if composer:
            comp_elem = ident.find('creator[@type="composer"]')
            if comp_elem is None:
                comp_elem = _new_creator(ident, "composer")
            comp_elem.text = composer

        lyricist = metadata.get("lyricist")
        if lyricist:
            lyr_elem = ident.find('creator[@type="lyricist"]')
            if lyr_elem is None:
                lyr_elem = _new_creator(ident, "lyricist")
            lyr_elem.text = lyricist

        arranger = metadata.get("arranger")
        if arranger:
            arr_elem = ident.find('creator[@type="arranger"]')
            if arr_elem is None:
                arr_elem = _new_creator(ident, "arranger")
            arr_elem.text = arranger

        # Parts naming if parts are generic (e.g. Part 1, Voice, or dummyPart)
        meta_parts = metadata.get("parts", [])
        if meta_parts:
            part_list = root.find("part-list")
            if part_list is not None:
                score_parts = part_list.findall("score-part")
                # Names only map one-to-one when the counts agree (two voices may share a staff)
                if len(meta_parts) != len(score_parts):
                    score_parts = []
                for idx, sp in enumerate(score_parts):
                    if idx < len(meta_parts):
                        pn = sp.find("part-name")
                        curr_name = (pn.text or "").strip()
                        if not curr_name or "part" in curr_name.lower() or "voice" in curr_name.lower() or "dummy" in curr_name.lower():
                            if pn is None:
                                pn = etree.SubElement(sp, "part-name")
                            pn.text = meta_parts[idx]

        return etree.tostring(
            root.getroottree(),
            pretty_print=True,
            xml_declaration=True,
            encoding="UTF-8",
            doctype=MUSICXML_DOCTYPE,
        ).decode("utf-8")
    except Exception:
        return xml_content
