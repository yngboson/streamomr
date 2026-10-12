"""ArUco corner markers and the metadata bit strip codec."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

ARUCO_DICT_NAME = "DICT_4X4_50"
MAX_PAGES = 12  # 4 ids per page, 50 ids in the dictionary -> 12 pages
CORNERS = ("TL", "TR", "BR", "BL")  # id offsets 0..3

META_FORM_BITS = 16
META_PAGE_BITS = 4
META_TOTAL_BITS = 4
META_CRC_BITS = 8
START_GUARD = (True, False, True, False)
STOP_GUARD = (False, True, False, True)
META_CELLS = len(START_GUARD) + META_FORM_BITS + META_PAGE_BITS + META_TOTAL_BITS + META_CRC_BITS + len(STOP_GUARD)


def aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))


def marker_id(page: int, corner: str) -> int:
    if not 0 <= page < MAX_PAGES:
        raise ValueError(f"page index {page} out of range 0..{MAX_PAGES - 1}")
    return 4 * page + CORNERS.index(corner)


def decode_marker_id(mid: int) -> tuple[int, str]:
    """marker id -> (page index, corner name)."""
    if not 0 <= mid < 4 * MAX_PAGES:
        raise ValueError(mid)
    return mid // 4, CORNERS[mid % 4]


def marker_bits(mid: int) -> np.ndarray:
    """Marker pattern incl. black border as a (k x k) bool array, True = black."""
    img = np.zeros((6, 6), np.uint8)
    img = cv2.aruco.generateImageMarker(aruco_dictionary(), mid, 6, img, 1)
    return img < 128


def marker_image(mid: int, px: int = 200, quiet_cells: int = 1) -> np.ndarray:
    """Grayscale uint8 image of the marker with a white quiet zone (in marker cells)."""
    bits = marker_bits(mid)
    k = bits.shape[0]
    cell = max(1, px // k)
    big = np.kron(~bits, np.ones((cell, cell), bool)).astype(np.uint8) * 255
    q = quiet_cells * cell
    return np.pad(big, q, constant_values=255)


# ---------------------------------------------------------------- metadata strip
@dataclass(frozen=True)
class Meta:
    form_id: int
    page: int  # 0-based
    total_pages: int


def crc8(data: bytes, poly: int = 0x07) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _payload_bytes(form_id: int, page: int, total_pages: int) -> bytes:
    return bytes([form_id >> 8, form_id & 0xFF, (page << 4) | total_pages])


def _int_bits(v: int, n: int) -> List[bool]:
    return [bool((v >> (n - 1 - i)) & 1) for i in range(n)]


def _bits_int(bits) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return v


def encode_meta(form_id: int, page: int, total_pages: int) -> List[bool]:
    if not 0 <= form_id < 1 << META_FORM_BITS:
        raise ValueError("form_id must be 16 bit")
    if not 1 <= total_pages <= MAX_PAGES:
        raise ValueError(f"total_pages must be 1..{MAX_PAGES}")
    if not 0 <= page < total_pages:
        raise ValueError("page must be in 0..total_pages-1")
    crc = crc8(_payload_bytes(form_id, page, total_pages))
    return (list(START_GUARD) + _int_bits(form_id, META_FORM_BITS) + _int_bits(page, META_PAGE_BITS)
            + _int_bits(total_pages, META_TOTAL_BITS) + _int_bits(crc, META_CRC_BITS) + list(STOP_GUARD))


def decode_meta(bits) -> Optional[Meta]:
    """Decode META_CELLS bits (True = black). None on guard/checksum/range failure."""
    bits = [bool(b) for b in bits]
    if len(bits) != META_CELLS:
        return None
    g = len(START_GUARD)
    if tuple(bits[:g]) != START_GUARD or tuple(bits[-len(STOP_GUARD):]) != STOP_GUARD:
        return None
    i = g
    form = _bits_int(bits[i:i + META_FORM_BITS]); i += META_FORM_BITS
    page = _bits_int(bits[i:i + META_PAGE_BITS]); i += META_PAGE_BITS
    total = _bits_int(bits[i:i + META_TOTAL_BITS]); i += META_TOTAL_BITS
    crc = _bits_int(bits[i:i + META_CRC_BITS])
    if crc != crc8(_payload_bytes(form, page, total)):
        return None
    if not (1 <= total <= MAX_PAGES and page < total):
        return None
    return Meta(form, page, total)


# ---------------------------------------------------------------- serial strip (per printed copy)
SERIAL_BITS = 12
SERIAL_MAX = (1 << SERIAL_BITS) - 1   # 4095; 0 = "no serial" (plain master PDF: nothing is printed)
SERIAL_CRC_BITS = 8
SERIAL_CELLS = len(START_GUARD) + SERIAL_BITS + SERIAL_CRC_BITS + len(STOP_GUARD)
_SERIAL_TAG = 0x5A   # domain separation from the meta strip's CRC


def _serial_payload(serial: int) -> bytes:
    return bytes([_SERIAL_TAG, serial >> 8, serial & 0xFF])


def encode_serial(serial: int) -> List[bool]:
    """Cells of the serial strip (True = black) for 1..SERIAL_MAX."""
    if not 1 <= serial <= SERIAL_MAX:
        raise ValueError(f"serial must be 1..{SERIAL_MAX}")
    return (list(START_GUARD) + _int_bits(serial, SERIAL_BITS)
            + _int_bits(crc8(_serial_payload(serial)), SERIAL_CRC_BITS) + list(STOP_GUARD))


def decode_serial(bits) -> Optional[int]:
    """SERIAL_CELLS bits -> serial (1..SERIAL_MAX), or None on guard/checksum failure."""
    bits = [bool(b) for b in bits]
    if len(bits) != SERIAL_CELLS:
        return None
    g = len(START_GUARD)
    if tuple(bits[:g]) != START_GUARD or tuple(bits[-len(STOP_GUARD):]) != STOP_GUARD:
        return None
    v = _bits_int(bits[g:g + SERIAL_BITS])
    crc = _bits_int(bits[g + SERIAL_BITS:g + SERIAL_BITS + SERIAL_CRC_BITS])
    if v == 0 or crc != crc8(_serial_payload(v)):
        return None
    return v


# ---------------------------------------------------------------- self-describing header (format v1)
# The header says how to read the sheet: no database is needed to recognise it.  Two rows of cells, each row
# START guard + 32 data cells + STOP guard.  The 64 data bits are a 48-bit payload + CRC-16, XOR-whitened with a
# fixed pattern (no long white / black runs, whatever the payload).  Guards differ from the legacy meta strip's,
# so neither decoder accepts the other's strip.
HEADER_VERSION = 1
HEADER_START_GUARD = (True, True, False, True)
HEADER_STOP_GUARD = (True, False, True, True)
HEADER_ROWS = 2
HEADER_ROW_DATA = 32
HEADER_ROW_CELLS = len(HEADER_START_GUARD) + HEADER_ROW_DATA + len(HEADER_STOP_GUARD)
LABEL_STYLES = ("1", "A", "a", "가")          # choice labels "1 2 3 ..", "A B C ..", "a b c ..", "가 나 다 .."
_HANGUL_LABELS = "가나다라마바사아자차"
BUBBLE_BASE_MM, BUBBLE_STEP_MM = 4.0, 0.25    # min bubble = base + step * class (class 0..7)
# (name, bits): field order of the payload
_HEADER_FIELDS = (("version", 3), ("questions", 10), ("used", 10), ("choices", 4), ("id_digits", 4),
                  ("subject_digits", 2), ("pages", 4), ("page", 4), ("labels", 2), ("bubble", 3), ("reserved", 2))
_HEADER_PAYLOAD_BITS = sum(b for _, b in _HEADER_FIELDS)
assert _HEADER_PAYLOAD_BITS + 16 == HEADER_ROWS * HEADER_ROW_DATA
_WHITEN = [bool(b) for b in np.random.default_rng(0x0A11CE).integers(0, 2, HEADER_ROWS * HEADER_ROW_DATA)]
_HEADER_TAG = b"OMR1"                          # domain separation for the CRC


@dataclass(frozen=True)
class Header:
    questions: int          # questions printed on the form (all pages)
    used: int               # questions actually used (1..questions); the rest are not printed / not read
    choices: int
    id_digits: int
    subject_digits: int     # 0..3 (marked subject code next to the student ID)
    pages: int
    page: int               # 0-based
    labels: int = 0         # index into LABEL_STYLES
    bubble: int = 0         # min bubble class: BUBBLE_BASE_MM + BUBBLE_STEP_MM * bubble
    version: int = HEADER_VERSION

    @property
    def form_key(self) -> str:
        """Identity of the printed form (everything but the page index): equal on every page of one form."""
        return (f"Q{self.questions}U{self.used}C{self.choices}D{self.id_digits}S{self.subject_digits}"
                f"P{self.pages}L{self.labels}B{self.bubble}V{self.version}")

    def choice_labels(self) -> List[str]:
        return choice_labels(self.labels, self.choices)


def choice_labels(style: int, n: int) -> List[str]:
    s = LABEL_STYLES[style]
    if s == "1":
        return [str(i + 1) for i in range(n)]
    if s == "가":
        return list(_HANGUL_LABELS[:n])
    return [chr(ord(s) + i) for i in range(n)]


def label_style(labels: List[str]) -> Optional[int]:
    """Style index of a label list, or None if it is not one of LABEL_STYLES."""
    for i in range(len(LABEL_STYLES)):
        if len(labels) <= (len(_HANGUL_LABELS) if LABEL_STYLES[i] == "가" else 26) and choice_labels(i, len(labels)) == list(labels):
            return i
    return None


def crc16(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE."""
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _header_payload(h: Header) -> List[bool]:
    vals = {"version": h.version, "questions": h.questions, "used": h.used, "choices": h.choices,
            "id_digits": h.id_digits, "subject_digits": h.subject_digits, "pages": h.pages, "page": h.page,
            "labels": h.labels, "bubble": h.bubble, "reserved": 0}
    out: List[bool] = []
    for name, n in _HEADER_FIELDS:
        v = vals[name]
        if not 0 <= v < 1 << n:
            raise ValueError(f"header field {name}={v} does not fit in {n} bits")
        out += _int_bits(v, n)
    return out


def _bits_bytes(bits) -> bytes:
    bits = list(bits) + [False] * (-len(bits) % 8)
    return bytes(_bits_int(bits[i:i + 8]) for i in range(0, len(bits), 8))


def validate_header(h: Header) -> None:
    if h.version != HEADER_VERSION:
        raise ValueError(f"unsupported header version {h.version}")
    if not 1 <= h.used <= h.questions:
        raise ValueError("used must be 1..questions")
    if not 2 <= h.choices <= 10 or not 1 <= h.id_digits <= 12 or not 0 <= h.subject_digits <= 3:
        raise ValueError("choices / id_digits / subject_digits out of range")
    if not 1 <= h.pages <= MAX_PAGES or not 0 <= h.page < h.pages:
        raise ValueError("pages / page out of range")
    if h.labels >= len(LABEL_STYLES) or not 0 <= h.bubble < 8:
        raise ValueError("labels / bubble out of range")


def encode_header(h: Header) -> List[List[bool]]:
    """Cells (True = black) of the header rows, guards included."""
    validate_header(h)
    pay = _header_payload(h)
    data = pay + _int_bits(crc16(_HEADER_TAG + _bits_bytes(pay)), 16)
    data = [b ^ w for b, w in zip(data, _WHITEN)]
    rows = []
    for r in range(HEADER_ROWS):
        rows.append(list(HEADER_START_GUARD) + data[r * HEADER_ROW_DATA:(r + 1) * HEADER_ROW_DATA] + list(HEADER_STOP_GUARD))
    return rows


def decode_header_data(data) -> Optional[Header]:
    """64 data bits (guards stripped, still whitened) -> Header, or None on CRC / range failure."""
    data = [bool(b) ^ w for b, w in zip(data, _WHITEN)]
    if len(data) != HEADER_ROWS * HEADER_ROW_DATA:
        return None
    pay, crc = data[:_HEADER_PAYLOAD_BITS], _bits_int(data[_HEADER_PAYLOAD_BITS:])
    if crc != crc16(_HEADER_TAG + _bits_bytes(pay)):
        return None
    vals, i = {}, 0
    for name, n in _HEADER_FIELDS:
        vals[name] = _bits_int(pay[i:i + n])
        i += n
    if vals.pop("reserved") != 0:
        return None
    h = Header(**vals)
    try:
        validate_header(h)
    except ValueError:
        return None
    return h


def decode_header(rows) -> Optional[Header]:
    """Rows of cells (guards included) -> Header, or None."""
    g0, g1 = len(HEADER_START_GUARD), len(HEADER_STOP_GUARD)
    data = []
    for row in rows:
        row = [bool(b) for b in row]
        if len(row) != HEADER_ROW_CELLS or tuple(row[:g0]) != HEADER_START_GUARD or tuple(row[-g1:]) != HEADER_STOP_GUARD:
            return None
        data += row[g0:-g1]
    return decode_header_data(data)
