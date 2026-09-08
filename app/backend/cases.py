"""In-memory case store with a small disk spill for the uploaded originals.

A case is one patient's visit: the images a clinician dropped in, their predicted
views, and whatever they have corrected so far.  Decoded images stay in memory
because the segmenter is called repeatedly on the same pixels while the clinician
switches views and re-runs; the encoded originals are written to disk so a saved
case is complete on its own.

Cases are evicted oldest-first past :data:`MAX_CASES`.  This is a single-clinician
review tool, not a multi-tenant service, and the honest limitation is documented in
``docs/viewer.md`` rather than hidden behind a database.
"""

from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Cases held before the oldest is evicted.  Five 12-megapixel images decode to
#: roughly 180 MB, so this bounds the store at a few gigabytes.
MAX_CASES = 8

_PATIENT_IN_NAME = re.compile(r"(?:patient|pat|pt)[ _-]?(\d+)", re.IGNORECASE)
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize(name: str, default: str) -> str:
    """Filesystem-safe token, never empty and never a relative path."""
    cleaned = _UNSAFE.sub("_", (name or "").strip()).strip("._")
    return cleaned or default


def patient_id_from_filename(filename: str, default: str = "unknown_patient") -> str:
    """Best-effort patient id from a filename, used only when none was supplied."""
    match = _PATIENT_IN_NAME.search(Path(filename or "").stem)
    return f"patient_{match.group(1)}" if match else default


@dataclass
class CaseImage:
    image_id: str
    filename: str
    stem: str
    width: int
    height: int
    bgr: np.ndarray
    original_bytes: bytes
    content_type: str
    view: str | None = None
    view_confidence: float = 0.0
    #: True when :attr:`bgr` is a horizontal mirror of :attr:`original_bytes`.
    #: Photographs acquired *through* an intraoral mirror are flipped on the way in
    #: so every model runs in the single left/right convention the cohort was
    #: annotated in; the contours are flipped back before they leave the API, so
    #: everything the clinician sees and everything ``/save`` writes stays in the
    #: uploaded photograph's frame.  See :func:`api.create_case`.
    flipped: bool = False


@dataclass
class Case:
    case_id: str
    patient_id: str
    images: "OrderedDict[str, CaseImage]" = field(default_factory=OrderedDict)

    def image(self, image_id: str) -> CaseImage | None:
        return self.images.get(image_id)


class CaseStore:
    """Thread-safe bounded store of active cases."""

    def __init__(self, max_cases: int = MAX_CASES):
        self._cases: "OrderedDict[str, Case]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_cases

    def create(self, patient_id: str) -> Case:
        case = Case(case_id=uuid.uuid4().hex[:12], patient_id=patient_id)
        with self._lock:
            self._cases[case.case_id] = case
            while len(self._cases) > self._max:
                self._cases.popitem(last=False)
        return case

    def get(self, case_id: str) -> Case | None:
        with self._lock:
            case = self._cases.get(case_id)
            if case is not None:
                self._cases.move_to_end(case_id)
            return case
