"""Extract ham-radio entities (callsigns, Q-codes, frequencies) from transcript
segments. Callsigns are recovered two ways: literal tokens Whisper wrote out
(e.g. "W1AW") and reconstructed from spoken phonetics ("whiskey one alpha whiskey").

An entity is {"type", "value", "start", "seg"} where start is the segment start
time (seconds) and seg is the segment index.
"""
import re

PHONETIC = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D",
    "echo": "E", "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I",
    "juliet": "J", "juliett": "J", "kilo": "K", "lima": "L", "mike": "M",
    "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R",
    "sierra": "S", "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W",
    "whisky": "W", "xray": "X", "yankee": "Y", "zulu": "Z",
}
NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9",
}

CALLSIGN_RE = re.compile(r"\b([A-Za-z]{1,2}\d[A-Za-z]{1,4})\b")
QCODE_RE = re.compile(r"\bQ[A-Za-z]{2}\b")
FREQ_RE = re.compile(
    r"\b(\d{1,4}(?:[.,]\d{1,4})?)\s?(khz|mhz|ghz|hz|megahertz|kilohertz)\b", re.I)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_UNIT = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz",
         "kilohertz": "kHz", "megahertz": "MHz"}


def _is_callsignish(s: str) -> bool:
    return 4 <= len(s) <= 7 and any(c.isdigit() for c in s) and any(c.isalpha() for c in s)


def _phonetic_callsigns(tokens: list[str]) -> list[str]:
    """Find maximal runs of phonetic/number words and keep the callsign-like ones."""
    out, run = [], []
    for tok in tokens:
        ch = PHONETIC.get(tok.lower()) or NUM_WORDS.get(tok.lower())
        if ch:
            run.append(ch)
        else:
            if run:
                out.append("".join(run))
            run = []
    if run:
        out.append("".join(run))
    return [r for r in out if _is_callsignish(r)]


def extract(segments: list[dict]) -> list[dict]:
    found, seen = [], set()

    def add(typ, value, start, seg):
        key = (typ, value, seg)
        if key not in seen:
            seen.add(key)
            found.append({"type": typ, "value": value, "start": start, "seg": seg})

    for i, seg in enumerate(segments):
        text = seg.get("text", "") or ""
        start = seg.get("start", 0) or 0
        tokens = _TOKEN_RE.findall(text)
        for cs in _phonetic_callsigns(tokens):
            add("callsign", cs, start, i)
        for m in CALLSIGN_RE.finditer(text):
            add("callsign", m.group(1).upper(), start, i)
        for m in QCODE_RE.finditer(text):
            add("qcode", m.group(0).upper(), start, i)
        for m in FREQ_RE.finditer(text):
            num = m.group(1).replace(",", ".")
            unit = _UNIT.get(m.group(2).lower(), m.group(2))
            add("frequency", f"{num} {unit}", start, i)
    return found


def match_terms(text: str, entities: list[dict], terms: list[str]) -> list[str]:
    """Return which watch terms appear in the transcript text or entity values."""
    if not terms:
        return []
    low = (text or "").lower()
    evals = {e["value"].lower() for e in entities}
    hits = []
    for term in terms:
        t = term.strip().lower()
        if not t:
            continue
        if t in low or t in evals:
            hits.append(term.strip())
    return hits
