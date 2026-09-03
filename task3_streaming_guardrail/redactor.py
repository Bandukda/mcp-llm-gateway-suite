r"""
Incremental PII redaction over a token stream.

The problem
-----------
Redacting PII from a finished string is a one-liner. Redacting it from a
*stream* is not, because the model does not emit PII in convenient units. A
single SSN arrives as four separate deltas:

    chunk 1: "Her SSN is 123"
    chunk 2: "-45"
    chunk 3: "-67"
    chunk 4: "89, please file it."

Forward chunk 1 as it arrives and "123" is already on the client's screen when
chunk 2 reveals what it was the start of. You cannot un-send a token.

Buffering the whole response and redacting at the end solves correctness and
destroys the product: time-to-first-token becomes time-to-*last*-token, and
memory grows with response length. The task explicitly rules this out.

The approach: emit everything that is provably safe
---------------------------------------------------
Keep a small buffer. On each chunk:

1. Append the chunk to the buffer.
2. Compute a **hold-back**: the length of the buffer's trailing run that could
   still grow into a match if more text arrives.
3. Redact complete matches in everything *before* the hold-back, and emit it.
4. Keep the rest.

Steps 2 and 3 are in that order on purpose. Redacting the whole buffer first
looks equivalent and is not: "ada@example.co" is already a complete match of the
email pattern, so it would be replaced, and the "m" arriving in the next chunk
would land after the placeholder. See ``feed()``.

On stream end, ``flush()`` redacts and emits whatever is left, because nothing
more can arrive to extend it.

The only interesting part is step 2, the hold-back. Two rules cover every
pattern here:

``TAIL_TOKEN``   a trailing run of characters that can appear inside an email,
                 an SSN, a card number or an API key, with no space. Prose is
                 unaffected because a space ends the run -- ``"Hello there"``
                 holds back only ``"there"``, five characters.

``TAIL_DIGITS``  a trailing run that starts a number -- optionally "+" or "(",
                 then a digit, then digits, spaces, dashes or parens. This
                 catches ``4111 1111 1111 1111`` and ``(555) 123-4567``, where
                 the spaces would otherwise end the run early.

Those two rules are a heuristic about text that might still *become* a match.
``pull_back_behind_straddling_match`` is the exact guard for text that already
*is* one, and it is what makes the no-straddle property structural rather than a
claim about character classes.

The hold-back is the longer of the two, capped at ``MAX_HOLDBACK`` (400). The cap
bounds both memory and added latency: **the buffer never exceeds ``MAX_BUFFER``
(800), whatever the response length**, and no character is delayed by more than
the time it takes for ``MAX_HOLDBACK`` more to arrive.

The cap is also the one place the guarantee is conditional: a match *longer* than
``MAX_HOLDBACK`` could straddle the boundary and have its head emitted in the
clear. That is why every pattern here is length-bounded -- the longest possible
match is a 64-octet local part, "@", a 255-octet domain, "." and a 24-character
TLD, which is 345 -- and why ``MAX_HOLDBACK`` is 400. Adding a pattern means
revisiting both numbers. ``assert_holdback_covers_patterns()`` fails loudly
if a new pattern introduces a character the tail rules do not know about.

Cost and correctness
--------------------
* Memory: O(1) in *response* length -- the retained state is bounded by
  ``MAX_BUFFER`` plus ``LOOKBEHIND_CONTEXT``. It is not bounded in *chunk* size:
  a chunk is appended before it can be examined, so a caller that hands the
  redactor one 1 MB delta has 1 MB resident for that call. That is inherent to
  receiving the chunk at all, and the buffer returns to the bound immediately
  after. Real SSE deltas are a few characters.
* Per chunk: one regex pass over ``len(chunk) + holdback`` characters.
* Added TTFT: zero for a first chunk containing a space; otherwise bounded by
  the cap.

Choosing the heuristic over exact partial matching is deliberate. Python's
stdlib ``re`` cannot answer "could this be the *prefix* of a match" -- the
third-party ``regex`` module can, via ``fullmatch(..., partial=True)``, and the
exact version is implemented in ``holdback_exact()`` below for comparison. The
heuristic is the default because it needs no extra dependency, is easy to reason
about in review, and over-holds rather than under-holds: it can never emit a
character that a longer match would have covered, which is the direction an
error has to fall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

# The cap must be at least as long as the longest match any pattern can produce,
# or a very long email could be split across the safe/held boundary and its first
# half emitted in the clear. 128 comfortably exceeds a 19-digit card written with
# separators (23 chars) and any realistic address.
# 64-octet local part + "@" + 255-octet domain + "." + 24-char TLD = 345 is the
# longest match any pattern can now produce. The hold-back must exceed it, or an
# in-progress email is cut and its head emitted in the clear.
MAX_HOLDBACK = 400

# Retained tail of already-emitted text, kept so that every scan sees the same
# left-hand context the whole-string pass would.
#
# One character would satisfy the lookbehind fences. It has to be MAX_MATCH,
# because the straddle guard needs to find matches that *start* before the
# buffer, not merely evaluate a lookbehind. With only 8 characters retained, a
# buffer beginning mid-token offered a shorter, differently-anchored match than
# the full text did -- and in one fuzz case the guard therefore let the split
# advance four characters into a real email address. Retaining MAX_MATCH means
# every match that could reach the emit window is visible in full.
LOOKBEHIND_CONTEXT = 400

# The buffer can exceed MAX_HOLDBACK. pull_back_behind_straddling_match moves the
# split to the *start* of a match that spans it, and that start can sit further
# back than the hold-back window -- five-character chunks of card numbers reach
# 145. The true bound is MAX_HOLDBACK plus the length of one match, which is
# still O(1) in response length, which is the property that matters. Holding a
# whole match is also the only safe option: the alternative is emitting its first
# half. MAX_MATCH is a generous ceiling for a single pattern match.
MAX_MATCH = 400
MAX_BUFFER = MAX_HOLDBACK + MAX_MATCH

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------
# Ordered: the first pattern to match a span wins. Longest/most specific first.
PATTERNS: list[tuple[str, str]] = [
    # Email. Deliberately not RFC 5322 -- that grammar matches things nobody
    # sends, and a redactor should be greedy, not pedantic.
    # Length-bounded on purpose. With unbounded "+" quantifiers a single match
    # can be as long as the response, which silently makes the buffer bound
    # below a lie -- "a"*40 + "@" + "bb."*1000 + "com" buffered 3,044 characters
    # against a claimed 448. The bounds are RFC 5321's: 64-octet local part,
    # 255-octet domain. Every pattern here is now shorter than MAX_HOLDBACK,
    # which is what makes MAX_BUFFER true.
    ("EMAIL", r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,24}"),
    # Provider API keys. The most common thing a model leaks back after being
    # shown a config file.
    ("API_KEY", r"\b(?:sk|pk|rk)-(?:live|test|proj|ant)?-?[A-Za-z0-9]{16,64}\b"),
    # Payment card: 13-19 digits in groups separated by spaces or dashes.
    # Luhn-checked after matching; see redact_complete().
    #
    # Ordered BEFORE the SSN rule and fenced with digit-run lookarounds. Without
    # both, "4111111111111111" is matched by the 9-digit SSN rule first, leaving
    # "[REDACTED]1111111" -- a redaction that publishes seven digits of a card
    # number. Alternation order is a correctness property here, not a style
    # choice, and test_complete_patterns pins it.
    ("CREDIT_CARD", r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])"),
    # US SSN, with or without separators. The negative lookaheads exclude the
    # never-issued ranges so that ordinary 9-digit numbers survive.
    (
        "SSN",
        r"(?<![\d\-])(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}(?![\d\-])",
    ),
    # E.164 and common US formats.
    ("PHONE", r"(?<![\d\-])(?:\+1[ \-]?)?\(?\d{3}\)?[ \-]\d{3}[ \-]\d{4}(?![\d\-])"),
]

COMBINED = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in PATTERNS))

# A trailing run with no spaces: emails, SSNs with dashes, API keys.
TAIL_TOKEN = re.compile(r"[A-Za-z0-9@._%+\-]+\Z")
# A trailing run that begins a number: spaced cards and phone numbers.
#
# The bracket and paren characters are here because PHONE contains them. Leaving
# them out is not a cosmetic gap: "Call me at (555) 123-456" + "7 tomorrow"
# leaked the whole number in the clear, because ")" truncated the held tail and
# neither half matched on its own. The invariant below is only true if this
# class is a superset of every character every pattern can produce.
#
# The "[+(]" branch stands alone so that a bare trailing "(" is held: at one
# character per chunk, "Call (" would otherwise emit the paren before the digits
# that make it a phone number ever arrive, leaving "Call ([REDACTED]".
TAIL_DIGITS = re.compile(r"(?:[+(][\d ()\-]*|\d[\d ()\-]*)\Z")

# Guard the invariant rather than trusting a comment: every literal character
# class in PATTERNS must be covered by the tail rules above.
_TAIL_ALPHABET = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@._%+-() ")


@lru_cache(maxsize=8192)
def luhn_ok(digits: str) -> bool:
    """Luhn check. Without it, any 16-digit order reference is redacted as a card."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


@dataclass
class RedactionStats:
    chunks_in: int = 0
    characters_in: int = 0
    characters_out: int = 0
    redactions: dict[str, int] = field(default_factory=dict)
    max_buffer: int = 0

    def record(self, label: str) -> None:
        self.redactions[label] = self.redactions.get(label, 0) + 1

    @property
    def total_redactions(self) -> int:
        return sum(self.redactions.values())


@lru_cache(maxsize=8192)
def _is_card(value: str) -> bool:
    """Luhn-validate a candidate card span.

    Cached because the buffer slides: as chunks arrive, the same spans are
    re-examined over and over. Without it, "1 " repeated (every digit is a valid
    card start, since each is preceded by a space) drove 930,000 Luhn checks
    through a 10 KB response -- the profile's entire hot path.
    """
    digits = "".join(char for char in value if char.isdigit())
    return 13 <= len(digits) <= 19 and luhn_ok(digits)


def iter_matches(text: str, start: int = 0):
    """Yield real PII matches from ``start``, skipping non-card digit runs.

    The subtlety this exists for. ``CREDIT_CARD`` matches any 13-19 digit run,
    and Luhn is what decides whether it is really a card. If a rejected span is
    *consumed* -- which is what ``re.sub`` with a returning replacement function
    does -- the scan resumes at its end and never looks inside it. Any real PII
    that the run swallowed is then emitted in the clear:

        "Invoice 12345 4111 1111 1111 1111 was charged."

    The card is preceded by a 5-digit invoice number, so the greedy alternative
    matches "5 4111 1111 1111 1111", fails Luhn, and is returned verbatim --
    publishing the whole card. Roughly 70% of cards behind a short digit prefix
    leaked this way. It is the mirror image of the SSN-before-CREDIT_CARD
    ordering bug: there, one pattern ate another; here, a *rejected* match ate a
    real one.

    So a Luhn rejection advances by a single character instead of consuming the
    span, letting a genuine card that starts further in still be found.
    """
    pos = start
    while True:
        match = COMBINED.search(text, pos)
        if match is None:
            return
        if match.lastgroup == "CREDIT_CARD" and not _is_card(match.group(0)):
            pos = match.start() + 1
            continue
        yield match
        pos = match.end()


DEFAULT_PLACEHOLDER = "[REDACTED]"


def _record(
    match: "re.Match[str]",
    stats: RedactionStats | None,
    placeholder: str = DEFAULT_PLACEHOLDER,
) -> str:
    if stats is not None:
        stats.record(match.lastgroup or "PII")
    return placeholder


def redact_complete(
    text: str, stats: RedactionStats | None = None, placeholder: str = DEFAULT_PLACEHOLDER
) -> str:
    """Replace every complete match in ``text``. Used on the buffer, not the chunk."""
    return redact_range(text, 0, len(text), stats, placeholder)


def redact_complete_from(
    text: str,
    start: int,
    stats: RedactionStats | None = None,
    placeholder: str = DEFAULT_PLACEHOLDER,
) -> str:
    """Redact from ``start`` to the end, letting lookbehinds see the text before it."""
    return redact_range(text, start, len(text), stats, placeholder)


def redact_range(
    text: str,
    start: int,
    end: int,
    stats: RedactionStats | None = None,
    placeholder: str = DEFAULT_PLACEHOLDER,
    matches: "list[re.Match[str]] | None" = None,
) -> str:
    r"""Emit ``text[start:end]`` redacted, while *scanning* the whole of ``text``.

    Both boundaries matter, and each one bit once.

    The left edge: several patterns are fenced with ``(?<![\d\-])``, and a
    lookbehind at position 0 of a truncated string sees "start of input", which
    satisfies a negative lookbehind even when the real preceding character was a
    digit. Scanning from ``start`` inside the full string -- ``re``'s
    ``search(string, pos)`` keeps lookbehinds looking behind ``pos`` -- fixes it.

    The right edge is the same bug reflected, and nastier. End-of-string also
    satisfies the trailing ``(?![\d\-])`` fence, so a buffer truncated mid-digit-run
    offers a *longer* spurious card match that the full text would never produce:

        "Card 9039 3080 7022 682 " + "8"*140

    In the full text the 15-digit card is matched and redacted. In a prefix cut
    inside the run of eights, the greedy match becomes 16 digits, fails Luhn, and
    the real card is emitted in the clear. Scanning the complete buffer and
    emitting only ``[start, end)`` means the match found is always the one the
    whole-string pass would find.
    """
    pieces: list[str] = []
    cursor = start
    # `matches` lets feed() scan once and hand the same list to the straddle
    # guard and to this emitter, instead of running two identical passes.
    for match in matches if matches is not None else iter_matches(text, start):
        if match.end() > end:
            # Only matches lying wholly inside the emit window are resolved here.
            # Testing match.start() instead was wrong in a way that took a fuzz
            # run to surface: a match beginning before `end` and running past it
            # got redacted, cursor jumped beyond `end`, and the trailing
            # text[cursor:end] slice went empty -- emitting a redaction the
            # whole-string pass never makes and swallowing the text after it.
            # A match that overruns the window is still growing; it belongs to
            # the held tail, and the next feed() or flush() will resolve it.
            break
        pieces.append(text[cursor : match.start()])
        pieces.append(_record(match, stats, placeholder))
        cursor = match.end()
    pieces.append(text[cursor:end])
    return "".join(pieces)


def pull_back_behind_straddling_match(
    buffer: str, split: int, offset: int = 0, matches: "list[re.Match[str]] | None" = None
) -> int:
    """Move ``split`` back so no complete match spans it.

    The tail rules are a heuristic for text that might *become* a match. This is
    the exact guard for text that already *is* one: if a complete match starts
    before the split and ends after it, the split lands inside a real value and
    the first half would be emitted in the clear.

    That is not hypothetical. With the tail rules alone,
    ``"Call (555" + ") 123-4567."`` emitted the phone number untouched: the
    buffer ends in "." so the numeric tail rule cannot reach the end, and the
    word rule only holds back "4567." -- ten characters *after* the match began.

    Cheap (one pass already being done on this buffer) and it makes the
    no-straddle property structural instead of a claim about character classes.
    """
    # Scanned from `offset`, exactly like the emitter in redact_range().
    #
    # These two used to disagree, and that was a PII leak. The guard scanned from
    # 0 while the emitter scanned from `offset`, and because `whole` is truncated
    # at LOOKBEHIND_CONTEXT the guard's scan could begin mid-value and segment an
    # ambiguous chain of adjacent matches out of phase with the emitter's. On
    # "123-45-6789 " repeated past ~820 characters, the guard saw a clean
    # boundary at the split and did not pull back; the emitter then found a match
    # spanning it, hit its `break`, and emitted that match's head in the clear --
    # after which the buffer was trimmed past it, so it was never redacted.
    #
    # Two scans of the same text from the same origin cannot disagree. Lookbehind
    # still sees the retained context, because re.finditer(string, pos) looks
    # behind pos.
    for match in matches if matches is not None else iter_matches(buffer, offset):
        if match.start() < split < match.end():
            split = match.start()
    # A pull-back into the retained context would mean a character inside a match
    # had already been emitted, which the hold-back makes impossible: the split
    # never advances closer than MAX_MATCH to the end of the buffer, so any match
    # reaching it is already complete and visible here. Clamped rather than
    # asserted so a future pattern change degrades into over-holding.
    return max(split, offset)


def assert_holdback_covers_patterns() -> None:
    """Fail loudly if a pattern can contain a character the tail rules cannot hold.

    This is the invariant the whole design rests on. It was broken once already:
    PHONE contains "(" and ")", the tail classes did not, and a parenthesised
    number split across a chunk boundary was emitted in the clear.
    """
    literals = set()
    for _, pattern in PATTERNS:
        # Characters a pattern can emit, minus regex metacharacters. Crude on
        # purpose: over-reporting here costs a wider hold-back class, which is
        # the safe direction.
        for char in re.sub(r"\\[dwsbAZ]|\{\d+(,\d+)?\}|\(\?[:!=<][^)]*\)", "", pattern):
            if char.isalnum() or char in "@._%+- ()":
                literals.add(char)
    missing = literals - _TAIL_ALPHABET
    if missing:
        raise AssertionError(
            f"PATTERNS can produce {sorted(missing)!r}, which the hold-back "
            "character classes do not cover; a match containing one can be split "
            "across a chunk boundary and leak. Widen TAIL_TOKEN / TAIL_DIGITS."
        )


def holdback(buffer: str, max_holdback: int = MAX_HOLDBACK) -> int:
    """How many trailing characters must be withheld as a possible partial match."""
    if not buffer:
        return 0
    window = buffer[-max_holdback:]
    token = TAIL_TOKEN.search(window)
    digits = TAIL_DIGITS.search(window)
    longest = max(
        len(token.group(0)) if token else 0,
        len(digits.group(0)) if digits else 0,
    )
    return min(longest, max_holdback)


def holdback_exact(buffer: str, max_holdback: int = MAX_HOLDBACK) -> int:
    """Exact hold-back using the third-party ``regex`` module's partial matching.

    Included as the reference implementation the heuristic approximates. It asks
    the real question -- "is this suffix a prefix of some pattern?" -- instead of
    approximating it with a character-class rule, so it holds back the minimum
    possible. It needs ``pip install regex``; the heuristic is the default so the
    guardrail has no dependency beyond the standard library.
    """
    try:
        import regex  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("holdback_exact() requires the 'regex' package") from exc

    compiled = regex.compile("|".join(pattern for _, pattern in PATTERNS))
    start = max(0, len(buffer) - max_holdback)
    for index in range(start, len(buffer)):
        if compiled.fullmatch(buffer[index:], partial=True):
            return len(buffer) - index
    return 0


class StreamRedactor:
    """Feed chunks in, get safe-to-emit text out.

    Usage::

        redactor = StreamRedactor()
        for chunk in upstream:
            safe = redactor.feed(chunk)
            if safe:
                yield safe
        yield redactor.flush()
    """

    def __init__(
        self, max_holdback: int = MAX_HOLDBACK, placeholder: str = DEFAULT_PLACEHOLDER
    ) -> None:
        self.max_holdback = max_holdback
        self.placeholder = placeholder
        self._buffer = ""
        self._left_context = ""
        self._closed = False
        self.stats = RedactionStats()

    @property
    def buffered(self) -> int:
        """Characters currently held back. At rest, never exceeds ``MAX_BUFFER``.

        Not ``max_holdback``: the straddle guard can pull the split back to the
        start of a match that began before the hold-back window. See MAX_BUFFER.
        """
        return len(self._buffer)

    def feed(self, chunk: str) -> str:
        if self._closed:
            raise RuntimeError("feed() after flush()")
        if not chunk:
            return ""

        self.stats.chunks_in += 1
        self.stats.characters_in += len(chunk)

        self._buffer += chunk

        # Order matters. Redacting the whole buffer *before* splitting looks
        # equivalent and is not: "ada@example.co" is already a complete match of
        # the email pattern, so it would be replaced, and the "m" arriving next
        # would leave "[REDACTED]m" on the client's screen. Holding back first
        # means a pattern that is still growing is never judged early.
        pre_trim = len(self._buffer)
        whole = self._left_context + self._buffer
        offset = len(self._left_context)

        matches = list(iter_matches(whole, offset))

        hold = holdback(self._buffer, self.max_holdback)
        split = offset + len(self._buffer) - hold
        # Two layers. The tail rules hold text that might still *become* a match;
        # this pulls the split back behind anything that already *is* one. Both
        # run against `whole`, so they agree with each other and with the
        # whole-string pass about where matches begin.
        split = pull_back_behind_straddling_match(whole, split, offset, matches)
        safe_len = split - offset
        safe_raw = self._buffer[:safe_len]

        # No match up to max_holdback characters can straddle the split: every
        # character these patterns can contain is in the hold-back character
        # class (enforced by assert_holdback_covers_patterns), so a match
        # touching the end of the buffer lies entirely inside the held tail.
        # Redact with the previously emitted tail visible to lookbehinds, then
        # keep the new tail for the next chunk. The context is the *raw* text,
        # because that is what a whole-string scan would have seen there.
        emit = redact_range(whole, offset, split, self.stats, self.placeholder, matches)
        self._left_context = (self._left_context + safe_raw)[-LOOKBEHIND_CONTEXT:]

        self._buffer = self._buffer[safe_len:]
        # Recorded on the pre-trim buffer as well as the post-trim one. Measuring
        # only after trimming could never observe the peak: a single 1 MB delta
        # showed max_buffer = 19 while a million characters were resident.
        self.stats.max_buffer = max(self.stats.max_buffer, pre_trim, len(self._buffer))
        self.stats.characters_out += len(emit)
        return emit

    def flush(self) -> str:
        """Emit the tail. Nothing more can arrive, so nothing needs holding back."""
        if self._closed:
            return ""
        self._closed = True
        combined = self._left_context + self._buffer
        remainder = redact_complete_from(
            combined, len(self._left_context), self.stats, self.placeholder
        )
        self._left_context = ""
        self._buffer = ""
        self.stats.characters_out += len(remainder)
        return remainder
