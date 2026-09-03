"""Unit tests for the incremental redactor: correctness and boundedness."""

import pytest

from redactor import (
    MAX_BUFFER,
    MAX_HOLDBACK,
    StreamRedactor,
    assert_holdback_covers_patterns,
    holdback,
    luhn_ok,
    redact_complete,
)


def stream(chunks, **kwargs) -> tuple[str, StreamRedactor]:
    redactor = StreamRedactor(**kwargs)
    out = "".join(redactor.feed(chunk) for chunk in chunks)
    return out + redactor.flush(), redactor


# --------------------------------------------------------------------------
# Whole-string redaction
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail ada@example.com now", "mail [REDACTED] now"),
        ("ssn 123-45-6789 ok", "ssn [REDACTED] ok"),
        ("ssn 123456789 ok", "ssn [REDACTED] ok"),
        ("card 4111111111111111 ok", "card [REDACTED] ok"),
        ("card 4111 1111 1111 1111 ok", "card [REDACTED] ok"),
        ("card 4111-1111-1111-1111 ok", "card [REDACTED] ok"),
        ("call 555-010-1234 now", "call [REDACTED] now"),
        ("key sk-live-abcdefghij0123456789 ok", "key [REDACTED] ok"),
    ],
)
def test_complete_patterns(text, expected):
    assert redact_complete(text) == expected


def test_multiple_hits_in_one_string():
    out = redact_complete("ada@example.com and 123-45-6789")
    assert out == "[REDACTED] and [REDACTED]"


# --------------------------------------------------------------------------
# False positives
# --------------------------------------------------------------------------
def test_non_luhn_long_number_is_not_a_card():
    """A 16-digit order id must survive; only Luhn-valid numbers are cards."""
    assert "1234567890123456" in redact_complete("Order 1234567890123456 shipped")


def test_luhn_check_is_real():
    assert luhn_ok("4111111111111111") is True
    assert luhn_ok("1234567890123456") is False


def test_never_issued_ssn_ranges_are_not_ssns():
    assert "000-00-0000" in redact_complete("Placeholder 000-00-0000 here")


def test_ordinary_prose_is_untouched():
    text = "The quick brown fox jumps over the lazy dog, 42 times."
    assert redact_complete(text) == text


# --------------------------------------------------------------------------
# Streaming: the point of the exercise
# --------------------------------------------------------------------------
def test_ssn_split_across_four_chunks():
    out, _ = stream(["Her SSN is 123", "-45", "-67", "89, filed."])
    assert out == "Her SSN is [REDACTED], filed."


def test_email_split_across_chunks():
    out, _ = stream(["Write to a", "da.love", "lace@exa", "mple.com today"])
    assert out == "Write to [REDACTED] today"


def test_card_with_spaces_split_mid_group():
    out, _ = stream(["Card 4111 ", "1111 ", "1111 ", "1111 charged."])
    assert out == "Card [REDACTED] charged."


def test_character_by_character_is_still_redacted():
    """The worst case: no single chunk contains a complete pattern."""
    out, _ = stream(list("Email: ada@example.com now"))
    assert out == "Email: [REDACTED] now"


def test_pii_at_the_very_end_is_flushed_not_truncated():
    """A response ending in PII must not silently lose its tail."""
    out, _ = stream(["Contact ", "ada@example.com"])
    assert out == "Contact [REDACTED]"


def test_partial_pii_at_end_is_emitted_verbatim():
    """An incomplete pattern is not PII; it must survive flush()."""
    out, _ = stream(["Her SSN starts with 123-45"])
    assert out.endswith("123-45")


CORPUS = [
    # Every pattern, and every separator style each one allows.
    "Ada (ada@example.com, SSN 123-45-6789) paid with 4111 1111 1111 1111 "
    "and her key sk-live-abcdefghij0123456789. Call 555-010-1234.",
    # Parenthesised phone. This corpus line is why the bug below was found.
    "Call me at (555) 123-4567 tomorrow",
    # Parenthesised phone ending the string, so the numeric tail rule cannot
    # reach the end of the buffer.
    "Call (555) 123-4567.",
    "Reach ada@example.com or (555) 123-4567; card 4111-1111-1111-1111.",
    "International: +1 555-010-1234 works too.",
    # Near-misses that must survive untouched.
    "Order 1234567890123456 shipped. Placeholder 000-00-0000.",
    "The quick brown fox (jumps) over the lazy dog, 42 times (really).",
    # PII at the very start and the very end.
    "ada@example.com is the address",
    "the address is ada@example.com",
]

CHUNK_SIZES = (1, 2, 3, 4, 5, 7, 11, 13, 17, 29, 64, 200)


@pytest.mark.parametrize("text", CORPUS)
@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_streaming_matches_whole_string_redaction(text, size):
    """Property: chunking must never change the result.

    This is the test that matters most, and its value is entirely in the corpus.
    An earlier version used only "555-010-1234" and therefore passed while a
    parenthesised number split across a chunk boundary was being emitted in the
    clear: ")" was not in the hold-back character class, so the held tail started
    *after* the match began. Adding "(555) 123-4567" here fails at chunk sizes 1
    through 17 against that version.
    """
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    out, _ = stream(chunks)
    assert out == redact_complete(text), f"chunk size {size} diverged"


def test_holdback_alphabet_covers_every_pattern():
    """Structural guard: a new pattern cannot introduce an unheld character."""
    assert_holdback_covers_patterns()


def test_parenthesised_phone_split_across_chunks():
    """The specific regression, named, so it cannot come back quietly."""
    out, _ = stream(["Call me at (555) 123-456", "7 tomorrow"])
    assert out == "Call me at [REDACTED] tomorrow"

    out, _ = stream(["Call (555", ") 123-4567."])
    assert out == "Call [REDACTED]."


def test_a_complete_match_is_never_split_by_the_holdback():
    """pull_back_behind_straddling_match, exercised directly."""
    from redactor import COMBINED, pull_back_behind_straddling_match

    buffer = "Call (555) 123-4567."
    match = next(COMBINED.finditer(buffer))
    # A naive split lands inside the number.
    naive_split = len(buffer) - len("4567.")
    assert match.start() < naive_split < match.end()
    assert pull_back_behind_straddling_match(buffer, naive_split) == match.start()


# --------------------------------------------------------------------------
# Memory and latency bounds
# --------------------------------------------------------------------------
def test_holdback_alone_never_exceeds_its_cap():
    """At rest the hold-back is capped; the peak also carries the incoming chunk.

    max_buffer is measured before trimming, so it legitimately includes the chunk
    just appended -- a chunk has to be in memory before it can be examined. The
    resting bound is what the memory claim is about.
    """
    chunk = 100
    text = "x" * 10_000  # one enormous unbroken token, no match in it
    _, redactor = stream([text[i : i + chunk] for i in range(0, len(text), chunk)])
    assert redactor.stats.max_buffer <= MAX_HOLDBACK + chunk
    assert redactor.buffered <= MAX_HOLDBACK


def test_straddle_guard_may_exceed_the_holdback_but_not_the_buffer_bound():
    """The honest bound, and why it is not MAX_HOLDBACK.

    pull_back_behind_straddling_match moves the split to the *start* of a match
    spanning it, and that start can sit further back than the hold-back window.
    Five-character chunks of card numbers reach ~145 against a 128 hold-back.
    Holding the whole match is the only safe option -- the alternative is
    emitting its first half -- so the bound is MAX_HOLDBACK + one match, which is
    still O(1) in response length. This test exists because the earlier bound
    tests used input containing no match at all and so could never see it.
    """
    chunk = 7
    text = "x" * 500 + "a" * 60 + "@example.com" + "y" * 500
    out, redactor = stream([text[i : i + chunk] for i in range(0, len(text), chunk)])
    assert out == redact_complete(text)
    assert redactor.stats.max_buffer > MAX_HOLDBACK
    assert redactor.stats.max_buffer <= MAX_BUFFER + chunk


def test_buffer_stays_small_over_a_long_clean_response():
    chunks = [f"Paragraph {i} of ordinary prose with no sensitive data. " for i in range(2000)]
    _, redactor = stream(chunks)
    assert redactor.stats.max_buffer <= MAX_HOLDBACK
    assert redactor.stats.characters_in == redactor.stats.characters_out


def test_first_chunk_with_a_space_emits_immediately():
    """TTFT: text before the trailing token goes out on the first feed()."""
    redactor = StreamRedactor()
    assert redactor.feed("Hello there") == "Hello "


def test_holdback_only_withholds_the_trailing_word():
    """Prose flows: a space ends the run, so only the final word is held.

    The period stays inside the run because "." is legal inside an email domain
    -- "ada@example." could still become "ada@example.com". Holding one word is
    the cost of never emitting half an address.
    """
    assert holdback("A complete sentence.") == len("sentence.")
    assert holdback("Hello there") == len("there")
    assert holdback("Ends with a space ") == 0


def test_holdback_is_bounded():
    assert holdback("a" * 5000) == MAX_HOLDBACK


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------
def test_stats_count_by_category():
    _, redactor = stream(["ada@example.com and 123-45-6789 and 4111111111111111"])
    assert redactor.stats.redactions == {"EMAIL": 1, "SSN": 1, "CREDIT_CARD": 1}
    assert redactor.stats.total_redactions == 3


def test_feed_after_flush_is_an_error():
    redactor = StreamRedactor()
    redactor.feed("hello ")
    redactor.flush()
    with pytest.raises(RuntimeError):
        redactor.feed("more")


def test_flush_is_idempotent():
    redactor = StreamRedactor()
    assert redactor.feed("ada@example.com") == ""  # entirely held: it may still grow
    assert redactor.flush() == "[REDACTED]"
    assert redactor.flush() == ""


def test_a_growing_match_is_not_judged_early():
    """The bug the feed() ordering prevents.

    "ada@example.co" is already a complete match of the email pattern. Redacting
    the buffer before splitting would emit "[REDACTED]" and then let the "m"
    arrive after it.
    """
    out, _ = stream(["ada@example.co", "m.au is the address"])
    assert out == "[REDACTED] is the address"


def test_long_email_is_not_split_across_the_holdback_boundary():
    """A full-length RFC-legal address survives chunking intact."""
    address = "a" * 60 + "@example.com"  # 60 <= the 64-octet local part limit
    out, _ = stream([address[i : i + 7] for i in range(0, len(address), 7)])
    assert out == "[REDACTED]"


def test_over_length_local_part_matches_only_its_legal_tail():
    """The EMAIL pattern is length-bounded, and streaming agrees with that.

    A 90-character local part is not a legal address, so only the trailing 64
    octets match -- and the streamed result must be the same as the whole-string
    result, not something else. Bounding the pattern is what makes MAX_BUFFER a
    real number: with an unbounded "+" a single match can be as long as the
    response, and "a"*40 + "@" + "bb."*1000 + "com" buffered 3,044 characters.
    """
    address = "a" * 90 + "@example.com"
    out, redactor = stream([address[i : i + 7] for i in range(0, len(address), 7)])
    assert out == redact_complete(address)
    assert out.endswith("[REDACTED]")
    assert redactor.stats.max_buffer <= MAX_BUFFER + 7


def test_unbounded_pattern_would_break_the_memory_bound():
    """Pin the reason EMAIL carries {1,64}/{1,255} rather than +."""
    text = "mail me at " + "a" * 40 + "@" + "bb." * 1000 + "com and bye"
    _, redactor = stream([text[i : i + 50] for i in range(0, len(text), 50)])
    assert redactor.stats.max_buffer <= MAX_BUFFER + 50
    assert redactor.buffered <= MAX_BUFFER


# --------------------------------------------------------------------------
# Differential fuzz
# --------------------------------------------------------------------------
def test_differential_fuzz_streaming_equals_whole_string():
    """Randomised equivalence over PII glued together with awkward filler.

    The hand-written corpus above is the readable version of this property; this
    is the thorough one. The filler deliberately includes "-", "(" and ")", the
    characters that sit next to a pattern boundary and interact with the
    lookbehind fences -- which is exactly where streaming used to diverge from
    whole-string redaction before redact_complete_from() carried the emitted tail
    into the lookbehind window.
    """
    import random

    pii = [
        "ada@example.com",
        "a.b+c@sub.example.co.uk",
        "123-45-6789",
        "123456789",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "4111111111111111",
        "(555) 123-4567",
        "+1 555-010-1234",
        "555-010-1234",
        "sk-live-abcdefghij0123456789",
    ]
    filler = ["hello ", "the total is ", "x", "-", "(", ")", ". ", "  ", "note:", "42", ",", "a" * 30, "9" * 14]

    random.seed(7)
    divergences = []
    for _ in range(600):
        text = "".join(random.choice(pii + filler) for _ in range(random.randint(1, 9)))
        expected = redact_complete(text)
        for size in (1, 2, 3, 5, 7, 11, 17, 64):
            out, redactor = stream([text[i : i + size] for i in range(0, len(text), size)])
            assert redactor.stats.max_buffer <= MAX_BUFFER + size
            if out != expected:
                divergences.append((size, text, out, expected))

    assert not divergences, f"{len(divergences)} divergences, first: {divergences[0]}"


def test_placeholder_is_configurable():
    """The constructor argument has to actually do something.

    It was accepted, stored on the instance and then ignored, because the
    redaction functions hard-coded "[REDACTED]". A parameter that silently does
    nothing is worse than no parameter.
    """
    redactor = StreamRedactor(placeholder="<<PII>>")
    out = redactor.feed("ssn 123-45-6789 and ada@example.com ok") + redactor.flush()
    assert out == "ssn <<PII>> and <<PII>> ok"
    assert "[REDACTED]" not in out


def test_default_placeholder_is_unchanged():
    assert redact_complete("ssn 123-45-6789") == "ssn [REDACTED]"


def test_max_buffer_stat_observes_the_true_peak():
    """The statistic used to be measured after trimming, so it never saw a peak.

    A single one-megabyte delta reported max_buffer = 19 while a million
    characters were resident. Memory is O(1) in *response* length, but a chunk
    has to be appended before it can be examined, so it is not O(1) in *chunk*
    size -- and the number ought to say so.
    """
    redactor = StreamRedactor()
    redactor.feed("x" * 1_000_000)
    redactor.flush()
    assert redactor.stats.max_buffer >= 1_000_000


def test_buffer_returns_to_the_bound_after_a_huge_chunk():
    redactor = StreamRedactor()
    redactor.feed("word " * 200_000)
    assert redactor.buffered <= MAX_BUFFER


def test_adversarial_input_does_not_blow_up_cpu():
    """A cheap regression guard on the hot path.

    `"1 "` repeated is the worst case for this design: every digit is preceded by
    a space, so every position is a valid CREDIT_CARD start and every candidate
    needs a Luhn check. Before the fix, 10 KB of it took 13.2s at chunk=1 and
    drove 930,000 Luhn calls -- the whole profile. Two changes brought it to
    ~1.3s: feed() now scans once and shares the match list with the straddle
    guard, and _is_card/luhn_ok are memoised because the sliding buffer
    re-examines the same spans repeatedly.

    The threshold is deliberately loose so this fails on a regression in kind,
    not on a slow machine.
    """
    import time

    text = "1 " * 5000
    redactor = StreamRedactor()
    started = time.perf_counter()
    for i in range(0, len(text), 4):
        redactor.feed(text[i : i + 4])
    redactor.flush()
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0, f"{elapsed:.2f}s for 10 KB of adversarial input"


def test_leak_family_that_defeated_the_straddle_guard():
    """Adjacent PII whose concatenation is itself a valid match.

    The guard scanned from 0 while the emitter scanned from `offset`. Because the
    retained context is truncated, the guard's scan could start mid-value and
    segment an ambiguous chain of adjacent matches out of phase with the
    emitter's. The guard then saw a clean boundary at the split, declined to pull
    back, and the emitter's `break` published the head of a real match -- after
    which the buffer was trimmed past it, so it was never redacted.

    Each of these leaked complete SSNs, emails or card numbers.
    """
    for text in (
        "123-45-6789 " * 150,
        "123-45-6789 " * 69,
        "x@y.io" * 150,
        "ada@example.com" * 150,
        "4111 1111 1111 11114111 1111 1111 1111 " * 33,
        "0 " * 600,
    ):
        expected = redact_complete(text)
        for size in (1, 2, 3, 5, 64, 401, 800):
            out, _ = stream([text[i : i + size] for i in range(0, len(text), size)])
            assert out == expected, f"{text[:20]!r} diverged at chunk size {size}"
