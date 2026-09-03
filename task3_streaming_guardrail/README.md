# LLM gateway streaming guardrail (PII redaction)

An LLM gateway that proxies `POST /v1/chat/completions` to a provider and
rewrites the SSE stream on the way back, replacing emails, SSNs, card numbers,
phone numbers and API keys with `[REDACTED]` — without ever holding the full
response in memory.

## Run it

```bash
uvicorn mock_llm:mock_llm_app --port 9011
LLM_UPSTREAM_URL=http://127.0.0.1:9011/v1/chat/completions uvicorn llm_gateway:app --port 9010

curl -N localhost:9010/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mock-model-1","stream":true,"scenario":"split_pii"}'

python -m pytest tests -q     # 190 tests
python benchmark.py           # TTFT and memory, over real sockets
```

Scenarios in `mock_llm.py` control exactly where chunk boundaries land:
`clean`, `split_pii`, `split_card`, `char_by_char`, `long_clean`,
`false_positives`, `api_key`.

---

## The actual problem

Redacting a finished string is a one-liner. Redacting a *stream* is not, because
the model does not emit PII in convenient units:

```
chunk 1: "Her SSN is 123"
chunk 2: "-45"
chunk 3: "-67"
chunk 4: "89, please file it."
```

Forward chunk 1 as it arrives and `123` is already on the user's screen when
chunk 2 reveals what it was the start of. **You cannot un-send a token.**

Buffering the whole response and redacting at the end is correct and useless:
time-to-first-token becomes time-to-*last*-token, and memory grows with response
response length. Streaming is the product; that approach throws it away.

## The approach: emit only what is provably safe

On each chunk (`redactor.py`):

1. Append the chunk to a buffer.
2. Compute a **hold-back**: the length of the trailing run that could still grow
   into a match.
3. Redact complete matches in everything *before* the hold-back, and emit it.
4. Keep the rest.

`flush()` redacts and emits the remainder at end of stream, when nothing more
can arrive to extend it.

Two rules compute the hold-back:

| Rule | Catches |
| --- | --- |
| `[A-Za-z0-9@._%+\-]+\Z` | trailing run with no space: emails, dashed SSNs, API keys |
| `(?:[+(][\d ()\-]*|\d[\d ()\-]*)\Z` | trailing run starting a number: `4111 1111 1111 1111`, `(555) 123-4567` |

Prose is unaffected because a space ends the first run — `"Hello there"` holds
back five characters, not the sentence.

### Step order is the whole trick

Redacting the buffer *before* splitting looks equivalent and is not:

```python
# WRONG
buffer = redact_complete(buffer + chunk)   # "ada@example.co" is already a match
emit = buffer[:len(buffer) - holdback(buffer)]
# client sees: "[REDACTED]" ... then "m" arrives
```

`"ada@example.co"` is a complete match of the email pattern. Redact it early and
the `m` from the next chunk lands *after* the placeholder. Holding back first
means a pattern that is still growing is never judged. `test_a_growing_match_is_not_judged_early`
pins this.

### Bounds

- **Memory is O(1) in response length**, bounded by `MAX_BUFFER` =
  `MAX_HOLDBACK` + one match (800), plus `LOOKBEHIND_CONTEXT` (400) of retained
  already-emitted text. Two things make that number real, and both were wrong at
  some point:
  - The straddle guard pulls the split back to the *start* of a match spanning
    it, which can sit further back than the hold-back window. So the bound is
    hold-back **plus one match**, not hold-back. Holding the whole match is the
    only safe option — the alternative is emitting its first half.
  - "One match" is only finite because the `EMAIL` pattern is length-bounded to
    RFC 5321's 64-octet local part and 255-octet domain. With unbounded `+`
    quantifiers, `"a"*40 + "@" + "bb."*1000 + "com"` buffered 3,044 characters
    against a claimed 448, and the bound test could not see it because its input
    was card-shaped.
- **No match can straddle the split**, enforced two ways. The tail rules hold
  text that might still *become* a match, and their character class is checked
  against the patterns by `assert_holdback_covers_patterns()`. Separately,
  `pull_back_behind_straddling_match()` moves the split back behind any complete
  match that spans it, which makes the property structural rather than a claim
  about character classes. The one remaining condition is the cap: a match
  *longer* than `MAX_HOLDBACK` can still be split, which is why the cap has to
  exceed the longest pattern.
- **Latency is bounded by the next word boundary**, not by response length.

## Measured cost

`python benchmark.py`, upstream pacing 7 deltas 30 ms apart:

One representative run (the millisecond figures move a few ms between runs; the
shape does not):

```
TTFT direct from mock  :    43.5 ms
TTFT through gateway   :    84.5 ms
guardrail adds         :    41.0 ms to TTFT
a buffering proxy would:   210.0 ms

median inter-frame gap : direct  34.6 ms   gateway  34.5 ms

clean text streamed    : 3.24 MB
  peak hold-back       : 65 chars (bound 800)
  peak traced alloc    : 4.9 KiB
PII-dense streamed     : 2.49 MB, 100,000 redactions
  peak hold-back       : 66 chars (bound 800)
```

The TTFT cost is roughly one inter-delta gap, and it is explainable: the
first delta `"Hello"` is a trailing token with no word boundary after it, so it
is held until the next delta proves it is not the start of an email address.
That cost is a function of **token pacing, not response length** — the gap does
not compound, as the near-identical inter-frame medians show. 3.24 MB of text
moves through the redactor with a few KiB of peak allocation.

## Pattern order is a correctness property

`CREDIT_CARD` is matched *before* `SSN`, and both are fenced with
`(?<![\d\-]) ... (?![\d\-])`. Without both, the nine-digit SSN rule matches the
first nine digits of a card and produces `card [REDACTED]1111111` — a redaction
that publishes seven digits of a card number. `test_complete_patterns` pins it.

Relatedly, a Luhn rejection must not *consume* the span it rejected. Returning
the span verbatim makes the scan resume at its end, so real PII the run had
swallowed is never examined — `"Invoice 12345 4111 1111 1111 1111"` leaves the
card untouched, because the invoice number shifts the greedy match one digit left
and Luhn then fails. `iter_matches()` advances a single character on rejection
instead.

## False positives are a product bug too

A guardrail that redacts order numbers is a guardrail somebody turns off.

| Input | Result | Why |
| --- | --- | --- |
| `4111 1111 1111 1111` | `[REDACTED]` | Luhn-valid |
| `Order 1234567890123456` | untouched | fails Luhn |
| `123-45-6789` | `[REDACTED]` | valid SSN shape |
| `000-00-0000` | untouched | never-issued area number |

## Transport details

| Concern | Handling |
| --- | --- |
| Redaction spanning frames | Output frames do not map 1:1 onto input frames |
| A frame entirely held back | Dropped, not emitted empty — some clients render an empty delta as a flicker |
| The tail | Flushed on `finish_reason` or `[DONE]`; without this a response ending in PII is silently truncated |
| Role / usage / unknown frames | Forwarded verbatim — a guardrail that strips unrecognised fields breaks on the provider's next release |
| An unparseable frame | Forwarded. Failing open on *shape* while failing closed on *content* is the right trade |
| Non-streaming requests | Same patterns, one pass. `test_streaming_and_non_streaming_agree` issues a real `stream:false` request through the gateway, so the two paths cannot drift |
| Upstream timeout, 4xx/5xx, or a non-JSON body | Clean 502/504 on both paths. `httpx` puts the upstream host and port in its exception text, so it never reaches the caller |

## A note on the test layout

The latency tests in `test_streaming_latency.py` run two real uvicorn servers on
ephemeral ports;
everything else runs in-process over `httpx.ASGITransport`. That split is
deliberate: **ASGITransport collects the whole response body before returning
it**, so an in-process test cannot tell a streaming gateway from a buffering
one. The first version of these tests "failed" for exactly that reason — the
harness was the buffer, not the gateway. Latency claims need real sockets.

## Production notes

- `MAX_HOLDBACK` must be ≥ the longest match any pattern can produce. Adding a
  pattern means revisiting it.
- The heuristic hold-back over-holds rather than under-holds, which is the
  direction the error has to fall. `holdback_exact()` is included as the precise
  version using the third-party `regex` module's partial matching, for
  comparison; the heuristic is the default because it needs no dependency.
- Regex is the floor, not the ceiling. Names and addresses need an NER pass, and
  at that point the design changes: an ML classifier cannot run per-character, so
  you redact at sentence boundaries and accept the extra latency.
- At high volume, replace the Python regex pass with Hyperscan or a compiled
  DFA. The stream plumbing here does not change.

## Test coverage

```
tests/test_redactor.py          150  patterns, bounds, false positives, an
                                     equivalence property, a differential fuzz,
                                     the leak family, and a CPU guard
tests/test_sse_robustness.py     12  frames the provider sends that this code
                                     did not design for
tests/test_stream_proxy.py       23  end-to-end SSE, the non-streaming path,
                                     and upstream failure handling on both
tests/test_streaming_latency.py   5  TTFT and progressive delivery over real sockets
```
