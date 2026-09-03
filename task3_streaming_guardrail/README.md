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

## False positives are a product bug too

A guardrail that redacts order numbers is a guardrail somebody turns off.

| Input | Result | Why |
| --- | --- | --- |
| `4111 1111 1111 1111` | `[REDACTED]` | Luhn-valid |
| `Order 1234567890123456` | untouched | fails Luhn |
| `123-45-6789` | `[REDACTED]` | valid SSN shape |
| `000-00-0000` | untouched | never-issued area number |

### Bugs worth calling out

The first draft ordered `SSN` before `CREDIT_CARD` in the alternation. The SSN
rule matched the first nine digits of a card and produced:

```
card [REDACTED]1111111 ok
```

A redaction that publishes seven digits of a card number. The fix was ordering
`CREDIT_CARD` first *and* fencing both rules with `(?<![\d\-]) … (?![\d\-])` so
neither can match inside a longer digit run. Alternation order is a correctness
property here, not a style choice, and `test_complete_patterns` pins it.

The second one was found by review, not by the tests, which is the more useful
story. The hold-back character classes did not contain `(` or `)` — but the phone
pattern does. So:

```python
stream(["Call me at (555) 123-456", "7 tomorrow"])
# -> "Call me at (555) 123-4567 tomorrow"   the whole number, in the clear
```

The `)` truncated the held tail, the match was cut in half, and neither half
matched alone. The equivalence property test was the right test to catch it and
did not, because its corpus only contained `555-010-1234`. Three things changed:
the character class was widened, `pull_back_behind_straddling_match()` was added
so the property no longer depends on getting the class right,
`assert_holdback_covers_patterns()` now fails loudly if a new pattern introduces
an uncovered character, and the corpus grew to nine texts across twelve chunk
sizes. Against the old code, the new corpus fails at chunk sizes 1 through 17.

The third only showed up under differential fuzzing, and it is the subtlest.
Several patterns are fenced with `(?<![\d\-])`, and **a lookbehind at position 0
of a truncated string sees "start of input"** — which satisfies a negative
lookbehind even when the real preceding character was a digit or a dash. So
redacting each safe prefix in isolation gave slightly different answers from
redacting the whole response:

```
"-(555) 123-4567"    whole string: "-([REDACTED]"    streamed: "-[REDACTED]"
```

Both are safe — streaming redacted *more* — but "chunking never changes the
result" is a far easier property to test and to trust than "chunking never makes
it worse". `re`'s `finditer(string, pos)` keeps lookbehinds looking behind `pos`,
so `redact_complete_from()` carries eight characters of already-emitted context
into the scan. Fuzzing 6,000 generated PII-dense texts across 8 chunk sizes went
from 2,996 divergences in 48,000 runs to **zero**.

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


## The bugs review and fuzzing found in this file

Kept here with their tests, because how they were found matters more than the
patches.

**1. `(` and `)` missing from the hold-back character class.** The phone pattern
contains them; the tail rules did not. `"Call me at (555) 123-456"` + `"7 tomorrow"`
streamed the whole number in the clear. The equivalence property test was exactly
the right test and missed it, because its corpus held only a dash-separated
number. Fixed structurally — `pull_back_behind_straddling_match()` plus
`assert_holdback_covers_patterns()` — not just by widening the class.

**2. A Luhn rejection consumed the span it rejected.** `CREDIT_CARD` matches any
13–19 digit run; Luhn decides whether it is really a card. Returning the span
verbatim made the scan resume at its *end*, so any real PII the run had swallowed
was never examined:

```
"Invoice 12345 4111 1111 1111 1111 was charged."   ->  unchanged
```

The five-digit invoice number pulls the greedy match one digit left, Luhn fails,
and the whole card is published. Roughly 70% of cards behind a short digit prefix
leaked this way, on both the streaming and non-streaming paths. `iter_matches()`
now advances a single character on rejection instead of consuming the span. It is
the mirror image of bug 4 below: there one pattern ate another, here a *rejected*
match ate a real one.

**3. Both edges of the buffer lied to the regex.** A lookbehind at position 0 of a
truncated string sees "start of input", satisfying `(?<![\d\-])`; end-of-string
likewise satisfies the trailing `(?![\d\-])` fence. The left edge made streaming
redact slightly *more* than a whole-string pass. The right edge was worse:

```
"Card 9039 3080 7022 682 " + "8"*140
```

Cut inside the run of eights, the greedy match becomes 16 digits, fails Luhn, and
the real card is emitted in the clear. `redact_range()` now scans the complete
buffer and emits only `[start, end)`, so the match found is always the one the
whole-string pass finds. A related case needed `LOOKBEHIND_CONTEXT` raised from 8
to `MAX_MATCH`: the straddle guard has to see matches that *begin* before the
buffer, not merely evaluate a lookbehind.

**4. `SSN` ordered before `CREDIT_CARD`.** The SSN rule matched the first nine
digits of a card and produced `card [REDACTED]1111111` — a redaction publishing
seven digits. Fixed by ordering and by fencing both rules with digit lookarounds.

After all four: a differential fuzz of 3,600 adversarial texts (every PII type
glued with digit prefixes, 400-character digit runs, over-length emails and bare
punctuation) across 6 chunk sizes — **21,600 streamings, 0 divergences from
whole-string redaction, 0 leaks, peak buffer 492 against a bound of 800.**


---

## Round two: what a dedicated attack pass found

After the four bugs above were fixed, an adversarial pass was run specifically
against this module. It found five more. They are worth reading in order,
because the first is the most instructive bug in the whole project.

### 1. The straddle guard and the emitter disagreed, and that was a leak

`feed()` ran two scans over the same buffer **from different origins**:

```python
pull_back_behind_straddling_match(whole, split, offset)   # iter_matches(whole, 0)
redact_range(whole, offset, split)                        # iter_matches(whole, offset)
```

`whole` is truncated at `LOOKBEHIND_CONTEXT`, so the guard's scan could begin
mid-value and segment an ambiguous chain of adjacent matches **out of phase**
with the emitter's. When they disagreed, the guard saw a clean boundary at the
split and declined to pull back; the emitter then found a match spanning it, hit
its `break`, and emitted that match's **head in the clear** — after which
`self._buffer = self._buffer[safe_len:]` discarded it, so it was never redacted.

The trigger is adjacent PII whose concatenation is itself a valid match,
sustained past ~820 characters:

| input | leaked verbatim |
| --- | --- |
| `"123-45-6789 " * 150` | 41 SSNs |
| `"ada@example.com" * 150` | 96 emails |
| `"4111 1111 1111 11114111 1111 1111 1111 " * 33` | 24 card numbers |

Confirmed end-to-end through `redact_sse_stream`: a client rendered 82 SSNs in
the clear that `redact_complete` masks. It did **not** fire on realistic varied
values with ordinary separators — 0 hits in 226 such trials — which is exactly
why the earlier fuzzing missed it.

The fix is one line and the reasoning is the whole point: **two scans of the same
text from the same origin cannot disagree.** The guard now scans from `offset`
too, and `feed()` computes the match list once and hands it to both.

### 2. Six ways a provider frame could abort the stream

`last_frame` was assigned before any shape validation, and `_with_content` did
`frame["choices"][0]`. Each of these killed the response mid-stream:

| frame | error |
| --- | --- |
| `{"choices": [], "usage": {...}}` with a held-back tail | `IndexError` |
| no `choices` key | `KeyError: 'choices'` |
| `choices` as a dict | `KeyError: 0` |
| `choices[0]` a string | `ValueError` |
| `choices: []` and no `[DONE]` | `IndexError` |
| any content frame after `[DONE]` | `RuntimeError: feed() after flush()` |

The first is not exotic. It is OpenAI's `stream_options={"include_usage": true}`
terminator.

### 3. Two redaction bypasses in the frame layer

Only `choices[0]` was inspected, so a frame whose first choice was a role delta
forwarded `choices[1]` **unredacted** — and the non-streaming path redacts every
choice, so the two paths disagreed about the same policy. Separately, non-`str`
`delta.content` (a structured content-block list, or a number) was passed through
verbatim. Both now go through the redactor or have the field dropped; unexamined
content is never forwarded.

### 4. The memory statistic could not observe a peak

`stats.max_buffer` was recorded *after* trimming. A single 1 MB delta reported
`max_buffer = 19` while a million characters were resident. Fixed, and the claim
corrected with it: memory is O(1) in **response** length, not in **chunk** size —
a chunk has to be appended before it can be examined. Real SSE deltas are a few
characters.

### 5. CPU amplification on adversarial input

`"1 "` repeated is the worst case: every digit is preceded by a space, so every
position is a valid card start and every candidate needs a Luhn check. 10 KB at
chunk=1 took **13.2 s**, driving 930,000 Luhn calls — the entire profile.

Two changes, no semantic effect:

```
              before    after
chunk=1      13.165s   1.256s
chunk=4       3.274s   0.324s
chunk=64      0.237s   0.022s
```

`feed()` scans once and shares the match list with the guard, and `_is_card` /
`luhn_ok` are memoised because the sliding buffer re-examines the same spans
repeatedly. Cost is linear in length, not superlinear, both before and after.
`test_adversarial_input_does_not_blow_up_cpu` guards it.

**After all five:** 24,549 streamings across the adversarial corpus and the exact
leak family — 0 divergences, 0 leaks.
