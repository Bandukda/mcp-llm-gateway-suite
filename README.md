# MCP and LLM gateway suite

[![tests](https://github.com/Bandukda/mcp-llm-gateway-suite/actions/workflows/tests.yml/badge.svg)](https://github.com/Bandukda/mcp-llm-gateway-suite/actions/workflows/tests.yml)

Four small services for the path between an AI agent and the systems it talks to.

| | Service | What it does |
| --- | --- | --- |
| 1 | `task1_mcp_server/` | An MCP tool server on stdio, with strict input validation |
| 2 | `task2_mcp_gateway/` | A security gateway in front of MCP servers — decides who may call which tool |
| 3 | `task3_streaming_guardrail/` | An LLM gateway that redacts PII out of a streaming response |
| 4 | `task4_router/` | A rate limiter and fallback router for model providers |

Everything runs offline. Mock providers and a mock downstream MCP server are
included, so there are no API keys to set and nothing to sign up for.

## Getting started

**Requires Python 3.10 or newer**, because the MCP SDK does not publish builds
for anything older. `python3 --version` will tell you. If yours is older, install
a newer one (`brew install python@3.12`) and point the setup script at it with
`PYTHON=python3.12 ./setup.sh`.

```bash
./setup.sh                  # creates a venv and installs dependencies
source .venv/bin/activate
./run_all_tests.sh          # 403 tests, plus two standalone proofs
```

## Documentation

Two documents, in the order worth reading them:

**[`docs/code-walkthrough.md`](docs/code-walkthrough.md)** — start here. Each
service in turn: the problem in plain words, why the obvious approach fails, and
the code that handles it.

**[`docs/architecture.md`](docs/architecture.md)** — how the four compose into
one system, a request traced end to end, and what is deliberately left out.

Each service directory also has its own README covering the design decisions
specific to it.

## Running a service

```bash
# 1. MCP tool server — speaks JSON-RPC on stdin/stdout
cd task1_mcp_server && python server.py

# 2. MCP security gateway, with its mock downstream
cd task2_mcp_gateway
uvicorn downstream:downstream_app --port 9001 &
MCP_DOWNSTREAM_URL=http://127.0.0.1:9001/mcp uvicorn mcp_gateway:app --port 9000

# 3. LLM gateway, with its mock provider
cd task3_streaming_guardrail
uvicorn mock_llm:mock_llm_app --port 9011 &
LLM_UPSTREAM_URL=http://127.0.0.1:9011/v1/chat/completions uvicorn llm_gateway:app --port 9010

# 4. Model router
cd task4_router && python app.py
```

## Two things worth seeing run

**The MCP server keeps stdout clean.** An MCP stdio server speaks JSON-RPC on
file descriptor 1, so a stray `print` corrupts the stream — and a `print` of a
JSON-shaped string is a forged response the client will believe. This proves the
guard rather than asserting it:

```bash
cd task1_mcp_server
python verify_stdout_purity.py              # the stream stays clean
python verify_stdout_purity.py --unguarded  # the same server without the guard
```

Both pass. The second removes the guard and checks that the stream *does* get
corrupted, which is what makes the first result mean anything:

```
PASS - the unguarded server corrupts the stream, as expected
  9 stdout lines, of which 4 are not protocol
  including a forged response a client would have accepted:
      {"jsonrpc": "2.0", "id": 999, "result": {"spoofed": true}}
```

**The streaming guardrail does not buffer the response.** Redacting a finished
string is easy; doing it to a stream without holding the whole thing in memory is
the actual problem:

```bash
cd task3_streaming_guardrail && python benchmark.py
```

One run on a laptop. The millisecond figures move a few ms run to run; the
ratio between them is the part that holds.

```
TTFT direct from mock  :    43.5 ms
TTFT through gateway   :    84.5 ms
a buffering proxy would:   210.0 ms

3.24 MB streamed, 4.9 KiB peak allocation
```

The added cost is roughly one inter-delta gap, and it does not compound with
response length — `benchmark.py` prints the inter-frame medians alongside, which
stay level.

## Requirements

Python 3.10 or newer. The MCP SDK is pinned to `>=1.9,<2`; see `requirements.txt`.

## Layout

```
.
├── setup.sh                     venv + dependencies
├── run_all_tests.sh             every suite plus the standalone proofs
├── requirements.txt
├── docs/
│   ├── code-walkthrough.md
│   └── architecture.md
├── task1_mcp_server/
├── task2_mcp_gateway/
├── task3_streaming_guardrail/
└── task4_router/
```
