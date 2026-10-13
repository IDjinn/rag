# rag

Local semantic search for ZCode. One static Rust binary indexes your
repositories into a local vector store and answers searches over MCP —
nothing leaves your machine, and cost/utility stats tell you whether
indexing is paying off.

- **Multi-provider embeddings** — any OpenAI-compatible `/v1/embeddings`
  backend: LM Studio (default), Ollama, or a cloud provider via
  `RAG_PROVIDER` + `RAG_API_KEY`.
- **Self-maintaining index** — the first `rag index` per root is manual;
  after that the MCP server re-embeds changed files automatically, and
  moved/renamed/copied files reuse their existing vectors for free
  (content-hash match).
- **Fast search** — SIMD cosine kernel over memory-mapped vectors,
  rayon-parallel: tens of milliseconds across 50k chunks.
- **Priority tiers** — source code and markdown (P0) outrank configs and
  YAML (P1) at search time. `.gitignore`, `.zcodeignore` and hidden files
  (so `.env`) are always respected.
- **Honest stats** — embedding tokens/time, searches served, tokens saved
  per search, and a judged usefulness rate with a verdict line.

## Install

```bash
cargo install --path .   # from a checkout; or grab a binary from Releases
```

## Quick start

```bash
rag index .                        # build / incrementally update the index
rag search "where is the token renewed"
rag serve                          # MCP stdio server
```

Register it in `~/.zcode/cli/config.json` (user scope):

```json
"rag": { "command": "C:\\path\\to\\rag.exe", "args": ["serve"] }
```

Restart ZCode (or Settings → MCP) afterwards. Exposed tools:
`rag_status`, `rag_index`, `rag_search`, `rag_feedback`, `rag_stats`,
`rag_clear`. Bare `rag` prints help.

## Commands

| Command | What it does |
| --- | --- |
| `rag index [PATH]` | Build or update the index (`--force`, `--dry-run`, `--no-progress`) |
| `rag search "q" [--k 5] [--judge]` | Semantic search; `--judge` returns a ready-to-run `jev_triage` payload |
| `rag stats [PATH]` | Cost/utility report (`--reset` clears it) |
| `rag status [PATH]` | Files, chunks, model, last index time |
| `rag feedback --json '...'` | Record usefulness verdicts from searches |
| `rag clear [PATH]` | Delete the index and its stats |
| `rag serve` | MCP stdio server |
| `rag hook` | ZCode hooks (see below) |
| `rag config [--init]` | Print effective config / write a commented default |

Live progress bars render on a TTY; pipes and CI get plain one-line output
on stderr.

## Providers & configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_PROVIDER` | `lmstudio` | `ollama` or `openai` (any OpenAI-compatible cloud) |
| `RAG_BASE_URL` | per provider | LM Studio `:1234/v1`, Ollama `:11434/v1`, OpenAI `/v1` |
| `RAG_EMBED_MODEL` | auto-detect | Pin the embedding model |
| `RAG_API_KEY` | — | Bearer token for cloud providers |
| `RAG_DATA_DIR` | `~/.zcode/ia` | Where indexes and stats live |
| `RAG_AUTO_REFRESH` / `_SECS` | `1` / `10` | Background auto-refresh on/off and interval |

Beyond env vars, `~/.zcode/ia/config.yaml` tunes batch size, concurrency,
tier lists and search (precedence: CLI flag > `RAG_*` env > YAML >
default). `rag config` prints the effective values; `rag config --init`
writes a fully commented default.

LM Studio tips: keep an **embedding-classified** model loaded (e.g.
`text-embedding-nomic-embed-text-v1.5`), and keep `concurrency: 1` — one
big batch beats parallel small requests on the local engine (~18×
degradation under contention).

## Hooks (optional)

`rag hook` makes ZCode passively aware of the index: on `SessionStart` it
injects a note telling the model to prefer `rag_search`, and on
`PreToolUse(Agent)` it reminds the model to paste relevant hits into
subagent prompts (subagents have no MCP access). Register the same binary
with `args: ["hook"]` under the `SessionStart` and `PreToolUse` (matcher
`Agent|Task`) events in `~/.zcode/cli/config.json`. Unindexed directories
get silence; it never touches the network and always exits 0.

## Development

```bash
cargo test                       # unit tests incl. SIMD-vs-scalar parity
cargo build --release            # target/release/rag.exe
cargo run -- index --dry-run .   # audit what would be indexed, per tier
```
