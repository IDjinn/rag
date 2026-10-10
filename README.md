# rag (Rust)

Local RAG for ZCode — a single static binary that indexes repository files into
a local vector index and serves semantic search, with explicit cost/utility
statistics so you can tell whether it is paying off.

- **Multi-provider embeddings** — anything speaking the OpenAI-compatible
  `/v1/embeddings` protocol: **LM Studio** (default), **Ollama**, or **OpenAI**
  / other clouds via `RAG_PROVIDER` + `RAG_API_KEY`.
- **Manual by design** — indexing only happens when you run `rag index`.
  Incremental: unchanged files are copied over without embeddings, so
  refreshes are cheap. Checkpoints every few thousand chunks keep work across
  crashes.
- **Interactive progress** — `rag index` shows live bars on a terminal
  (files + ETA, chunks, current file, tokens); under pipes/CI or with
  `--no-progress` it falls back to plain one-line reports on stderr.
- **Priority tiers** — P0 (source code + markdown) is embedded first and
  outranks P1 (configs, YAML, HTML, …) at search time (P1 score × 0.9).
- **Ignore rules respected** — `.gitignore`, `.ignore`, nested/global git
  excludes and ZCode's `.zcodeignore` are honored even outside git repos;
  hidden files are skipped (so `.env` never gets indexed).
- **Performance** — SIMD cosine kernel (`wide` f32x8, two accumulators),
  memory-mapped vectors (zero-copy search), rayon-parallel scoring, parallel
  file hashing/chunking, `target-cpu=native` + fat LTO. See the
  `throughput_report` test (`cargo test throughput_report -- --nocapture`).
- **Statistics** — embedding tokens/time/requests, searches served, context
  delivered vs full-file reads avoided, jev-judged usefulness rate, and a
  verdict line. With a cloud provider, set `RAG_PRICE_PER_1M_TOKENS` to get
  estimated USD spend.

## Usage

```bash
rag index [PATH] [--force] [--dry-run] [--batch-size N] [--concurrency N] [--no-ignore] [--no-progress]
rag search "query" [--k 5] [--judge] [--no-boost] [PATH]
rag stats [PATH] [--reset]
rag status [PATH]
rag clear [PATH]
rag feedback --json '{"verdicts":[{"id":"r1","useful":true}]}' [PATH]
rag serve        # MCP stdio server (also the default with no arguments)
```

As an MCP server it exposes `rag_status`, `rag_index`, `rag_search`,
`rag_feedback`, `rag_stats`, `rag_clear`. `rag_search` with `judge:true`
returns a ready-to-run `jev_triage` payload; feed the verdicts back through
`rag_feedback` and `rag_stats` reports the judged usefulness rate.

## Providers (env)

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_PROVIDER` | `lmstudio` | `ollama` or `openai` (any OpenAI-compatible cloud) |
| `RAG_BASE_URL` (or legacy `RAG_LMSTUDIO_URL`) | per provider | `http://localhost:1234/v1` (LM Studio), `http://localhost:11434/v1` (Ollama), `https://api.openai.com/v1` |
| `RAG_EMBED_MODEL` | auto-detect (`*embed*` in loaded models) | Pin the embedding model |
| `RAG_API_KEY` | — | Bearer token for cloud providers |
| `RAG_PRICE_PER_1M_TOKENS` | — | USD price for the cost report |
| `RAG_DATA_DIR` | `~/.zcode/ia` | Index/stat storage root |

LM Studio notes: keep at least one **embedding-classified** model loaded (e.g.
`text-embedding-nomic-embed-text-v1.5`) — models classified as LLM (such as
`qwen.qwen3-vl-embedding-2b`) do not initialize the embeddings pipeline alone,
and current LM Studio builds serve `/v1/embeddings` from the loaded embedding
engine regardless of the requested model id. The id is still recorded per
index so a future version that routes per-model triggers a rebuild instead of
mixing vector spaces. If LM Studio unloads models on idle, re-run
`lms load <model>` before indexing.

## ZCode registration

`~/.zcode/cli/config.json` (user scope, all workspaces):

```json
"rag": {
  "command": "C:\\dev\\ia\\rag\\target\\release\\rag.exe",
  "args": ["serve"],
  "env": { "RAG_EMBED_MODEL": "qwen.qwen3-vl-embedding-2b" }
}
```

Restart the session (or Settings → MCP) after changing the file.

## Performance notes (measured)

- **Search-side** (the part this tool controls): SIMD cosine kernel + mmap +
  rayon. The `throughput_report` test runs 50k × 768-dim rows — tens of
  milliseconds end-to-end on the host machine.
- **Embedding-side** (LM Studio-controlled): on the author's setup the local
  embedding engine processes ~600 tokens/s per request and **degrades ~18×
  under concurrent request contention** (benchmarked: 1 input 33 ms idle, but
  15.7 s when 3 batch requests were in flight). Hence the YAML defaults
  `batch_size: 128`, `concurrency: 1` — one big request at a time beats
  parallel small ones. Your machine may differ; `rag index --dry-run` plus a
  timed run will tell.
- Incremental re-index only embeds changed files; a killed run keeps committed
  windows (checkpoint every ~25k vectors).

## Configuration file

`~/.zcode/ia/config.yaml` (override path with `RAG_CONFIG`). `rag config`
prints the effective values; `rag config --init` writes a commented default.
Precedence: CLI flag > env var (`RAG_*`) > YAML > built-in default.

```yaml
provider: lmstudio                # lmstudio | ollama | openai (OpenAI-compatible)
# base_url: http://localhost:11434/v1
# api_key: ""                     # cloud providers
embed_model: qwen.qwen3-vl-embedding-2b
index:
  batch_size: 128
  concurrency: 1
search:
  k: 5
  p1_damp: 0.9
```

## On GPU

Embeddings already run on the GPU through LM Studio/Ollama — that is where
the GPU pays off. The cosine search is CPU-side on purpose: at this tool's
scale (tens of thousands of chunks) the SIMD + rayon kernel answers in
single-digit milliseconds, while a GPU path (wgpu/CUDA) would add VRAM
contention with the embedding/LLM models, transfer overhead, and two more
dependency trees. Revisit if indexes grow past a few million vectors (f16/int8
quantization is the cheaper first step, then GPU).

## Development

```bash
cargo test                       # unit tests incl. SIMD-vs-scalar and throughput report
cargo build --release            # target/release/rag.exe
cargo run -- index --dry-run .   # audit what would be indexed, per tier
```

The MCP surface is smoke-tested by spawning the release binary as a stdio
server (see the Node harness `smoke-rust.mjs` in the zcode-rag reference
project) against either a mock embeddings server or the real LM Studio.
