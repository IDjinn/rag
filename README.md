# rag (Rust)

Local RAG for ZCode — a single static binary that indexes repository files into
a local vector index and serves semantic search, with explicit cost/utility
statistics so you can tell whether it is paying off.

- **Multi-provider embeddings** — anything speaking the OpenAI-compatible
  `/v1/embeddings` protocol: **LM Studio** (default), **Ollama**, or **OpenAI**
  / other clouds via `RAG_PROVIDER` + `RAG_API_KEY`.
- **Automatic incremental refresh** — the first `rag index` per root is
  manual; afterwards the MCP server re-embeds changed/new files and drops
  removed ones by itself (stat-only diff every `auto_refresh_interval_secs`,
  default 10s; idle ticks do no network I/O and write nothing). Disable with
  `index.auto_refresh: false` or `RAG_AUTO_REFRESH=0`. Incremental runs only
  embed changed files, and checkpoints every few thousand chunks keep work
  across crashes.
- **Move/rename/copy awareness** — files matched by content hash (SHA-1)
  adopt their existing embeddings instead of being re-embedded: renaming or
  moving a folder costs zero embedding requests, and copied files reuse the
  source vectors. Only genuinely new or edited content reaches the
  embedding backend.
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
- **Statistics** — embedding tokens/time/requests, searches served, tokens
  saved per search (full-file reads avoided minus context delivered,
  including a savings-over-time bar chart), jev-judged usefulness rate, and
  a verdict line. With a cloud provider, set `RAG_PRICE_PER_1M_TOKENS` to
  get estimated USD spend.

## Usage

```bash
rag index [PATH] [--force] [--dry-run] [--batch-size N] [--concurrency N] [--no-ignore] [--no-progress]
rag search "query" [--k 5] [--judge] [--no-boost] [PATH]
rag stats [PATH] [--reset]
rag status [PATH]
rag clear [PATH]
rag feedback --json '{"verdicts":[{"id":"r1","useful":true}]}' [PATH]
rag serve        # MCP stdio server (must be launched explicitly — bare `rag` prints help)
rag hook         # ZCode hooks: SessionStart context + PreToolUse(Agent) reminder
rag config [--init]
```

Bare `rag` prints the help; the MCP server is launched as `rag serve`
explicitly (the ZCode registration below passes `serve`). CLI reports are
laid out for the terminal whenever stdout is a TTY — underlined headings,
aligned tables, no raw markdown; piped/redirected output keeps the markdown
format, and the MCP tools always speak markdown.

As an MCP server it exposes `rag_status`, `rag_index`, `rag_search`,
`rag_feedback`, `rag_stats`, `rag_clear`. `rag_search` with `judge:true`
returns a ready-to-run `jev_triage` payload; feed the verdicts back through
`rag_feedback` and `rag_stats` reports the judged usefulness rate.

## Auto-refresh

The first index of a root is manual (`rag index` or the `rag_index` tool).
After that, `rag serve` keeps it current on its own:

- **Watched roots** — the server's cwd (its initial index may come from an
  earlier session or the CLI) plus every root indexed through a `rag_index`
  tool call in the current session. Roots without an index are ignored: the
  initial index is never created automatically.
- **Each tick** — a stat-only walk (mtime/size, the same ignore rules as
  indexing) diffs the root against the index. Changed/new files are content-
  hashed and re-embedded only if the hash differs; removed files are dropped.
  Files whose hash matches an entry anywhere in the index — moved, renamed,
  copied, or merely touched — adopt that entry's vectors at no embedding
  cost. An idle tick costs one directory walk, zero network requests and
  writes nothing to disk. Searches never wait for a refresh (per-root locks
  keep the `index.json`/`vectors.bin` pair consistent).
- **Errors** — if the embedding server is down or unloaded, the loop logs one
  stderr line and backs off exponentially (up to 5 minutes), resetting on the
  first success. Search results still flag `⚠️ changed since indexing` in the
  meantime.
- **Limitations** — refresh lives with the session (no daemon once the server
  exits), and a `rag index` run from a separate process while the server is
  refreshing the same root is still last-writer-wins (no cross-process file
  lock). Config changes (interval, on/off) require a server restart.

## Providers (env)

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_PROVIDER` | `lmstudio` | `ollama` or `openai` (any OpenAI-compatible cloud) |
| `RAG_BASE_URL` (or legacy `RAG_LMSTUDIO_URL`) | per provider | `http://localhost:1234/v1` (LM Studio), `http://localhost:11434/v1` (Ollama), `https://api.openai.com/v1` |
| `RAG_EMBED_MODEL` | auto-detect (`*embed*` in loaded models) | Pin the embedding model |
| `RAG_API_KEY` | — | Bearer token for cloud providers |
| `RAG_PRICE_PER_1M_TOKENS` | — | USD price for the cost report |
| `RAG_DATA_DIR` | `~/.zcode/ia` | Index/stat storage root |
| `RAG_AUTO_REFRESH` | `1` | `0`/`false` disables background auto-refresh |
| `RAG_AUTO_REFRESH_SECS` | `10` | Auto-refresh scan interval (min 1s) |

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
  "command": "C:\\path\\to\\rag.exe",
  "args": ["serve"],
  "env": { "RAG_EMBED_MODEL": "qwen.qwen3-vl-embedding-2b" }
}
```

Point `command` at the built binary (e.g. `~/.cargo/bin/rag.exe` after
`cargo install --path .`, or `target/release/rag.exe` inside a checkout).
Restart the session (or Settings → MCP) after changing the file.

## ZCode hooks (SessionStart + Agent delegation)

`rag hook` turns ZCode's hook system into passive index awareness. It handles
two events, both strictly advisory (unindexed directories get silence; it
never touches the network and always exits 0):

- **`SessionStart`** (startup, resume, clear, post-compact): injects a short
  `additionalContext` note announcing the index and defining the flow —
  `rag_search` first for semantic code location, exploration subagents only
  as a fallback for what the index doesn't cover.
- **`PreToolUse` on `Agent`**: fires at the moment the model spawns a
  subagent and reminds it to run `rag_search` itself and paste the relevant
  file:line hits into the subagent prompt (subagents start without the hook
  context, and Explore agents have no MCP tool access).

Register both in `~/.zcode/cli/config.json` next to the MCP server (same
`command` path as above):

```json
"hooks": {
  "enabled": true,
  "events": {
    "SessionStart": [
      { "hooks": [
        { "type": "process", "command": "C:\\path\\to\\rag.exe",
          "args": ["hook"], "timeoutMs": 10000, "statusMessage": "Checking RAG index" }
      ] }
    ],
    "PreToolUse": [
      { "matcher": "Agent|Task",
        "hooks": [
          { "type": "process", "command": "C:\\path\\to\\rag.exe",
            "args": ["hook"], "timeoutMs": 10000, "statusMessage": "RAG: rag_search first" }
        ] }
    ]
  }
}
```

Manual check (mimics what ZCode pipes in; expect JSON on an indexed cwd and
silence otherwise):

```bash
echo '{"hook_event_name":"SessionStart","cwd":"/some/indexed/repo"}' | rag hook
echo '{"hook_event_name":"PreToolUse","tool_name":"Agent","cwd":"/some/indexed/repo"}' | rag hook
```

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
  windows (checkpoint every ~25k vectors), and an incremental run that finds
  nothing to do writes nothing at all (the auto-refresh idle path).

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
  auto_refresh: true              # MCP server re-embeds changed files automatically
  auto_refresh_interval_secs: 10  # how often watched roots are scanned
  p0_exts: ["rs", "ts", "..."]    # full tier lists — see `rag config --init`
  p1_exts: ["json", "yml", "..."]
  p1_names: ["dockerfile", "..."]
  dependency_files: ["cargo.lock", "..."]
  p0_globs: []                    # e.g. ["**/migrations/**", "*.proto"]
  p1_globs: []                    # e.g. ["**/*.csv", "**/fixtures/**"]
search:
  k: 5
  p1_damp: 0.9
```

The tier lists (`p0_exts`, `p1_exts`, `p1_names`, `dependency_files`,
`p0_globs`, `p1_globs`) control which files get indexed and at which
priority. A list present in the file **replaces** the built-in default
entirely (`rag config --init` writes the full defaults out so every entry is
visible and editable). Globs are matched case-insensitively against the
root-relative path and the bare file name — `*` does not cross `/`, use
`**/` for directories. Classification order: P1 names → extensions (P0, then
P1) → P0 globs → P1 globs.

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
