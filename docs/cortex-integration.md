# Cortex integration — future work in `rag`

Cortex (github.com/IDjinn/cortex) now shells out to the `rag` binary from its
.NET backend: it materializes project sources into a per-project workdir on the
server, runs `rag index` / `rag search` as a subprocess, and reads `index.json`
for status. This document lists the changes to `rag` that would make that
integration robust — and eventually let Cortex drop its workaround parsing.

The Cortex side pins an exact commit (`RAG_REF` in its backend Dockerfile), so
each item here lands independently and Cortex bumps the pin when it wants it.

## What Cortex depends on today (do not break silently)

- `rag index <root> --no-progress` — exit 0 on success, exit 1 with
  `rag error: …` on stderr on failure. `--no-progress` keeps stdout clean.
- `rag search <query> --k N <root>` — stdout is markdown: one header per hit
  (`### r1. [P0] rel/path.rs:12-40 — score 0.873`, em dash U+2014, optional
  `⚠️ changed since indexing` suffix) followed by a fenced code block with the
  snippet, then a `---` footer line. Snippets are read live from the files on
  disk, so the indexed root must stay on disk after indexing.
- Env contract per invocation: `RAG_DATA_DIR`, `RAG_PROVIDER`
  (`lmstudio|ollama|openai`), `RAG_BASE_URL` (OpenAI-compatible, `/v1`-suffixed),
  `RAG_API_KEY`, `RAG_EMBED_MODEL`.
- `index.json` v1 fields read directly: `provider`, `model`, `dim`,
  `vector_count`, `files` → per-file `chunks[] {start,end}` and `chunk_count`.
- Index dir keying: `<data_dir>/indexes/<basename>-<sha1(canonical path)[..5]>`
  (path lowercased on Windows, basename sanitized to `[A-Za-z0-9._-]`, max 40).

## P0 — machine-readable output

Cortex currently parses the search markdown with a tolerant parser (it handles
snippets that themselves contain fences by only closing a block when a header
or the `---` footer follows). This is the first thing to replace.

1. `rag search --json` → one JSON object on stdout:
   ```json
   {
     "query": "…",
     "root": "…",
     "results": [
       { "path": "src/main.rs", "start_line": 12, "end_line": 40,
         "score": 0.873, "tier": "P0", "stale": false, "content": "…" }
     ],
     "embed_ms": 12, "search_ms": 3, "total_ms": 15
   }
   ```
   `embed_ms`/`search_ms` let Cortex fill its `RagMeta` stage timings with real
   numbers (today it reports wall-clock of the whole subprocess as `search_ms`).
2. `rag status --json` → `{ "root", "data_dir", "index": { "provider", "model",
   "dim", "file_count", "chunk_count", "updated_at" } | null }` so Cortex stops
   reading `index.json` directly.
3. Optional: `rag index --json` → the report counters as JSON (`changed`,
   `embedded_files`, `reused_files`, `removed`, `vector_count`).

Keep the markdown output exactly as is when `--json` is absent.

## P0 — cross-process locking

Locks are in-process only (`RootLocks`); two concurrent CLI processes on the
same index dir can race on `index.json`/`vectors.bin`. Cortex serializes its
own invocations per workdir, but any second writer (a human running `rag` in
the container, a future `rag serve`) breaks that.

- Take an advisory file lock (e.g. `fs2`/`fd-lock`) on `<index_dir>/.lock` for
  the `op` critical section, and a short-held variant for store reads/writes.
- Non-blocking mode (`--wait-secs N`, default blocking) so hosts can decide
  between queueing and failing fast.

## P1 — Gemini embeddings provider

Cortex had to drop its Gemini indexing preset because `rag` only speaks
OpenAI-compatible `/v1/embeddings`. Add:

- `RAG_PROVIDER=gemini` — `POST …/models/{model}:batchEmbedContents`
  (requests of ≤100 texts, `taskType: RETRIEVAL_DOCUMENT` when indexing,
  `RETRIEVAL_QUERY` for the query embedding; the distinction matters for
  retrieval quality).
- API key via `RAG_API_KEY` (Gemini accepts `x-goog-api-key`).
- Default base `https://generativelanguage.googleapis.com/v1beta`.

Cortex will re-add its `Gemini/gemini-embedding-001` preset once this ships.

## P1 — structured MCP tool results

`rag serve` tools return plain text/markdown content blocks. Add
`structuredContent` (same JSON shapes as the `--json` flags above) to
`rag_search`/`rag_status`/`rag_index` results, keeping the text block for
humans. Cortex plans to talk MCP instead of spawning CLI processes later —
this is the prerequisite (note: MCP stdio also solves cross-process state by
being a single long-lived process per host).

## P2 — smaller conveniences

- `--data-dir <path>` CLI flag (today only `RAG_DATA_DIR` env) — nicer for
  multi-tenant hosts that spawn per scope.
- Optional caps enforced by rag itself: `--max-files N`, `--max-chunks N`
  (fail the run when exceeded). Cortex enforces the chunk cap by reading the
  index after the run; enforcing inside rag would fail before the last embed.
- `rag clear` returning JSON (`--json`) with whether anything was removed
  (Cortex deletes the index dir directly today to avoid a spawn).

## Release notes for Cortex bumps

Cortex's parser/env contract is listed above. When any of these change shape
(markdown headers, index.json fields, env names, exit-code semantics), call it
out in `CHANGELOG.md` under a **breaking** heading so the pin bump in Cortex is
deliberate.
