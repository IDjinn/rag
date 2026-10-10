# AGENTS.md — rag

Single Rust binary providing local RAG for ZCode: manual file indexing into a
local vector store (OpenAI-compatible embeddings — LM Studio by default,
Ollama/cloud via `RAG_PROVIDER`), SIMD cosine search, cost/utility stats, and
an MCP stdio server (`rag serve`, also the default command). See `README.md`
for the full user-facing documentation.

## Layout

- `src/main.rs` — clap CLI entry point
- `src/mcp.rs` — MCP stdio server (rmcp): `rag_status/index/search/feedback/stats/clear`
- `src/hook.rs` — ZCode hooks (`rag hook`): SessionStart index announcement +
  PreToolUse(Agent) delegation reminder, both emitting `additionalContext`
  when the cwd is indexed; silent + exit 0 otherwise
- `src/lmstudio.rs` — embeddings client (`/v1/embeddings`, provider-agnostic)
- `src/store.rs` — index persistence + memory-mapped vectors
- `src/simd.rs` — `wide` f32x8 cosine kernel + scalar fallback + parity tests
- `src/chunk.rs`, `src/files.rs` — chunking; file discovery, P0/P1 tiers, ignore rules
- `src/config.rs` — config precedence: CLI flag > `RAG_*` env > `~/.zcode/ia/config.yaml` > default
- `src/stats.rs`, `src/ops.rs`, `src/progress.rs` — stats/feedback, commands, progress bars
- `testdata/` — test fixtures (committed)
- `.cargo/config.toml` — sets `target-cpu=native` on x86_64 (host-tuned dev builds)

Index/stat data lives under `~/.zcode/ia` (`RAG_DATA_DIR`), never in the repo.

## Commands

```bash
rtk cargo test                                  # unit tests (incl. SIMD-vs-scalar parity)
rtk cargo test throughput_report -- --nocapture # search throughput report
rtk cargo build --release                       # target/release/rag.exe
cargo run -- index --dry-run .                  # audit what would be indexed, per tier
```

Real indexing/search needs a running embeddings backend (LM Studio default
`http://localhost:1234/v1`, with an embedding-classified model loaded).

## Gotchas

- **`target-cpu=native` is dev-only.** `.cargo/config.toml` tunes builds for the
  host CPU; binaries built with it can crash with illegal instructions on other
  machines. Any distributable/release build must override it (e.g.
  `CARGO_ENCODED_RUSTFLAGS` with `-C target-cpu=x86-64-v2` or baseline).
- **Keep embedding `concurrency: 1`.** The local LM Studio embedding engine
  degrades ~18x under concurrent requests; one big batch beats parallel small
  ones (see README performance notes).
- Windows host with Git Bash: Linux/macOS targets cannot be cross-compiled
  locally — use the GitHub Actions release workflow.
- Release profile uses `panic = "abort"` + fat LTO: release builds are slow to
  compile; that is expected.

## Git & release policy (user-mandated)

The repo is published on GitHub (account `IDjinn`, `gh` CLI is authenticated).
Keep the repo description and topics in sync with the project's purpose, and
keep `.gitignore` covering `/target` and `/.mimosa` (local tool state — never
commit build outputs, scan history, or session snapshots). `Cargo.lock` is
committed (binary crate).

**After every completed feature with tests passing, cut a release:**

1. Update `CHANGELOG.md` (create it if missing) with what changed.
2. Bump `version` in `Cargo.toml` — semver: minor for features, patch for fixes.
3. Commit, then tag `vX.Y.Z` and push the tag.
4. Publish a GitHub Release titled `vX.Y.Z` whose notes carry the changelog
   section ("what changed"), with binaries for the 3 main architectures:
   - `x86_64-unknown-linux-gnu` → linux-amd64 archive
   - `x86_64-pc-windows-msvc` → windows-amd64 zip (`rag.exe`)
   - `aarch64-apple-darwin` → macos-arm64 archive
5. Distributable builds must override `target-cpu=native` (see Gotchas) and
   strip/mmap-friendly release profile (`--release`).

The intended mechanism is a `.github/workflows/release.yml` triggered by the
tag push that builds each target on its native runner, overrides the native
CPU flag, and attaches the archives to the release. It does not exist yet —
creating it is part of the first release.
