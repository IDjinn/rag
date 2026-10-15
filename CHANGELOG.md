# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## 0.6.0 — 2026-10-15

### Added

- **`rag stats --global`** — aggregate cost/utility report across every
  indexed root under the data dir: counters summed, query histories merged
  (newest first), and a per-root table of files, chunks, disk size and last
  index time. `--global --reset` zeroes every root's counters in one go;
  the flag conflicts with `PATH`, and the `rag_stats` MCP tool gained the
  matching `global` argument.

## 0.5.0 — 2026-10-12

### Added

- **Move/rename/copy awareness** — files whose content hash matches any
  entry in the existing index now adopt that entry's vectors instead of
  being re-embedded. Renaming or moving a folder drops from N embedding
  requests to zero (cost becomes file reads + hashing + one vectors
  rewrite), copied files reuse the source vectors, and the index report
  gained a "Reused by content" line splitting moved/renamed vs copied vs
  touched. `rag stats` tracks cumulative reuse (`files_reused`).

## 0.4.0 — 2026-10-11

### Added

- **Embedding throughput in the index UI** — live progress now shows
  tok/s and chunks/s (computed from the embedding wall clock), in both
  the interactive display and the plain one-line mode; the final
  `rag index` report gained a `{chunks}/s · {tok}/s` summary on the
  "Chunks embedded" line.

### Changed

- **Yarn-style live progress** — the two-bar index display was replaced
  by a yarn-like interface: one persistent header spinner (`➤ ⠙ rag index …`)
  with files/chunks/tokens/throughput, one spinner line per file being
  embedded (path + tok/s), and each finished file checked off as
  `✔ <file>`.

## 0.3.0 — 2026-10-10

### Added

- **Tokens saved per search** — `rag stats` now reports net savings
  (full-file reads avoided minus context delivered) as a lifetime total and
  a per-search average. Each search records its own token counts
  (`delivered_tokens`/`saved_tokens` in the query history; older stats.json
  files load unchanged and simply have no per-search values yet).
- **Savings-over-time chart** — `rag stats` renders a horizontal bar chart
  of the last 20 searches (oldest → newest) showing tokens saved per
  search; a `·` marks searches with no net saving, and the chart appears
  once searches start recording token counts.

### Changed

- **Simplified `rag stats` output** — the Index/Cost/Utility sections
  collapsed into four summary lines (index shape, searches + usefulness,
  saved tokens, index cost) followed by the verdict, the savings chart and
  the recent-queries table. Error counts and estimated USD spend only show
  when nonzero / when a price is configured.

## 0.2.2 — 2026-10-10

### Changed

- **Bare `rag` prints help** instead of silently starting the MCP stdio
  server (which just sat in the terminal reading stdin). Launch the server
  explicitly with `rag serve` — the documented ZCode registration already
  passes it.
- **Terminal-friendly CLI reports** — when stdout is a terminal, command
  reports (stats, status, index, search, …) render as aligned plain text:
  underlined headings, fixed-width tables with the markdown separator rows
  dropped, inline `**`/backtick markers stripped. Piped/redirected output and
  the MCP tools keep the markdown format.

### Fixed

- `rag stats` in a terminal no longer prints raw markdown tables
  (`| when | query | …`), including the Recent queries table shown after
  judged searches.

## 0.2.1 — 2026-10-10

### Changed

- **Hook now covers subagent delegation.** Real-world sessions showed the
  SessionStart note alone doesn't stop exploration subagents from running
  in parallel with (or instead of) the index: subagents start fresh without
  the hook context, and Explore agents have no MCP tool access at all.
  - The `SessionStart` message now defines the flow explicitly —
    `rag_search` first, paste relevant file:line hits into subagent prompts,
    fall back to Grep only for exact-string/regex sweeps and to subagents
    only for what the index doesn't cover.
  - New `PreToolUse` handler (register with matcher `Agent|Task`): emits a
    short reminder at the exact moment the model spawns a subagent. Still
    strictly advisory — nothing is blocked.

## 0.2.0 — 2026-10-10

### Added

- **Automatic incremental refresh** — `rag serve` now keeps indexed roots up
  to date: a stat-only diff every `auto_refresh_interval_secs` (default 10s)
  re-embeds changed/new files, drops removed ones, and checkpoints every few
  thousand chunks. Idle ticks do no network I/O and write nothing. Configure
  with `index.auto_refresh` / `index.auto_refresh_interval_secs` or
  `RAG_AUTO_REFRESH` / `RAG_AUTO_REFRESH_SECS`. The first index per root
  stays manual.
- **ZCode SessionStart hook** — new `rag hook` subcommand for ZCode's hook
  system: on session start (startup, resume, clear, post-compact) it checks
  whether the session's working directory has an index and emits an
  `additionalContext` note telling the agent to prefer `rag_search` over
  Grep/Glob for semantic code location. Unindexed directories get silence;
  the hook never touches the network and always exits 0.
- **Configurable tier classification** — `config.yaml` gained `index.p0_exts`,
  `index.p1_exts`, `index.p1_names`, `index.dependency_files` and custom glob
  patterns `index.p0_globs` / `index.p1_globs` (case-insensitive, matched
  against the root-relative path and the file name). A list present in the
  file replaces the built-in default entirely; `rag config --init` now writes
  all default lists out so every entry is visible and editable.
- `strip = true` in the release profile for smaller distributable binaries.
