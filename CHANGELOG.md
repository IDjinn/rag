# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/) and the project
adheres to [Semantic Versioning](https://semver.org/).

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
