use crate::chunk;
use crate::config::{self, Tier};
use crate::files::{self, DiscoveredFile};
use crate::lmstudio::{self, LmClient};
use crate::progress::{current_file, Progress};
use crate::simd;
use crate::stats::{self, QueryRecord, RagStats};
use crate::store::{self, vectors_as_slice, FileEntry, IndexData, Store};
use anyhow::{bail, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, MutexGuard, OnceLock};
use std::time::Instant;

pub fn resolve_root(path: Option<&str>) -> PathBuf {
    path.map(PathBuf::from).unwrap_or_else(|| {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    })
}

pub(crate) fn fmt_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn est_tokens(chars: usize) -> u64 {
    (chars as u64).div_ceil(4)
}

/// Rough per-file chunk count for progress totals: size over the effective
/// chunk step (target minus overlap).
fn est_chunks(size: u64) -> u64 {
    let idx = &config::settings().index;
    let step = idx.chunk_target_chars.saturating_sub(idx.chunk_overlap_chars).max(1) as u64;
    (size / step).max(1)
}

// ---------------------------------------------------------------------------
// Per-root locking
// ---------------------------------------------------------------------------

/// Locks guarding one index directory. `op` serializes whole index runs
/// (manual tool calls vs the background auto-refresh) and is async because it
/// is held across embedding requests. `store` guards the on-disk
/// index.json/vectors.bin pair and stats.json against torn reads/writes, and
/// is only ever held across short synchronous I/O — never across an await.
struct RootLocks {
    op: tokio::sync::Mutex<()>,
    store: std::sync::Mutex<()>,
}

fn root_locks(root: &Path) -> Arc<RootLocks> {
    static LOCKS: OnceLock<std::sync::Mutex<HashMap<PathBuf, Arc<RootLocks>>>> = OnceLock::new();
    // Keyed by index directory so roots that map to the same index share locks.
    let key = config::index_dir_for(root);
    let map = LOCKS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .entry(key)
        .or_insert_with(|| {
            Arc::new(RootLocks {
                op: tokio::sync::Mutex::new(()),
                store: std::sync::Mutex::new(()),
            })
        })
        .clone()
}

fn lock_store(locks: &RootLocks) -> MutexGuard<'_, ()> {
    locks.store.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Whether a root already has an index on disk. The initial index stays
/// manual; auto-refresh only maintains existing indexes.
pub fn index_exists(root: &Path) -> bool {
    Store::for_root(root).load().is_some()
}

/// Whether an incremental run has work to do: any indexed file changed on
/// disk (mtime/size, mirroring the pass-1 skip condition), any new file, or
/// any file gone since the last index. Pure stat-level check — no reads.
fn has_changes(old: &IndexData, all_files: &[DiscoveredFile], old_dim: usize) -> bool {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::with_capacity(all_files.len());
    for f in all_files {
        seen.insert(f.rel.as_str());
        let unchanged = old
            .files
            .get(&f.rel)
            .is_some_and(|e| files::same_state((e.mtime_ms, e.size), (f.mtime_ms, f.size)) && old_dim > 0);
        if !unchanged {
            return true;
        }
    }
    old.files.keys().any(|k| !seen.contains(k.as_str()))
}

/// Chunk rows of a file that actually fit in the vectors blob. Guards the
/// scoring loop against a mismatched index.json/vectors.bin pair (possible
/// with cross-process writes); with panic=abort any slice panic is fatal.
fn visible_chunks(base: usize, chunk_count: usize, dim: usize, rows_len: usize) -> usize {
    if dim == 0 {
        return 0;
    }
    let avail = rows_len.saturating_sub(base) / dim;
    chunk_count.min(avail)
}

// ---------------------------------------------------------------------------
// rag_index
// ---------------------------------------------------------------------------

/// Options for one `rag index` run.
pub struct IndexOpts {
    pub path: Option<String>,
    /// Rebuild from scratch, re-embedding everything.
    pub force: bool,
    /// Report per-tier estimates without embedding or writing.
    pub dry_run: bool,
    pub batch: Option<usize>,
    pub concurrency: Option<usize>,
    /// Ignore .gitignore/.ignore rules.
    pub no_ignore: bool,
    /// Plain non-interactive progress output.
    pub no_progress: bool,
    /// Fully silent progress (background auto-refresh ticks).
    pub quiet: bool,
}

impl Default for IndexOpts {
    fn default() -> Self {
        Self {
            path: None,
            force: false,
            dry_run: false,
            batch: None,
            concurrency: None,
            no_ignore: false,
            no_progress: false,
            quiet: false,
        }
    }
}

/// What a completed run did; the auto-refresh loop logs a summary line only
/// when `changed` is set.
pub struct IndexOutcome {
    pub report: String,
    pub changed: bool,
    pub embedded_files: u64,
    /// Files whose mtime/size changed but content matched — re-embedded nothing.
    pub reused_files: u64,
    pub removed: u64,
}

pub async fn index(opts: IndexOpts) -> Result<String> {
    index_run(opts).await.map(|o| o.report)
}

pub(crate) async fn index_run(opts: IndexOpts) -> Result<IndexOutcome> {
    let IndexOpts { path, force, dry_run, batch, concurrency, no_ignore, no_progress, quiet } = opts;
    let batch = batch.unwrap_or(config::settings().index.batch_size);
    let concurrency = concurrency.unwrap_or(config::settings().index.concurrency);
    let started = Instant::now();
    let root = resolve_root(path.as_deref());
    if !root.is_dir() {
        bail!("path is not a directory: {}", root.display());
    }
    let root_str = root.to_string_lossy().replace('\\', "/");
    let all_files = files::walk(&root, !no_ignore);
    if all_files.is_empty() {
        return Ok(IndexOutcome {
            report: format!("No indexable files found under `{root_str}`."),
            changed: false,
            embedded_files: 0,
            reused_files: 0,
            removed: 0,
        });
    }

    if dry_run {
        return Ok(IndexOutcome {
            report: dry_run_report(&root_str, &all_files, batch, concurrency)?,
            changed: false,
            embedded_files: 0,
            reused_files: 0,
            removed: 0,
        });
    }

    let prog = if quiet { Progress::silent() } else { Progress::new(no_progress) };

    let store = Store::for_root(&root);
    // One index run per root at a time: a manual tool call and the background
    // auto-refresh both rebuild from the previous index's vectors, so
    // overlapping runs would embed from stale state.
    let locks = root_locks(&root);
    let _op_guard = locks.op.lock().await;

    let old = store.load().filter(|_| !force);
    let old_dim = old.as_ref().map_or(0, |d| d.dim);

    // Up-to-date fast path before any network I/O: idle auto-refresh ticks
    // must not touch LM Studio or rewrite vectors.bin.
    if old.as_ref().is_some_and(|o| !has_changes(o, &all_files, old_dim)) {
        prog.finish();
        return Ok(IndexOutcome {
            report: format!(
                "## rag index — up to date\n\n\
                 - **Root:** {root_str}\n\
                 - Nothing changed since the last index ({} file(s) checked). No embeddings, nothing written.",
                all_files.len()
            ),
            changed: false,
            embedded_files: 0,
            reused_files: 0,
            removed: 0,
        });
    }

    let client = LmClient::new()?;
    let model = client.resolve_embed_model().await?;
    prog.header(&format!(
        "Indexing {root_str} · {} · model `{model}` · {} file(s){}",
        config::provider_name(),
        all_files.len(),
        if force { " · force rebuild" } else { "" },
    ));
    if let Some(old_data) = &old {
        let different_model = old_data.model != model || old_data.provider != config::provider_name();
        if different_model && old_data.vector_count > 0 {
            bail!(
                "The existing index was built with {} model \"{}\" but the current one is \"{}\". Vectors from different models are not comparable — re-run with --force to rebuild.",
                old_data.provider,
                old_data.model,
                model
            );
        }
    }
    let old_mmap = old.as_ref().and_then(|_| store.mmap_vectors());
    let removed = old
        .as_ref()
        .map(|o| {
            o.files
                .keys()
                .filter(|k| !all_files.iter().any(|f| &f.rel == *k))
                .count() as u64
        })
        .unwrap_or(0);

    let mut data = IndexData {
        version: 1,
        root: root_str.clone(),
        model: model.clone(),
        provider: config::provider_name().to_string(),
        dim: old.as_ref().map_or(0, |d| d.dim),
        files: BTreeMap::new(),
        vector_count: 0,
        updated_at: String::new(),
    };
    let mut vectors: Vec<f32> = Vec::new();
    let mut counters = EmbedCounters::default();
    let mut rows_since_checkpoint = 0usize;
    let mut last_index_ms = 0u64;

    // Pass 1 (both tiers upfront): unchanged files are copied straight over,
    // so the embedding totals below are exact file counts and close estimates
    // for chunks.
    let mut pending: [Vec<&DiscoveredFile>; 2] = [Vec::new(), Vec::new()];
    for (ti, tier) in [Tier::P0, Tier::P1].into_iter().enumerate() {
        for f in all_files.iter().filter(|f| f.tier == tier) {
            if let Some(entry) = old.as_ref().and_then(|o| o.files.get(&f.rel)) {
                if files::same_state((entry.mtime_ms, entry.size), (f.mtime_ms, f.size)) && old_dim > 0 {
                    if let Some(mmap) = &old_mmap {
                        let rows = vectors_as_slice(mmap);
                        let from = entry.vec_offset * old_dim;
                        let to = from + entry.chunk_count * old_dim;
                        if to <= rows.len() {
                            vectors.extend_from_slice(&rows[from..to]);
                        }
                    }
                    let mut e = entry.clone();
                    e.vec_offset = data.vector_count;
                    data.vector_count += e.chunk_count;
                    data.files.insert(f.rel.clone(), e);
                    counters.skipped += 1;
                    continue;
                }
            }
            pending[ti].push(f);
        }
    }

    let total_pending: u64 = pending.iter().map(|v| v.len() as u64).sum();
    let total_est: u64 = pending.iter().flat_map(|v| v.iter()).map(|f| est_chunks(f.size)).sum();
    prog.header(&format!(
        "{} unchanged · {} to embed (~{total_est} chunks) · {removed} removed",
        counters.skipped, total_pending,
    ));
    if total_pending > 0 {
        prog.begin(total_pending, total_est);
    }

    // Pass 2: windows of files per tier, embedded with bounded concurrency.
    let window_target = (batch.max(1) * concurrency.max(1)).max(64) * 2;
    let mut chunk_base = 0u64;
    for (ti, tier) in [Tier::P0, Tier::P1].into_iter().enumerate() {
        if pending[ti].is_empty() {
            continue;
        }
        prog.phase(
            tier,
            pending[ti].len() as u64,
            pending[ti].iter().map(|f| est_chunks(f.size)).sum(),
        );
        let mut window: Vec<&DiscoveredFile> = Vec::new();
        let mut window_est = 0usize;
        for f in pending[ti].drain(..) {
            window.push(f);
            window_est += (f.size as usize / config::settings().index.chunk_target_chars).max(1);
            if window_est >= window_target {
                let (ms, staged) = embed_and_commit(
                    &window, &mut data, &mut vectors, old.as_ref(), old_mmap.as_ref(), old_dim,
                    &client, &model, batch, concurrency, tier, &prog, chunk_base, &mut counters,
                )
                .await?;
                last_index_ms += ms;
                chunk_base += staged as u64;
                prog.file_done(window.len() as u64);
                prog.window_done(chunk_base);
                rows_since_checkpoint += window_est * data.dim.max(1);
                if rows_since_checkpoint > 25_000 {
                    let _store_guard = lock_store(&locks);
                    store.save(&data, &vectors)?; // checkpoint: crash keeps committed work
                    rows_since_checkpoint = 0;
                }
                window.clear();
                window_est = 0;
            }
        }
        if !window.is_empty() {
            let (ms, staged) = embed_and_commit(
                &window, &mut data, &mut vectors, old.as_ref(), old_mmap.as_ref(), old_dim,
                &client, &model, batch, concurrency, tier, &prog, chunk_base, &mut counters,
            )
            .await?;
            last_index_ms += ms;
            chunk_base += staged as u64;
            prog.file_done(window.len() as u64);
            prog.window_done(chunk_base);
        }
    }

    data.vector_count = if data.dim > 0 { vectors.len() / data.dim } else { 0 };
    data.updated_at = store::now_iso();
    {
        let _store_guard = lock_store(&locks);
        store.save(&data, &vectors)?;
    }
    prog.finish();

    {
        let _store_guard = lock_store(&locks);
        let mut s = stats::load(&store.dir);
        s.index_runs += 1;
        s.last_indexed_at = Some(data.updated_at.clone());
        s.last_index_ms = Some(last_index_ms);
        s.files_indexed += counters.embedded_files;
        s.files_skipped += counters.skipped;
        s.files_removed += removed;
        s.chunks_embedded += counters.embedded_chunks;
        s.embed_requests += counters.requests;
        s.embed_tokens += counters.tokens;
        s.embed_ms += counters.embed_ms;
        s.p0_files = data.files.values().filter(|e| e.tier == Tier::P0).count() as u64;
        s.p1_files = data.files.values().filter(|e| e.tier == Tier::P1).count() as u64;
        stats::save(&store.dir, &s);
    }

    let wall = started.elapsed().as_millis() as u64;
    let embed_rate = if counters.embed_ms > 0 {
        format!(
            "{:.0} chunks/s · {:.0} tok/s",
            counters.embedded_chunks as f64 / (counters.embed_ms as f64 / 1000.0),
            counters.tokens as f64 / (counters.embed_ms as f64 / 1000.0)
        )
    } else {
        "n/a".to_string()
    };
    let p0 = data.files.values().filter(|e| e.tier == Tier::P0).count();
    let p1 = data.files.len() - p0;
    let report = format!(
        "## rag index complete\n\n\
         - **Root:** {root_str}\n\
         - **Provider:** {} — **Model:** {model} (dim {})\n\
         - **Files in index:** {} (P0: {p0}, P1: {p1}) — {} embedded, {} unchanged, {removed} removed\n\
         - **Chunks embedded:** {} (~{} tokens, {} request(s), {:.1}s, {embed_rate})\n\
         - **Total vectors:** {} | index size: {:.0} KB\n\
         - **Wall time:** {:.1}s\n\n\
         This root now refreshes automatically every {}s while a `rag serve` session is alive; \
         re-run manually only to force a full rebuild.",
        config::provider_name(),
        data.dim,
        data.files.len(),
        counters.embedded_files,
        counters.skipped,
        counters.embedded_chunks,
        fmt_tokens(counters.tokens),
        counters.requests,
        counters.embed_ms as f64 / 1000.0,
        data.vector_count,
        store.disk_bytes() as f64 / 1024.0,
        wall as f64 / 1000.0,
        config::auto_refresh_interval_secs(),
    );
    Ok(IndexOutcome {
        report,
        changed: true,
        embedded_files: counters.embedded_files,
        reused_files: counters.reused,
        removed,
    })
}

#[derive(Default)]
struct EmbedCounters {
    skipped: u64,
    reused: u64,
    embedded_files: u64,
    embedded_chunks: u64,
    tokens: u64,
    requests: u64,
    embed_ms: u64,
}

/// Read, chunk, embed and commit one window of files. Returns the embedding
/// wall time and the number of chunks staged (for progress bookkeeping).
#[allow(clippy::too_many_arguments)]
async fn embed_and_commit(
    window: &[&DiscoveredFile],
    data: &mut IndexData,
    vectors: &mut Vec<f32>,
    old: Option<&IndexData>,
    old_mmap: Option<&Mmap>,
    old_dim: usize,
    client: &LmClient,
    model: &str,
    batch: usize,
    concurrency: usize,
    tier: Tier,
    prog: &Progress,
    chunk_base: u64,
    counters: &mut EmbedCounters,
) -> Result<(u64, usize)> {
    struct Job<'a> {
        rel: &'a str,
        size: u64,
        mtime_ms: i64,
        hash: String,
        refs: Vec<store::ChunkRef>,
        first_row: usize,
    }
    let mut jobs: Vec<Job<'_>> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut staged = 0usize;

    for f in window {
        let Some(text) = files::read_text(&f.abs) else {
            continue; // binary or unreadable
        };
        let hash = store::sha1_hex(&text);
        if let Some(entry) = old.and_then(|o| o.files.get(&f.rel)) {
            if entry.hash == hash {
                if old_dim > 0 {
                    if let Some(mmap) = old_mmap {
                        let rows = vectors_as_slice(mmap);
                        let from = entry.vec_offset * old_dim;
                        let to = from + entry.chunk_count * old_dim;
                        if to <= rows.len() {
                            vectors.extend_from_slice(&rows[from..to]);
                        }
                    }
                }
                let mut e = entry.clone();
                e.vec_offset = data.vector_count;
                // Refresh the stat-level state: otherwise a touched-but-unchanged
                // file stays "pending" in every later run (mtime churn).
                e.mtime_ms = f.mtime_ms;
                e.size = f.size;
                data.vector_count += e.chunk_count;
                data.files.insert(f.rel.to_string(), e);
                counters.reused += 1;
                continue;
            }
        }
        let chunks = chunk::chunk_default(&text);
        if chunks.is_empty() {
            continue;
        }
        jobs.push(Job {
            rel: f.rel.as_str(),
            size: f.size,
            mtime_ms: f.mtime_ms,
            hash,
            refs: chunks.iter().map(|c| store::ChunkRef { start: c.start, end: c.end }).collect(),
            first_row: staged,
        });
        staged += chunks.len();
        texts.extend(chunks.iter().map(|c| c.text.clone()));
    }
    if texts.is_empty() {
        return Ok((0, 0));
    }

    // Map the chunk-row frontier back to the file being embedded so the
    // progress UI can show it.
    let rows: Vec<(usize, &str)> = jobs.iter().map(|j| (j.first_row, j.rel)).collect();
    let mut on_batch = |done: usize, tokens: u64| {
        prog.batch(chunk_base + done as u64, tokens, current_file(&rows, done));
    };
    let mut run = lmstudio::embed_all(client, model, &texts, batch, concurrency, &mut on_batch).await?;
    if data.dim == 0 {
        data.dim = run.dim;
    } else if data.dim != run.dim {
        bail!(
            "embedding dimension changed ({} -> {}); re-run with --force to rebuild",
            data.dim,
            run.dim
        );
    }
    if data.dim == 0 {
        bail!("embedding dimension is zero");
    }

    run.vectors.par_chunks_mut(data.dim).for_each(simd::l2_normalize);
    vectors.extend_from_slice(&run.vectors);

    for job in &jobs {
        data.files.insert(
            job.rel.to_string(),
            FileEntry {
                hash: job.hash.clone(),
                size: job.size,
                mtime_ms: job.mtime_ms,
                chunk_count: job.refs.len(),
                vec_offset: data.vector_count + job.first_row,
                chunks: job.refs.clone(),
                tier,
            },
        );
    }
    data.vector_count += staged;
    counters.embedded_files += jobs.len() as u64;
    counters.embedded_chunks += texts.len() as u64;
    counters.tokens += run.tokens;
    counters.requests += run.requests;
    counters.embed_ms += run.ms;
    Ok((run.ms, staged))
}

fn dry_run_report(root_str: &str, all_files: &[DiscoveredFile], batch: usize, concurrency: usize) -> Result<String> {
    let mut lines = vec![
        "## rag index --dry-run".to_string(),
        String::new(),
        format!("Root: `{root_str}`"),
        String::new(),
        "| tier | files | MB | chunks | ~tokens |".to_string(),
        "|---|---|---|---|---|".to_string(),
    ];
    let mut total_chunks = 0u64;
    for tier in [Tier::P0, Tier::P1] {
        let selected: Vec<&DiscoveredFile> = all_files.iter().filter(|f| f.tier == tier).collect();
        let (count, bytes, chunks, tokens) = selected
            .par_iter()
            .fold(
                || (0u64, 0u64, 0u64, 0u64),
                |acc, f| match files::read_text(&f.abs) {
                    Some(text) => (
                        acc.0 + 1,
                        acc.1 + f.size,
                        acc.2 + chunk::chunk_default(&text).len() as u64,
                        acc.3 + est_tokens(text.len()),
                    ),
                    None => acc,
                },
            )
            .reduce(
                || (0u64, 0u64, 0u64, 0u64),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
            );
        total_chunks += chunks;
        lines.push(format!(
            "| {} | {count} | {:.1} | {chunks} | {} |",
            tier.as_str(),
            bytes as f64 / 1e6,
            fmt_tokens(tokens)
        ));
    }
    lines.push(String::new());
    lines.push(format!(
        "Total: {} files, ~{total_chunks} chunks to embed (batch {batch}, concurrency {concurrency}). \
         P0 is embedded first; P1 follows. Nothing was written.",
        all_files.len()
    ));
    Ok(lines.join("\n"))
}

// ---------------------------------------------------------------------------
// rag_search
// ---------------------------------------------------------------------------

pub async fn search(query: &str, k: Option<usize>, path: Option<&str>, judge: bool, no_boost: bool) -> Result<String> {
    let k = k.unwrap_or(config::settings().search.k).max(1);
    let started = Instant::now();
    let root = resolve_root(path);
    let store = Store::for_root(&root);
    let locks = root_locks(&root);
    // Load the index.json + vectors.bin pair under the store lock: the two
    // renames in Store::save are not atomic as a pair, so an unlocked reader
    // could see new metadata over old vectors. The mmap keeps the old
    // snapshot valid after the lock is released.
    let (data, mmap) = {
        let _store_guard = lock_store(&locks);
        let data = store.load();
        let mmap = data
            .as_ref()
            .filter(|d| d.dim > 0 && d.vector_count > 0)
            .and_then(|_| store.mmap_vectors());
        (data, mmap)
    };
    let Some(data) = data else {
        return Ok(format!(
            "No index found for `{}`. Run `rag index` once to build it (afterwards changes refresh automatically while a session is alive).",
            root.display()
        ));
    };
    let client = LmClient::new()?;
    let model = client.resolve_embed_model().await?;
    if model != data.model || config::provider_name() != data.provider {
        return Ok(format!(
            "Index was built with {} model `{}` but the current one is `{}`. Run `rag index --force` to rebuild before searching.",
            data.provider, data.model, model
        ));
    }
    if data.dim == 0 || data.vector_count == 0 {
        return Ok("Index is empty. Run `rag index` to fill it.".to_string());
    }
    let (mut q, _) = client.embed_batch(&model, &[query.to_string()]).await?;
    q.truncate(data.dim);
    simd::l2_normalize(&mut q);

    let Some(mmap) = mmap else {
        return Ok("Index vectors are missing; re-run `rag index`.".to_string());
    };
    let rows = vectors_as_slice(&mmap);
    let dim = data.dim;

    let file_list: Vec<(&String, &FileEntry)> = data.files.iter().collect();
    let q_ref: &[f32] = &q;
    let mut scored: Vec<(f32, u32, u32)> = file_list
        .par_iter()
        .enumerate()
        .flat_map_iter(|(fi, (_, entry))| {
            let damp = if entry.tier == Tier::P1 && !no_boost { config::settings().search.p1_damp } else { 1.0 };
            let base = entry.vec_offset.saturating_mul(dim);
            let count = visible_chunks(base, entry.chunk_count, dim, rows.len());
            (0..count).map(move |c| {
                let off = base + c * dim;
                (simd::dot(&rows[off..off + dim], q_ref) * damp, fi as u32, c as u32)
            })
        })
        .collect();
    scored.retain(|(s, _, _)| s.is_finite());
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(k.max(1));
    let elapsed = started.elapsed().as_millis() as u64;

    let mut out = String::new();
    let mut delivered_chars = 0usize;
    let mut full_file_chars = 0usize;
    let mut seen_files: Vec<&String> = Vec::new();
    let mut judge_items: Vec<serde_json::Value> = Vec::new();

    if scored.is_empty() {
        out.push_str(&format!("No results for \"{query}\".\n"));
    }
    for (rank, (score, fi, ci)) in scored.iter().enumerate() {
        let (rel, entry) = file_list[*fi as usize];
        let id = format!("r{}", rank + 1);
        let mut abs = root.clone();
        for part in rel.split('/') {
            abs.push(part);
        }
        let (snippet, stale, file_chars) = match std::fs::read_to_string(&abs) {
            Ok(content) => {
                let lines: Vec<&str> = content.lines().collect();
                let start = entry.chunks[*ci as usize].start.saturating_sub(1);
                let end = (entry.chunks[*ci as usize].end).min(lines.len());
                let mut snip: String = lines[start..end].join("\n");
                let snippet_max = config::settings().search.snippet_max_chars;
                if snip.chars().count() > snippet_max {
                    snip = snip.chars().take(snippet_max).collect::<String>() + " …";
                }
                let current = std::fs::metadata(&abs)
                    .ok()
                    .map(|md| {
                        let mtime = md
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_millis() as i64)
                            .unwrap_or(0);
                        (mtime, md.len())
                    })
                    .unwrap_or((0, 0));
                let changed = !files::same_state((entry.mtime_ms, entry.size), current);
                (snip, changed, content.len())
            }
            Err(_) => ("_(file missing since indexing)_".to_string(), true, 0),
        };
        delivered_chars += snippet.len();
        if !seen_files.contains(&rel) {
            seen_files.push(rel);
            full_file_chars += file_chars;
        }
        let tier_badge = entry.tier.as_str();
        out.push_str(&format!(
            "### {id}. [{tier_badge}] {rel}:{}-{} — score {:.3}{}\n\n```\n{snippet}\n```\n\n",
            entry.chunks[*ci as usize].start,
            entry.chunks[*ci as usize].end,
            score,
            if stale { " ⚠️ changed since indexing" } else { "" }
        ));
        judge_items.push(serde_json::json!({
            "id": id,
            "path": abs.to_string_lossy(),
            "text": snippet.chars().take(500).collect::<String>(),
        }));
    }

    let delivered = est_tokens(delivered_chars);
    let full = est_tokens(full_file_chars);
    let top_score = scored.first().map_or(0.0, |(s, _, _)| *s);

    {
        let _store_guard = lock_store(&locks);
        let mut s = stats::load(&store.dir);
        s.searches += 1;
        s.search_results += scored.len() as u64;
        if top_score >= config::settings().search.hit_threshold {
            s.search_hits += 1;
        }
        s.search_ms += elapsed;
        s.context_tokens_delivered += delivered;
        s.full_file_tokens_avoided += full;
        stats::push_history(
            &mut s,
            QueryRecord {
                ts: store::now_iso(),
                query: query.to_string(),
                results: scored.len(),
                top_score,
                ms: elapsed,
                judged_count: Some(0),
                useful_count: Some(0),
                delivered_tokens: delivered,
                saved_tokens: full.saturating_sub(delivered),
            },
        );
        stats::save(&store.dir, &s);
    }

    out.push_str(&format!(
        "---\n\n~{} tokens added to context (reading the {} matched file(s) fully would cost ~{} tokens). Search took {elapsed} ms.\n",
        fmt_tokens(delivered),
        seen_files.len(),
        fmt_tokens(full)
    ));

    if judge && !scored.is_empty() {
        let payload = serde_json::json!({
            "items": judge_items,
            "questions": [{
                "id": "relevant",
                "type": "check",
                "question": format!("Is this chunk relevant for answering: \"{query}\"?"),
                "yes_means": "directly helps answer the query",
                "no_means": "not helpful for this query",
            }],
        });
        out.push_str(
            "\n### Jev relevance check (optional, cheap)\n\n\
             Run `jev_triage` with the payload below (item texts are truncated; pass only `path` \
             to let jev read whole files without them entering context), then record verdicts via \
             `rag_feedback`:\n\n```json\n",
        );
        out.push_str(&serde_json::to_string_pretty(&payload)?);
        out.push_str(
            "\n```\n\nThen call `rag_feedback` with `{ \"verdicts\": [{ \"id\": \"r1\", \"useful\": true }, …] }`.\n",
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// rag_feedback / rag_stats / rag_status / rag_clear
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct FeedbackVerdict {
    #[allow(dead_code)] // part of the public API shape (matches result ids r1..rn)
    pub id: String,
    pub useful: bool,
}

pub fn feedback(json_str: &str, path: Option<&str>) -> Result<String> {
    let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| anyhow::anyhow!("invalid feedback JSON: {e}"))?;
    let verdicts: Vec<FeedbackVerdict> =
        serde_json::from_value(parsed.get("verdicts").cloned().unwrap_or(serde_json::Value::Null))
            .map_err(|e| anyhow::anyhow!("invalid verdicts: {e}"))?;
    feedback_verdicts(verdicts, path)
}

pub fn feedback_verdicts(verdicts: Vec<FeedbackVerdict>, path: Option<&str>) -> Result<String> {
    if verdicts.is_empty() {
        return Ok("No verdicts provided.".to_string());
    }
    let root = resolve_root(path);
    let store = Store::for_root(&root);
    let locks = root_locks(&root);
    let _store_guard = lock_store(&locks);
    let mut s = stats::load(&store.dir);
    let Some(rec) = s.history.get_mut(0) else {
        return Ok("No recent search to attach verdicts to. Run rag_search first.".to_string());
    };
    let useful = verdicts.iter().filter(|v| v.useful).count() as u64;
    rec.judged_count = Some(verdicts.len() as u64);
    rec.useful_count = Some(useful);
    let query = rec.query.clone();
    s.judge_sessions += 1;
    s.judged_chunks += verdicts.len() as u64;
    s.useful_chunks += useful;
    stats::save(&store.dir, &s);
    let lifetime = if s.judged_chunks > 0 { s.useful_chunks * 100 / s.judged_chunks } else { 0 };
    Ok(format!(
        "Recorded {useful}/{} useful for \"{query}\". Lifetime jev-judged usefulness: {lifetime}% ({}/{} chunks).",
        verdicts.len(),
        s.useful_chunks,
        s.judged_chunks
    ))
}

/// Rows in the stats savings chart and the width of its longest bar.
const SAVINGS_CHART_ROWS: usize = 20;
const SAVINGS_BAR_WIDTH: usize = 24;

/// Horizontal bar chart of net tokens saved per search (full-file reads
/// avoided minus context delivered), oldest → newest, over the most recent
/// `rows` history records. `None` when nothing shows a saving — including
/// histories recorded before per-search token tracking existed.
fn savings_chart(history: &[QueryRecord], rows: usize) -> Option<String> {
    let recent: Vec<&QueryRecord> = history.iter().take(rows).collect();
    let max_saved = recent.iter().map(|h| h.saved_tokens).max()?;
    if max_saved == 0 {
        return None;
    }
    let mut out = String::from("```text\n");
    for h in recent.iter().rev() {
        let when = h.ts.get(5..16).unwrap_or("--").replace('T', " ");
        let bar_len =
            ((h.saved_tokens as f64 / max_saved as f64) * SAVINGS_BAR_WIDTH as f64).round() as usize;
        let bar = if bar_len == 0 { "·".to_string() } else { "█".repeat(bar_len) };
        out.push_str(&format!("{when} │{bar} {}\n", fmt_tokens(h.saved_tokens)));
    }
    out.push_str("```");
    Some(out)
}

pub fn stats(root_path: Option<&str>, reset: bool) -> Result<String> {
    let root = resolve_root(root_path);
    let store = Store::for_root(&root);
    let locks = root_locks(&root);
    let mut s = stats::load(&store.dir);
    if reset {
        let _store_guard = lock_store(&locks);
        s = RagStats::default();
        stats::save(&store.dir, &s);
    }
    let data = store.load();

    let hit_rate = if s.searches > 0 { s.search_hits * 100 / s.searches } else { 0 };
    let useful_rate = if s.judged_chunks > 0 { Some(s.useful_chunks * 100 / s.judged_chunks) } else { None };
    let saved_total = s.full_file_tokens_avoided.saturating_sub(s.context_tokens_delivered);
    let saved_per_search = if s.searches > 0 { saved_total / s.searches } else { 0 };
    let ratio = if s.context_tokens_delivered > 0 {
        s.full_file_tokens_avoided as f64 / s.context_tokens_delivered as f64
    } else {
        0.0
    };
    let disk_kb = store.disk_bytes() as f64 / 1024.0;

    let mut out = String::new();
    out.push_str("## rag_stats\n\n");

    match &data {
        Some(d) => {
            let p0 = d.files.values().filter(|e| e.tier == Tier::P0).count();
            out.push_str(&format!(
                "- **Index:** {} files (P0 {p0} · P1 {}) · {} chunks · dim {} · {disk_kb:.0} KB · model `{}` · last indexed {}\n",
                d.files.len(),
                d.files.len() - p0,
                d.vector_count,
                d.dim,
                d.model,
                d.updated_at
            ));
        }
        None => out.push_str("- **Index:** none yet — run `rag index` once.\n"),
    }

    let usefulness = match useful_rate {
        Some(rate) => format!("{rate}% ({}/{} chunks)", s.useful_chunks, s.judged_chunks),
        None => "not measured".to_string(),
    };
    let search_errors = if s.search_errors > 0 {
        format!(" · {} search errors", s.search_errors)
    } else {
        String::new()
    };
    out.push_str(&format!(
        "- **Searches:** {} · {hit_rate}% hits · usefulness {usefulness}{search_errors}\n",
        s.searches
    ));
    out.push_str(&format!(
        "- **Saved:** ~{}/search · ~{} total — ~{} delivered vs ~{} full-file reads (×{ratio:.1})\n",
        fmt_tokens(saved_per_search),
        fmt_tokens(saved_total),
        fmt_tokens(s.context_tokens_delivered),
        fmt_tokens(s.full_file_tokens_avoided)
    ));

    let amortized = if s.searches > 0 {
        format!(" · ~{} tokens/search amortized", fmt_tokens(s.embed_tokens / s.searches))
    } else {
        String::new()
    };
    let spend = config::price_per_1m_tokens()
        .map(|price| format!(" · ~${:.4} est. spend", s.embed_tokens as f64 / 1e6 * price))
        .unwrap_or_default();
    let embed_errors =
        if s.embed_errors > 0 { format!(" · {} embed errors", s.embed_errors) } else { String::new() };
    out.push_str(&format!(
        "- **Index cost:** {} run(s) · ~{} tokens embedded · {:.1}s{amortized}{spend}{embed_errors}\n",
        s.index_runs,
        fmt_tokens(s.embed_tokens),
        s.embed_ms as f64 / 1000.0,
    ));

    let verdict = if s.searches == 0 {
        "No searches yet — embedding cost is paid but utility is unproven."
    } else if useful_rate.is_some_and(|r| r < 40) {
        "Judged usefulness is low — the index may not be worth keeping for this root."
    } else if ratio >= 3.0 {
        "Paying off: search delivers a small fraction of the tokens full-file reads would cost."
    } else if ratio >= 1.0 {
        "Marginal — delivered context is comparable to just reading the matched files."
    } else {
        "Not paying off yet — few searches, or results nearly as big as the files."
    };
    out.push_str(&format!("\n**Verdict:** {verdict}\n"));

    if let Some(chart) = savings_chart(&s.history, SAVINGS_CHART_ROWS) {
        out.push_str(&format!(
            "\n### Savings per search (older → newer, last ≤{SAVINGS_CHART_ROWS})\n\n{chart}\n"
        ));
    } else if s.searches > 0 {
        out.push_str("\n_No per-search savings to chart yet._\n");
    }

    if !s.history.is_empty() {
        out.push_str("\n### Recent queries\n\n| when | query | results | top score | judged |\n|---|---|---|---|---|\n");
        for h in s.history.iter().take(10) {
            let judged = h
                .judged_count
                .filter(|c| *c > 0)
                .map_or("—".to_string(), |c| format!("{}/{}", h.useful_count.unwrap_or(0), c));
            let when = h.ts.get(5..16).unwrap_or("").replace('T', " ");
            let query = h.query.chars().take(40).collect::<String>().replace('|', "\\|");
            out.push_str(&format!("| {when} | {query} | {} | {:.2} | {judged} |\n", h.results, h.top_score));
        }
    }
    out.push_str("\nUse `rag stats --reset` (or rag_stats with reset:true) to zero the counters (keeps the index).");
    Ok(out)
}

pub async fn status(path: Option<&str>) -> Result<String> {
    let root = resolve_root(path);
    let store = Store::for_root(&root);
    let data = store.load();
    let mut out = String::new();
    out.push_str(&format!("## rag_status\n\n- **Root:** {}\n- **Data dir:** {}\n", root.display(), store.dir.display()));
    match LmClient::new().and_then(|c| Ok(c)) {
        Ok(client) => match client.resolve_embed_model().await {
            Ok(model) => out.push_str(&format!("- **LM Studio:** reachable, embedding model: `{model}`\n")),
            Err(e) => out.push_str(&format!("- **LM Studio:** {e:#}\n")),
        },
        Err(e) => out.push_str(&format!("- **LM Studio:** {e:#}\n")),
    }
    match &data {
        Some(d) => {
            let p0 = d.files.values().filter(|e| e.tier == Tier::P0).count();
            out.push_str(&format!(
                "- **Index:** {} files (P0: {p0}, P1: {}), {} chunks, dim {}, model `{}`\n",
                d.files.len(),
                d.files.len() - p0,
                d.vector_count,
                d.dim,
                d.model
            ));
            out.push_str(&format!("- **Last indexed:** {}\n", d.updated_at));
            out.push_str(&format!("- **Disk:** {:.0} KB\n", store.disk_bytes() as f64 / 1024.0));
        }
        None => out.push_str("- **Index:** none yet — run `rag index` once; afterwards changed files refresh automatically.\n"),
    }
    let refresh = if config::auto_refresh() {
        format!("on (every {}s)", config::auto_refresh_interval_secs())
    } else {
        "off".to_string()
    };
    out.push_str(&format!(
        "- **Auto-refresh:** {refresh} — the MCP server re-embeds changed files in watched roots (server cwd + roots indexed this session)\n"
    ));
    Ok(out)
}

pub async fn clear(path: Option<&str>) -> Result<String> {
    let root = resolve_root(path);
    let store = Store::for_root(&root);
    let locks = root_locks(&root);
    // Wait out any in-flight index run, then remove without a save racing us.
    let _op_guard = locks.op.lock().await;
    let removed = {
        let _store_guard = lock_store(&locks);
        store.remove()
    };
    Ok(if removed {
        format!("Index and stats for root deleted ({}).", store.dir.display())
    } else {
        format!("No index found ({}).", store.dir.display())
    })
}

// ---------------------------------------------------------------------------
// rag config
// ---------------------------------------------------------------------------

pub fn config_cmd(init: bool) -> Result<String> {
    let path = config::config_path();
    if init {
        if path.exists() {
            return Ok(format!("Config already exists: {}", path.display()));
        }
        config::write_default_config(&path)?;
        return Ok(format!("Wrote default config to {}", path.display()));
    }
    let s = config::settings();
    let mut out = format!(
        "config file: {} ({})\nprovider: {} | base_url: {} | api_key: {} | model: {}\n\n",
        path.display(),
        if path.exists() { "present" } else { "absent — defaults in effect" },
        config::provider().name(),
        config::base_url(),
        if config::api_key().is_some() { "set" } else { "not set" },
        config::embed_model_env().unwrap_or_else(|| "(auto-detect)".to_string()),
    );
    out.push_str(&serde_yaml::to_string(s)?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ChunkRef;

    fn disc(rel: &str, size: u64, mtime_ms: i64) -> DiscoveredFile {
        DiscoveredFile {
            abs: PathBuf::from(rel),
            rel: rel.to_string(),
            size,
            mtime_ms,
            tier: Tier::P0,
        }
    }

    fn entry(size: u64, mtime_ms: i64) -> FileEntry {
        FileEntry {
            hash: "h".into(),
            size,
            mtime_ms,
            chunk_count: 1,
            vec_offset: 0,
            chunks: vec![ChunkRef { start: 1, end: 2 }],
            tier: Tier::P0,
        }
    }

    fn index_with(files: &[(&str, FileEntry)]) -> IndexData {
        let mut data = IndexData::empty("root", "m");
        data.dim = 4;
        for (rel, e) in files {
            data.files.insert(rel.to_string(), e.clone());
        }
        data
    }

    #[test]
    fn has_changes_detects_diff_size_mtime_new_and_removed() {
        let old = index_with(&[("a.rs", entry(10, 100)), ("b.rs", entry(20, 200))]);

        // Identical state (within the 2ms mtime tolerance): no changes.
        let same = vec![disc("a.rs", 10, 100), disc("b.rs", 20, 201)];
        assert!(!has_changes(&old, &same, 4));

        // Size changed.
        let resized = vec![disc("a.rs", 11, 100), disc("b.rs", 20, 200)];
        assert!(has_changes(&old, &resized, 4));

        // mtime moved beyond tolerance.
        let touched = vec![disc("a.rs", 10, 150), disc("b.rs", 20, 200)];
        assert!(has_changes(&old, &touched, 4));

        // New file on disk.
        let added = vec![disc("a.rs", 10, 100), disc("b.rs", 20, 200), disc("c.rs", 5, 5)];
        assert!(has_changes(&old, &added, 4));

        // File removed from disk.
        let gone = vec![disc("a.rs", 10, 100)];
        assert!(has_changes(&old, &gone, 4));
    }

    #[test]
    fn has_changes_forces_work_when_dim_is_zero() {
        let mut old = index_with(&[("a.rs", entry(10, 100))]);
        old.dim = 0; // unusable vectors: everything must be re-embedded
        let same = vec![disc("a.rs", 10, 100)];
        assert!(has_changes(&old, &same, 0));
    }

    #[test]
    fn visible_chunks_clamps_to_blob() {
        // 8 f32 rows, dim 4 → 2 chunks fit at base 0.
        assert_eq!(visible_chunks(0, 3, 4, 8), 2);
        // Starting past the blob yields nothing.
        assert_eq!(visible_chunks(8, 2, 4, 8), 0);
        assert_eq!(visible_chunks(100, 2, 4, 8), 0);
        // Fully contained.
        assert_eq!(visible_chunks(4, 1, 4, 8), 1);
        // Guard rails.
        assert_eq!(visible_chunks(0, 2, 0, 8), 0);
    }

    #[test]
    fn root_locks_share_one_entry_per_root() {
        let a = root_locks(Path::new("C:/some/root"));
        let b = root_locks(Path::new("C:/some/root"));
        let c = root_locks(Path::new("C:/other/root"));
        assert!(Arc::ptr_eq(&a, &b));
        assert!(!Arc::ptr_eq(&a, &c));
        // The store lock must be usable (poison-free) and relockable.
        drop(lock_store(&a));
        drop(lock_store(&a));
    }

    fn query_record(ts: &str, delivered: u64, saved: u64) -> QueryRecord {
        QueryRecord {
            ts: ts.to_string(),
            query: "q".to_string(),
            results: 1,
            top_score: 0.5,
            ms: 1,
            judged_count: None,
            useful_count: None,
            delivered_tokens: delivered,
            saved_tokens: saved,
        }
    }

    #[test]
    fn savings_chart_orders_oldest_first_and_scales_to_max() {
        // Stored newest-first, as search() inserts at index 0.
        let history = vec![
            query_record("2026-10-10T09:30:00Z", 0, 2000),
            query_record("2026-10-09T15:10:00Z", 0, 1000),
            query_record("2026-10-09T14:02:00Z", 0, 0),
        ];
        let chart = savings_chart(&history, SAVINGS_CHART_ROWS).unwrap();
        let lines: Vec<&str> = chart.lines().collect();
        assert_eq!(lines.len(), 5); // fence + 3 rows + fence
        let rows = &lines[1..4];
        // Oldest first; a zero saving renders as a dot.
        assert!(rows[0].starts_with("10-09 14:02 │· 0"));
        // Half the max → half the bar; the max gets the full width.
        assert_eq!(rows[1].matches('█').count(), SAVINGS_BAR_WIDTH / 2);
        assert!(rows[2].starts_with("10-10 09:30 │"));
        assert_eq!(rows[2].matches('█').count(), SAVINGS_BAR_WIDTH);
    }

    #[test]
    fn savings_chart_keeps_the_newest_rows_only() {
        let history: Vec<QueryRecord> = (0..25)
            .rev()
            .map(|i| query_record(&format!("2026-10-09T00:{i:02}:00Z"), 0, 100))
            .collect(); // index 0 is the newest, as search() stores it
        let chart = savings_chart(&history, SAVINGS_CHART_ROWS).unwrap();
        assert_eq!(chart.lines().count(), SAVINGS_CHART_ROWS + 2); // fence + rows + fence
        assert!(chart.contains("00:05")); // oldest of the newest 20 (indices 5..=24)
        assert!(!chart.contains("00:04"));
    }

    #[test]
    fn savings_chart_is_none_without_savings() {
        let history: Vec<QueryRecord> = (0..3)
            .map(|i| query_record(&format!("2026-10-09T00:{i:02}:00Z"), 10, 0))
            .collect();
        assert!(savings_chart(&history, SAVINGS_CHART_ROWS).is_none());
        assert!(savings_chart(&[], SAVINGS_CHART_ROWS).is_none());
    }
}
