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
use std::collections::BTreeMap;
use std::path::PathBuf;
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
// rag_index
// ---------------------------------------------------------------------------

pub async fn index(
    path: Option<&str>,
    force: bool,
    dry_run: bool,
    batch: Option<usize>,
    concurrency: Option<usize>,
    no_ignore: bool,
    no_progress: bool,
) -> Result<String> {
    let batch = batch.unwrap_or(config::settings().index.batch_size);
    let concurrency = concurrency.unwrap_or(config::settings().index.concurrency);
    let started = Instant::now();
    let root = resolve_root(path);
    if !root.is_dir() {
        bail!("path is not a directory: {}", root.display());
    }
    let root_str = root.to_string_lossy().replace('\\', "/");
    let all_files = files::walk(&root, !no_ignore);
    if all_files.is_empty() {
        return Ok(format!("No indexable files found under `{}`.", root_str));
    }

    if dry_run {
        return dry_run_report(&root_str, &all_files, batch, concurrency);
    }

    let client = LmClient::new()?;
    let model = client.resolve_embed_model().await?;
    let store = Store::for_root(&root);
    let old = store.load().filter(|_| !force);
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
    let old_dim = old.as_ref().map_or(0, |d| d.dim);
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

    let prog = Progress::new(no_progress);
    prog.header(&format!(
        "Indexing {root_str} · {} · model `{model}` · {} file(s){}",
        config::provider_name(),
        all_files.len(),
        if force { " · force rebuild" } else { "" },
    ));

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
    store.save(&data, &vectors)?;
    prog.finish();

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

    let wall = started.elapsed().as_millis() as u64;
    let p0 = data.files.values().filter(|e| e.tier == Tier::P0).count();
    let p1 = data.files.len() - p0;
    Ok(format!(
        "## rag index complete\n\n\
         - **Root:** {root_str}\n\
         - **Provider:** {} — **Model:** {model} (dim {})\n\
         - **Files in index:** {} (P0: {p0}, P1: {p1}) — {} embedded, {} unchanged, {removed} removed\n\
         - **Chunks embedded:** {} (~{} tokens, {} request(s), {:.1}s)\n\
         - **Total vectors:** {} | index size: {:.0} KB\n\
         - **Wall time:** {:.1}s\n\n\
         Indexing is manual by design. Re-run `rag index` after meaningful changes — \
         unchanged files are copied over without embeddings.",
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
    ))
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
    let Some(data) = store.load() else {
        return Ok(format!(
            "No index found for `{}`. Run `rag index` first (indexing is manual by default).",
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

    let Some(mmap) = store.mmap_vectors() else {
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
            let base = entry.vec_offset * dim;
            (0..entry.chunk_count).map(move |c| {
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
        },
    );
    stats::save(&store.dir, &s);

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
    let store = Store::for_root(&resolve_root(path));
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

pub fn stats(root_path: Option<&str>, reset: bool) -> Result<String> {
    let root = resolve_root(root_path);
    let store = Store::for_root(&root);
    let mut s = stats::load(&store.dir);
    if reset {
        s = RagStats::default();
        stats::save(&store.dir, &s);
    }
    let data = store.load();

    let hit_rate = if s.searches > 0 { s.search_hits * 100 / s.searches } else { 0 };
    let useful_rate = if s.judged_chunks > 0 { Some(s.useful_chunks * 100 / s.judged_chunks) } else { None };
    let embed_per_search = if s.searches > 0 { s.embed_tokens / s.searches } else { s.embed_tokens };
    let ratio = if s.context_tokens_delivered > 0 {
        s.full_file_tokens_avoided as f64 / s.context_tokens_delivered as f64
    } else {
        0.0
    };
    let disk_kb = store.disk_bytes() as f64 / 1024.0;

    let mut out = String::new();
    out.push_str(&format!("## rag_stats\n\nRoot: `{}`\n\n", root.display()));

    out.push_str("### Index\n\n");
    match &data {
        Some(d) => {
            let p0 = d.files.values().filter(|e| e.tier == Tier::P0).count();
            out.push_str(&format!(
                "- Files: {} (P0: {p0}, P1: {}) | chunks: {} | dim: {} | provider: {} | model: `{}`\n",
                d.files.len(),
                d.files.len() - p0,
                d.vector_count,
                d.dim,
                d.provider,
                d.model
            ));
            out.push_str(&format!(
                "- Last indexed: {} | disk: {disk_kb:.0} KB\n\n",
                d.updated_at
            ));
        }
        None => out.push_str("- No index yet. Run `rag index` (manual by default).\n\n"),
    }

    out.push_str("### Cost (local: time + tokens embedded)\n\n");
    out.push_str(&format!(
        "- Index runs: {} (last wall time: {})\n",
        s.index_runs,
        s.last_index_ms.map_or("n/a".to_string(), |ms| format!("{:.1}s", ms as f64 / 1000.0))
    ));
    out.push_str(&format!(
        "- Embedded so far: {} chunks / ~{} tokens / {:.1}s / {} requests / {} errors\n",
        fmt_tokens(s.chunks_embedded),
        fmt_tokens(s.embed_tokens),
        s.embed_ms as f64 / 1000.0,
        s.embed_requests,
        s.embed_errors
    ));
    out.push_str(&format!("- Index disk: {disk_kb:.0} KB\n"));
    if let Some(price) = config::price_per_1m_tokens() {
        let usd = s.embed_tokens as f64 / 1e6 * price;
        out.push_str(&format!(
            "- Estimated embedding spend: ~${usd:.4} (at ${price:.4} / 1M tokens via RAG_PRICE_PER_1M_TOKENS)\n"
        ));
    }
    out.push('\n');

    out.push_str("### Utility\n\n");
    out.push_str(&format!(
        "- Searches: {} (errors: {}) | avg results: {:.1} | hit rate (top score ≥ {}): {hit_rate}%\n",
        s.searches,
        s.search_errors,
        if s.searches > 0 { s.search_results as f64 / s.searches as f64 } else { 0.0 },
        config::settings().search.hit_threshold
    ));
    out.push_str(&format!(
        "- Context delivered: ~{} tokens | full-file reads avoided: ~{} tokens (×{ratio:.1})\n",
        fmt_tokens(s.context_tokens_delivered),
        fmt_tokens(s.full_file_tokens_avoided)
    ));
    match useful_rate {
        Some(rate) => out.push_str(&format!(
            "- Jev-judged usefulness: {rate}% ({}/{} chunks)\n",
            s.useful_chunks, s.judged_chunks
        )),
        None => out.push_str(
            "- Jev-judged usefulness: not measured yet (use rag_search with judge:true, then rag_feedback)\n",
        ),
    }
    out.push_str(&format!(
        "- Index cost per search so far: ~{} embedded tokens (amortized)\n\n",
        fmt_tokens(embed_per_search)
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
    out.push_str(&format!("**Verdict:** {verdict}\n\n"));

    if !s.history.is_empty() {
        out.push_str("### Recent queries\n\n| when | query | results | top score | judged |\n|---|---|---|---|---|\n");
        for h in s.history.iter().take(10) {
            let judged = h
                .judged_count
                .filter(|c| *c > 0)
                .map_or("—".to_string(), |c| format!("{}/{}", h.useful_count.unwrap_or(0), c));
            let when = h.ts.get(5..16).unwrap_or("").replace('T', " ");
            let query = h.query.chars().take(40).collect::<String>().replace('|', "\\|");
            out.push_str(&format!("| {when} | {query} | {} | {:.2} | {judged} |\n", h.results, h.top_score));
        }
        out.push('\n');
    }
    out.push_str("Use `rag stats --reset` (or rag_stats with reset:true) to zero the counters (keeps the index).");
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
        None => out.push_str("- **Index:** none yet — run `rag index` (manual by default).\n"),
    }
    Ok(out)
}

pub fn clear(path: Option<&str>) -> Result<String> {
    let store = Store::for_root(&resolve_root(path));
    Ok(if store.remove() {
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
