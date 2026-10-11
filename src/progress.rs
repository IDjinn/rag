use crate::config::Tier;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Progress reporting for `rag index`: two live bars on an interactive
/// terminal (files + chunks, current file and tokens), or plain one-line
/// stderr reports when stderr is not a terminal or `--no-progress` was given.
/// Silent mode (background auto-refresh) emits nothing at all.
pub struct Progress {
    live: Option<Live>,
    silent: bool,
    plain: Mutex<Plain>,
}

struct Live {
    multi: MultiProgress,
    /// Created by `begin()`; headers may print before any bar exists.
    yarn: Mutex<Option<Yarn>>,
    tokens: AtomicU64,
}

/// Yarn-style UI: one persistent header spinner (`➤ ⠙ rag index …`) plus one
/// spinner line per file being embedded, finished with `✔ <file>` when done.
struct Yarn {
    header: ProgressBar,
    tier: String,
    total_files: u64,
    total_chunks: u64,
    files_done: u64,
    active: Option<ProgressBar>,
    active_name: String,
    /// Anchor for the live tok/s and chunks/s rates.
    started: Instant,
}

struct Plain {
    phase: &'static str,
    total: u64,
    tokens: u64,
    batches: u32,
    started: Instant,
}

/// Human-readable rate, e.g. `1.2k tok/s`; `0 tok/s` before any time passes.
fn rate(n: u64, elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64();
    if secs < 0.05 {
        return "—".to_string();
    }
    let per_sec = n as f64 / secs;
    if per_sec >= 1000.0 {
        format!("{:.1}k", per_sec / 1000.0)
    } else {
        format!("{per_sec:.0}")
    }
}

const TICKS: &str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

fn files_style() -> ProgressStyle {
    ProgressStyle::with_template("➤ {spinner:.green} {msg}")
        .expect("header spinner template must parse")
        .tick_chars(TICKS)
}

fn file_style() -> ProgressStyle {
    ProgressStyle::with_template("  {spinner:.green} {msg}")
        .expect("file spinner template must parse")
        .tick_chars(TICKS)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Progress {
    /// `force_plain` is the `--no-progress` flag; a non-terminal stderr
    /// downgrades to plain on its own.
    pub fn new(force_plain: bool) -> Self {
        let live = if force_plain || !stderr_is_term() { None } else { Some(Live::new(MultiProgress::new())) };
        Progress {
            live,
            silent: false,
            plain: Mutex::new(Plain { phase: "", total: 0, tokens: 0, batches: 0, started: Instant::now() }),
        }
    }

    /// Fully quiet: background runs report through a single summary line
    /// instead of per-batch progress.
    pub fn silent() -> Self {
        Progress {
            live: None,
            silent: true,
            plain: Mutex::new(Plain { phase: "", total: 0, tokens: 0, batches: 0, started: Instant::now() }),
        }
    }

    /// Live mode wired to a caller-provided terminal (tests only).
    #[cfg(test)]
    fn new_for_term(multi: MultiProgress) -> Self {
        Progress {
            live: Some(Live::new(multi)),
            silent: false,
            plain: Mutex::new(Plain { phase: "", total: 0, tokens: 0, batches: 0, started: Instant::now() }),
        }
    }

    /// One-time status line. Only call before `begin()`: in live mode it goes
    /// straight to stderr, above where the bars will appear.
    pub fn header(&self, line: &str) {
        if self.silent {
            return;
        }
        match &self.live {
            Some(_) => eprintln!("{line}"),
            None => eprintln!("[rag] {line}"),
        }
    }

    /// Reveal the yarn-style UI with the run totals (`total_chunks` is an
    /// estimate).
    pub fn begin(&self, total_files: u64, total_chunks: u64) {
        {
            let mut p = lock(&self.plain);
            p.total = total_chunks;
            p.started = Instant::now();
        }
        if let Some(l) = &self.live {
            let header = l.multi.add(ProgressBar::new_spinner());
            header.set_style(files_style());
            header.enable_steady_tick(Duration::from_millis(120));
            header.set_message("rag index");
            *lock(&l.yarn) = Some(Yarn {
                header,
                tier: String::new(),
                total_files,
                total_chunks,
                files_done: 0,
                active: None,
                active_name: String::new(),
                started: Instant::now(),
            });
        }
    }

    /// Label the current tier and, in plain mode, report its share.
    pub fn phase(&self, tier: Tier, files: u64, est_chunks: u64) {
        lock(&self.plain).phase = tier.as_str();
        match &self.live {
            Some(l) => {
                if let Some(y) = lock(&l.yarn).as_mut() {
                    y.tier = tier.as_str().to_string();
                    y.header.set_message(format!("rag index — embedding {}", tier.as_str()));
                }
            }
            None => {
                if !self.silent {
                    eprintln!("[rag] embedding {}: {files} files (~{est_chunks} chunks)", tier.as_str())
                }
            }
        }
    }

    pub fn file_done(&self, n: u64) {
        if let Some(l) = &self.live {
            if let Some(y) = lock(&l.yarn).as_mut() {
                y.files_done += n;
                y.refresh_header();
            }
        }
    }

    /// One embedding batch finished; `done` is the cumulative chunk count.
    pub fn batch(&self, done: u64, batch_tokens: u64, current: &str) {
        let print = {
            let mut p = lock(&self.plain);
            p.tokens += batch_tokens;
            p.batches += 1;
            p.batches % 10 == 0
        };
        match &self.live {
            Some(l) => {
                let tokens = l.tokens.fetch_add(batch_tokens, Ordering::Relaxed) + batch_tokens;
                if let Some(y) = lock(&l.yarn).as_mut() {
                    let elapsed = y.started.elapsed();
                    y.refresh_header_full(done, tokens, elapsed);
                    if !current.is_empty() {
                        if current != y.active_name {
                            if let Some(prev) = y.active.take() {
                                prev.finish_with_message(format!("✔ {}", y.active_name));
                                y.files_done += 1;
                            }
                            let bar = l.multi.add(ProgressBar::new_spinner());
                            bar.set_style(file_style());
                            bar.enable_steady_tick(Duration::from_millis(120));
                            bar.set_message(current.to_string());
                            y.active = Some(bar);
                            y.active_name = current.to_string();
                        }
                        if let Some(bar) = y.active.as_ref() {
                            bar.set_message(format!("{current} — {} tok/s", rate(tokens, elapsed)));
                        }
                    }
                }
            }
            None => {
                if print && !self.silent {
                    let p = lock(&self.plain);
                    eprintln!(
                        "[rag] {} {}/{} chunks (~{} tokens, {} tok/s, {}/s chunks)",
                        p.phase,
                        done,
                        p.total,
                        crate::ops::fmt_tokens(p.tokens),
                        rate(p.tokens, p.started.elapsed()),
                        rate(done, p.started.elapsed()),
                    );
                }
            }
        }
    }

    /// A whole window of files finished (plain mode: print the running total).
    pub fn window_done(&self, done: u64) {
        if self.live.is_none() && !self.silent {
            let p = lock(&self.plain);
            eprintln!(
                "[rag] {} {}/{} chunks (~{} tokens, {} tok/s, {}/s chunks)",
                p.phase,
                done,
                p.total,
                crate::ops::fmt_tokens(p.tokens),
                rate(p.tokens, p.started.elapsed()),
                rate(done, p.started.elapsed()),
            );
        }
    }

    /// Clear the live UI; the final report follows on stdout.
    pub fn finish(&self) {
        if let Some(l) = &self.live {
            if let Some(mut y) = lock(&l.yarn).take() {
                if let Some(bar) = y.active.take() {
                    bar.finish_with_message(format!("✔ {}", y.active_name));
                }
                let elapsed = y.started.elapsed().as_secs_f64();
                y.header.finish_with_message(format!(
                    "✔ rag index — {} files, {} chunks in {:.1}s",
                    y.files_done, y.total_chunks, elapsed
                ));
            }
        }
    }
}

impl Yarn {
    fn refresh_header(&self) {
        self.header.set_message(format!(
            "rag index{} — {}/{} files",
            self.tier_suffix(),
            self.files_done,
            self.total_files
        ));
    }

    fn refresh_header_full(&self, chunks_done: u64, tokens: u64, elapsed: Duration) {
        self.header.set_message(format!(
            "rag index{} — {}/{} files · {}/{} chunks · {} tokens · {} tok/s · {}/s chunks",
            self.tier_suffix(),
            self.files_done,
            self.total_files,
            chunks_done,
            self.total_chunks,
            crate::ops::fmt_tokens(tokens),
            rate(tokens, elapsed),
            rate(chunks_done, elapsed),
        ));
    }

    fn tier_suffix(&self) -> String {
        if self.tier.is_empty() { String::new() } else { format!(" [{}]", self.tier) }
    }
}

impl Live {
    fn new(multi: MultiProgress) -> Self {
        Live { multi, yarn: Mutex::new(None), tokens: AtomicU64::new(0) }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        if let Some(l) = self.live.take() {
            if let Some(mut y) = l.yarn.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner()) {
                if let Some(bar) = y.active.take() {
                    bar.finish_with_message(format!("✔ {}", y.active_name));
                }
                y.header.finish_and_clear();
            }
        }
    }
}

/// Whether stderr is an interactive terminal (msys/cygwin ptys included).
fn stderr_is_term() -> bool {
    let probe = ProgressBar::new_spinner();
    probe.set_draw_target(ProgressDrawTarget::stderr());
    !probe.is_hidden()
}

/// The file containing chunk row `done`, given `(first_row, rel)` pairs in
/// ascending row order; empty slice or before the first row yields "".
pub(crate) fn current_file<'a>(rows: &[(usize, &'a str)], done: usize) -> &'a str {
    let idx = rows.partition_point(|(r, _)| *r <= done);
    if idx == 0 { "" } else { rows[idx - 1].1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indicatif::TermLike;
    use std::io;
    use std::sync::Arc;

    #[test]
    fn templates_parse() {
        // Invalid templates only surface at first draw otherwise.
        files_style();
        file_style();
    }

    #[test]
    fn rate_formats() {
        assert_eq!(rate(100, Duration::ZERO), "—");
        assert_eq!(rate(40, Duration::from_secs(2)), "20");
        assert_eq!(rate(2400, Duration::from_secs(2)), "1.2k");
    }

    #[test]
    fn silent_mode_exercises_all_paths() {
        // No assertion on output (stderr); the guarantee is structural — every
        // print site returns before eprintln! when silent is set.
        let prog = Progress::silent();
        prog.header("Indexing something");
        prog.begin(3, 12);
        prog.phase(Tier::P0, 2, 8);
        prog.batch(4, 100, "src/a.rs");
        prog.file_done(1);
        prog.window_done(4);
        prog.finish();
    }

    #[test]
    fn current_file_lookup() {
        let rows: Vec<(usize, &str)> = vec![(0, "a.rs"), (3, "b.rs"), (9, "c.rs")];
        assert_eq!(current_file(&rows, 0), "a.rs");
        assert_eq!(current_file(&rows, 2), "a.rs");
        assert_eq!(current_file(&rows, 3), "b.rs");
        assert_eq!(current_file(&rows, 11), "c.rs");
        assert_eq!(current_file(&[], 5), "");
    }

    /// Accumulates everything indicatif would paint on a terminal.
    #[derive(Debug)]
    struct FakeTerm {
        buf: Arc<Mutex<String>>,
    }

    impl TermLike for FakeTerm {
        fn width(&self) -> u16 {
            120
        }
        fn move_cursor_up(&self, _n: usize) -> io::Result<()> {
            Ok(())
        }
        fn move_cursor_down(&self, _n: usize) -> io::Result<()> {
            Ok(())
        }
        fn move_cursor_right(&self, _n: usize) -> io::Result<()> {
            Ok(())
        }
        fn move_cursor_left(&self, _n: usize) -> io::Result<()> {
            Ok(())
        }
        fn write_line(&self, s: &str) -> io::Result<()> {
            let mut buf = lock(&self.buf);
            buf.push_str(s);
            buf.push('\n');
            Ok(())
        }
        fn write_str(&self, s: &str) -> io::Result<()> {
            lock(&self.buf).push_str(s);
            Ok(())
        }
        fn clear_line(&self) -> io::Result<()> {
            Ok(())
        }
        fn flush(&self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn live_rendering_shows_status() {
        let buf = Arc::new(Mutex::new(String::new()));
        let target = ProgressDrawTarget::term_like(Box::new(FakeTerm { buf: buf.clone() }));
        let prog = Progress::new_for_term(MultiProgress::with_draw_target(target));

        prog.begin(10, 40);
        prog.phase(Tier::P0, 6, 24);
        prog.file_done(1);
        // The header spinner redraws on its 120ms steady ticker, so give it a beat.
        std::thread::sleep(Duration::from_millis(300));
        prog.batch(3, 120, "src/main.rs");
        prog.finish();

        let out = lock(&buf).clone();
        assert!(out.contains("rag index"), "header spinner missing in: {out:?}");
        assert!(out.contains("P0"), "tier label missing in: {out:?}");
        assert!(out.contains("3/40 chunks"), "chunk progress missing in: {out:?}");
        assert!(out.contains("src/main.rs"), "current file missing in: {out:?}");
        assert!(out.contains("tok/s"), "embedding rate missing in: {out:?}");
    }
}
