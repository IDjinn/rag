//! GitHub metadata for the index: issues, pull requests (with their commits,
//! reviews and inline review comments), discussions and the wiki of the root's
//! `origin` remote. Everything is materialized as markdown under the index
//! directory and embedded together with the root's files. Each item links to
//! the items closest to it (closing references, `#N` mentions, commits and
//! reviews of a PR), and `tree.md` gives the whole origin tree in one page.
//!
//! Incremental sync: a small signature file (`github-sigs.json`) keeps, per
//! document, its `updated_at`, its link sets and the relation set it was last
//! rendered with. A run lists issues and PRs (one cheap call per 100 items),
//! fetches comments, commits and reviews only for items whose signature moved,
//! and rewrites only the documents that changed, so unchanged content keeps its
//! file and its embeddings.
//!
//! Network access goes through the `gh` CLI (already authenticated for the
//! user) and `git clone` for the wiki. Sync runs only from explicit `rag index`
//! runs; background auto-refresh re-reads the cached files and never calls out.

use crate::config::{GithubSettings, Tier};
use crate::files::DiscoveredFile;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;
use std::time::UNIX_EPOCH;

/// Root-relative prefix of the materialized documents (`<index>/github-meta/`).
pub const DIR: &str = "github-meta";
/// Signature file, next to the index (not inside `DIR`, so it is never embedded).
const SIGS_FILE: &str = "github-sigs.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repo {
    pub owner: String,
    pub name: String,
}

impl Repo {
    fn slug(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }
}

/// Outcome of one sync, for the index report.
pub struct SyncReport {
    pub repo: String,
    pub issues: usize,
    pub pulls: usize,
    pub discussions: usize,
    pub wiki_pages: usize,
    /// Documents fetched and rendered again in this run.
    pub changed: usize,
    /// Documents whose content disappeared upstream.
    pub removed: usize,
    /// Pieces that failed (missing permissions, disabled features, no `gh`).
    pub warnings: Vec<String>,
}

/// The `origin` of `root` when it is a GitHub repository.
pub fn detect_repo(root: &Path) -> Option<Repo> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["remote", "get-url", "origin"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_remote(String::from_utf8_lossy(&out.stdout).trim())
}

/// Accepts `https://github.com/o/r(.git)`, `git@github.com:o/r.git` and similar.
pub fn parse_remote(url: &str) -> Option<Repo> {
    let at = url.find("github.com")?;
    let rest = url[at + "github.com".len()..].trim_start_matches([':', '/']);
    let rest = rest.trim_end_matches('/').trim_end_matches(".git");
    let (owner, name) = rest.split_once('/')?;
    let name = name.split('/').next()?;
    if owner.is_empty() || name.is_empty() {
        return None;
    }
    Some(Repo { owner: owner.to_string(), name: name.to_string() })
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// An issue or a pull request.
#[derive(Default)]
struct Thread {
    number: u64,
    is_pr: bool,
    title: String,
    body: String,
    state: String,
    author: String,
    created: String,
    closed: String,
    updated: String,
    url: String,
    labels: Vec<String>,
    /// Comments, reviews and commits were fetched in this run.
    detailed: bool,
    /// Conversation comments (issues and PRs alike).
    comments: Vec<Note>,
    /// PR only: reviews, with inline comments attached by review id.
    reviews: Vec<Review>,
    inline: Vec<Note>,
    /// PR only: commits in the PR, oldest first.
    commits: Vec<CommitRef>,
    /// Issues this PR closes (closing keywords in its text).
    closes: BTreeSet<u64>,
    /// Other items mentioned with `#N` in its text.
    refs: BTreeSet<u64>,
    /// Lines of the knowledge tree for this item (PR commits and reviews).
    tree: Vec<String>,
}

impl Thread {
    fn rel(&self) -> String {
        format!("{}/{}.md", if self.is_pr { "pulls" } else { "issues" }, self.number)
    }
}

struct Note {
    author: String,
    created: String,
    body: String,
    url: String,
    /// `path:line` for inline review comments, empty otherwise.
    location: String,
    commit: String,
    review_id: u64,
}

struct Review {
    id: u64,
    author: String,
    state: String,
    body: String,
    submitted: String,
    commit: String,
}

struct CommitRef {
    sha: String,
    title: String,
    author: String,
    date: String,
}

struct Discussion {
    number: u64,
    title: String,
    body: String,
    url: String,
    author: String,
    created: String,
    updated: String,
    category: String,
    comments: Vec<Note>,
}

/// What the last sync knew about one document. Enough to skip unchanged items
/// without fetching them, and to tell when their relations moved.
#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Debug)]
struct Sig {
    updated: String,
    title: String,
    state: String,
    closes: BTreeSet<u64>,
    refs: BTreeSet<u64>,
    tree: Vec<String>,
    /// Reverse relations (closed by / referenced by) at the last render.
    relations: String,
}

#[derive(Serialize, Deserialize, Default)]
struct Sigs {
    repo: String,
    /// Keyed by the document path inside `DIR`, e.g. `issues/12.md`.
    items: BTreeMap<String, Sig>,
}

fn load_sigs(index_dir: &Path, repo: &Repo) -> Sigs {
    let loaded: Option<Sigs> = std::fs::read_to_string(index_dir.join(SIGS_FILE))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok());
    match loaded {
        Some(sigs) if sigs.repo == repo.slug() => sigs,
        _ => Sigs::default(),
    }
}

fn save_sigs(index_dir: &Path, sigs: &Sigs) -> Result<()> {
    let text = serde_json::to_string_pretty(sigs)?;
    std::fs::write(index_dir.join(SIGS_FILE), text)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// gh / git access
// ---------------------------------------------------------------------------

fn gh_json(args: &[String]) -> Result<Value> {
    let out = Command::new("gh")
        .args(args)
        .output()
        .context("gh not found (install GitHub CLI and run `gh auth login`)")?;
    if !out.status.success() {
        bail!("gh {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}

/// Every page of a REST list endpoint (`path` without query string).
fn gh_pages(path: &str) -> Result<Vec<Value>> {
    let sep = if path.contains('?') { '&' } else { '?' };
    let mut all = Vec::new();
    let mut page = 1;
    loop {
        let endpoint = format!("{path}{sep}per_page=100&page={page}");
        let items = gh_json(&["api".to_string(), endpoint])?;
        let items = items.as_array().cloned().unwrap_or_default();
        let n = items.len();
        all.extend(items);
        if n < 100 {
            return Ok(all);
        }
        page += 1;
    }
}

fn s(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

fn login(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(|u| u.get("login"))
        .and_then(Value::as_str)
        .map_or_else(|| "@ghost".to_string(), |l| format!("@{l}"))
}

/// Leading `YYYY-MM-DD` of an ISO timestamp.
fn day(text: &str) -> String {
    text.get(..10).unwrap_or("").to_string()
}

/// Trailing number of a URL such as `.../issues/12` or `.../pulls/12`.
fn soft<T: Default>(result: Result<T>, what: &str, warnings: &mut Vec<String>) -> T {
    result.unwrap_or_else(|e| {
        warnings.push(format!("{what}: {e:#}"));
        T::default()
    })
}

// ---------------------------------------------------------------------------
// Fetch
// ---------------------------------------------------------------------------

fn issue_from(v: &Value) -> Thread {
    let labels = v
        .get("labels")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|l| l.get("name").and_then(Value::as_str)).map(str::to_string).collect())
        .unwrap_or_default();
    Thread {
        number: v.get("number").and_then(Value::as_u64).unwrap_or(0),
        is_pr: false,
        title: s(v, "title"),
        body: s(v, "body"),
        state: s(v, "state"),
        author: login(v, "user"),
        created: day(&s(v, "created_at")),
        closed: day(&s(v, "closed_at")),
        updated: s(v, "updated_at"),
        url: s(v, "html_url"),
        labels,
        ..Thread::default()
    }
}

fn pull_from(v: &Value) -> Thread {
    let merged = !s(v, "merged_at").is_empty();
    Thread {
        number: v.get("number").and_then(Value::as_u64).unwrap_or(0),
        is_pr: true,
        title: s(v, "title"),
        body: s(v, "body"),
        state: if merged { "merged".to_string() } else { s(v, "state") },
        author: login(v, "user"),
        created: day(&s(v, "created_at")),
        closed: day(&s(v, if merged { "merged_at" } else { "closed_at" })),
        updated: s(v, "updated_at"),
        url: s(v, "html_url"),
        ..Thread::default()
    }
}

fn note_from(v: &Value) -> Note {
    Note {
        author: login(v, "user"),
        created: day(&s(v, "created_at")),
        body: s(v, "body"),
        url: s(v, "html_url"),
        location: String::new(),
        commit: String::new(),
        review_id: 0,
    }
}

fn inline_from(v: &Value) -> Note {
    let line = v.get("line").or_else(|| v.get("original_line")).and_then(Value::as_u64).unwrap_or(0);
    Note {
        location: format!("{}:{line}", s(v, "path")),
        commit: short_sha(&s(v, "commit_id")),
        review_id: v.get("pull_request_review_id").and_then(Value::as_u64).unwrap_or(0),
        ..note_from(v)
    }
}

fn short_sha(sha: &str) -> String {
    sha.get(..7).unwrap_or(sha).to_string()
}

fn commit_from(v: &Value) -> CommitRef {
    let commit = v.get("commit");
    let message = commit.map(|c| s(c, "message")).unwrap_or_default();
    let name = commit.and_then(|c| c.get("author")).map(|a| s(a, "name")).unwrap_or_default();
    let date = commit.and_then(|c| c.get("author")).map(|a| day(&s(a, "date"))).unwrap_or_default();
    let author = v
        .get("author")
        .and_then(|a| a.get("login"))
        .and_then(Value::as_str)
        .map_or(name, |l| format!("@{l}"));
    CommitRef {
        sha: short_sha(&s(v, "sha")),
        title: message.lines().next().unwrap_or("").to_string(),
        author,
        date,
    }
}

/// Issues and PRs without their details: one listing per 100 items.
fn fetch_lists(repo: &Repo, cfg: &GithubSettings, warnings: &mut Vec<String>) -> BTreeMap<u64, Thread> {
    let base = format!("repos/{}/{}", repo.owner, repo.name);
    let mut threads: BTreeMap<u64, Thread> = BTreeMap::new();
    if cfg.issues {
        let items = soft(gh_pages(&format!("{base}/issues?state=all")), "issues", warnings);
        for v in items.iter().filter(|v| v.get("pull_request").is_none()) {
            let t = issue_from(v);
            threads.insert(t.number, t);
        }
    }
    if cfg.pulls {
        let items = soft(gh_pages(&format!("{base}/pulls?state=all")), "pulls", warnings);
        for v in &items {
            let t = pull_from(v);
            threads.insert(t.number, t);
        }
    }
    threads
}

/// Comments, inline review comments, reviews and commits of one item.
fn fetch_details(repo: &Repo, t: &mut Thread, warnings: &mut Vec<String>) {
    let base = format!("repos/{}/{}", repo.owner, repo.name);
    let n = t.number;
    let comments = soft(gh_pages(&format!("{base}/issues/{n}/comments")), &format!("#{n} comments"), warnings);
    t.comments = comments.iter().map(note_from).collect();
    if t.is_pr {
        let inline = soft(gh_pages(&format!("{base}/pulls/{n}/comments")), &format!("PR #{n} review comments"), warnings);
        t.inline = inline.iter().map(inline_from).collect();
        let commits = soft(gh_pages(&format!("{base}/pulls/{n}/commits")), &format!("PR #{n} commits"), warnings);
        t.commits = commits.iter().map(commit_from).collect();
        let reviews = soft(gh_pages(&format!("{base}/pulls/{n}/reviews")), &format!("PR #{n} reviews"), warnings);
        t.reviews = reviews
            .iter()
            .map(|v| Review {
                id: v.get("id").and_then(Value::as_u64).unwrap_or(0),
                author: login(v, "user"),
                state: s(v, "state"),
                body: s(v, "body"),
                submitted: day(&s(v, "submitted_at")),
                commit: short_sha(&s(v, "commit_id")),
            })
            .collect();
        t.tree = tree_lines(t);
    }
    t.detailed = true;
}

/// Commit and review lines of a PR, as shown under it in `tree.md`.
fn tree_lines(t: &Thread) -> Vec<String> {
    let mut lines: Vec<String> = t
        .commits
        .iter()
        .map(|c| format!("commit `{}` {} ({}, {})", c.sha, c.title, c.author, c.date))
        .collect();
    lines.extend(
        t.reviews
            .iter()
            .map(|r| format!("review {} by {}: {} on `{}`", r.state, r.author, r.submitted, r.commit)),
    );
    lines
}

const DISCUSSIONS_QUERY: &str = "query($owner:String!,$name:String!,$cursor:String){\
  repository(owner:$owner,name:$name){discussions(first:50,after:$cursor){\
  pageInfo{hasNextPage endCursor}\
  nodes{number title body url createdAt updatedAt author{login} category{name}\
  comments(first:100){nodes{author{login} body createdAt url}}}}}}";

fn fetch_discussions(repo: &Repo) -> Result<Vec<Discussion>> {
    let mut out = Vec::new();
    let mut cursor = "null".to_string();
    loop {
        let args = [
            "api".to_string(),
            "graphql".to_string(),
            "-f".to_string(),
            format!("query={DISCUSSIONS_QUERY}"),
            "-F".to_string(),
            format!("owner={}", repo.owner),
            "-F".to_string(),
            format!("name={}", repo.name),
            "-F".to_string(),
            format!("cursor={cursor}"),
        ];
        let data = gh_json(&args)?;
        let discussions = data
            .pointer("/data/repository/discussions")
            .context("no discussions in the GraphQL response")?;
        for v in discussions.get("nodes").and_then(Value::as_array).into_iter().flatten() {
            out.push(Discussion {
                number: v.get("number").and_then(Value::as_u64).unwrap_or(0),
                title: s(v, "title"),
                body: s(v, "body"),
                url: s(v, "url"),
                author: v
                    .get("author")
                    .and_then(|a| a.get("login"))
                    .and_then(Value::as_str)
                    .map_or("@ghost".into(), |l| format!("@{l}")),
                created: day(&s(v, "createdAt")),
                updated: s(v, "updatedAt"),
                category: v.get("category").and_then(|c| c.get("name")).and_then(Value::as_str).unwrap_or("").to_string(),
                comments: v
                    .pointer("/comments/nodes")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|c| Note {
                        author: c
                            .get("author")
                            .and_then(|a| a.get("login"))
                            .and_then(Value::as_str)
                            .map_or("@ghost".into(), |l| format!("@{l}")),
                        created: day(&s(c, "createdAt")),
                        body: s(c, "body"),
                        url: s(c, "url"),
                        location: String::new(),
                        commit: String::new(),
                        review_id: 0,
                    })
                    .collect(),
            });
        }
        let page = discussions.get("pageInfo");
        let more = page.and_then(|p| p.get("hasNextPage")).and_then(Value::as_bool).unwrap_or(false);
        if !more {
            return Ok(out);
        }
        // Cursors are opaque strings; re-encode as a JSON string for `-F`.
        cursor = format!("\"{}\"", page.map(|p| s(p, "endCursor")).unwrap_or_default());
    }
}

/// Clones `<repo>.wiki.git` shallowly and returns its markdown pages.
fn fetch_wiki(index_dir: &Path, repo: &Repo) -> Result<Vec<(String, String)>> {
    let dir = index_dir.join("wiki-clone");
    let _ = std::fs::remove_dir_all(&dir);
    let url = format!("https://github.com/{}.wiki.git", repo.slug());
    let out = Command::new("git")
        .env("GIT_TERMINAL_PROMPT", "0")
        .args(["clone", "--depth", "1", "--quiet"])
        .arg(&url)
        .arg(&dir)
        .output()
        .context("running git")?;
    if !out.status.success() {
        bail!("git clone {url}: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let mut pages = Vec::new();
    for entry in std::fs::read_dir(&dir)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) == Some("md") {
            let name = entry.file_name().to_string_lossy().into_owned();
            pages.push((name, std::fs::read_to_string(&path).unwrap_or_default()));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    pages.sort();
    Ok(pages)
}

// ---------------------------------------------------------------------------
// Relations
// ---------------------------------------------------------------------------

const CLOSING_WORDS: &[&str] = &[
    "close", "closes", "closed", "fix", "fixes", "fixed", "resolve", "resolves", "resolved",
];

/// Issue numbers closed by GitHub closing keywords (`Fixes #12`, `closes: #3`).
fn closing_refs(text: &str) -> BTreeSet<u64> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = BTreeSet::new();
    for pair in words.windows(2) {
        let keyword = pair[0].trim_end_matches(':').to_ascii_lowercase();
        if CLOSING_WORDS.contains(&keyword.as_str()) {
            if let Some(n) = issue_number(pair[1]) {
                out.insert(n);
            }
        }
    }
    out
}

fn issue_number(word: &str) -> Option<u64> {
    let word = word.trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
    let digits = word.strip_prefix('#')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Every `#N` mention that is not part of a longer word (`owner/repo#N` is
/// skipped: it points at another repository's item).
fn mention_refs(text: &str, own: u64) -> BTreeSet<u64> {
    let mut out = BTreeSet::new();
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b != b'#' || (i > 0 && bytes[i - 1].is_ascii_alphanumeric()) {
            continue;
        }
        let digits: String = text[i + 1..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() || digits.len() > 9 {
            continue;
        }
        if let Ok(n) = digits.parse::<u64>() {
            if n > 0 && n != own {
                out.insert(n);
            }
        }
    }
    out
}

/// Fills `closes` and `refs` of every fetched thread from its own text: title,
/// body, comments, reviews and commit messages. Unfetched threads keep the
/// links stored in their signature.
fn link(threads: &mut BTreeMap<u64, Thread>) {
    for t in threads.values_mut().filter(|t| t.detailed) {
        let mut text = format!("{}\n{}\n", t.title, t.body);
        for c in t.comments.iter().chain(t.inline.iter()) {
            text.push_str(&c.body);
            text.push('\n');
        }
        for r in &t.reviews {
            text.push_str(&r.body);
            text.push('\n');
        }
        for c in &t.commits {
            text.push_str(&c.title);
            text.push('\n');
        }
        if t.is_pr {
            t.closes = closing_refs(&text);
        }
        t.refs = mention_refs(&text, t.number);
    }
}

/// `(issue -> PRs closing it, item -> items that mention it)`.
fn reverse_links(threads: &BTreeMap<u64, Thread>) -> (HashMap<u64, BTreeSet<u64>>, HashMap<u64, BTreeSet<u64>>) {
    let mut closed_by: HashMap<u64, BTreeSet<u64>> = HashMap::new();
    let mut referenced_by: HashMap<u64, BTreeSet<u64>> = HashMap::new();
    for t in threads.values() {
        for &issue in &t.closes {
            closed_by.entry(issue).or_default().insert(t.number);
        }
        for &other in &t.refs {
            referenced_by.entry(other).or_default().insert(t.number);
        }
    }
    (closed_by, referenced_by)
}

/// Canonical text of one item's reverse relations, stored in its signature.
fn relations_of(number: u64, closed_by: &HashMap<u64, BTreeSet<u64>>, referenced_by: &HashMap<u64, BTreeSet<u64>>) -> String {
    let closed = closed_by.get(&number).cloned().unwrap_or_default();
    let referenced = referenced_by.get(&number).cloned().unwrap_or_default();
    format!("closed_by={closed:?};referenced_by={referenced:?}")
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Markdown link to a rendered item, relative to `github-meta/<kind>/`; plain
/// `#N` when the item is not part of this repository's sync.
fn ref_link(n: u64, threads: &BTreeMap<u64, Thread>) -> String {
    match threads.get(&n) {
        Some(t) if t.is_pr => format!("[PR #{n}](../pulls/{n}.md)"),
        Some(_) => format!("[Issue #{n}](../issues/{n}.md)"),
        None => format!("#{n}"),
    }
}

fn ref_list<'a>(items: impl IntoIterator<Item = &'a u64>, threads: &BTreeMap<u64, Thread>) -> String {
    let parts: Vec<String> = items.into_iter().map(|n| ref_link(*n, threads)).collect();
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(", ")
    }
}

fn quote(body: &str) -> String {
    body.trim().lines().map(|l| format!("> {l}\n")).collect()
}

fn or_empty(body: &str) -> &str {
    if body.trim().is_empty() {
        "_(no text)_"
    } else {
        body.trim()
    }
}

fn render_thread(
    repo: &Repo,
    t: &Thread,
    threads: &BTreeMap<u64, Thread>,
    closed_by: &BTreeSet<u64>,
    referenced_by: &BTreeSet<u64>,
) -> String {
    let mut out = String::new();
    let kind = if t.is_pr { "Pull request" } else { "Issue" };
    let _ = writeln!(out, "# {kind} #{}: {}\n", t.number, t.title);
    let _ = writeln!(out, "- **Repository:** {}", repo.slug());
    let _ = writeln!(out, "- **State:** {}", t.state);
    let _ = writeln!(out, "- **Author:** {}", t.author);
    let _ = writeln!(out, "- **Created:** {}", t.created);
    if !t.closed.is_empty() {
        let _ = writeln!(out, "- **Closed/merged:** {}", t.closed);
    }
    if !t.labels.is_empty() {
        let _ = writeln!(out, "- **Labels:** {}", t.labels.join(", "));
    }
    let _ = writeln!(out, "- **URL:** {}\n", t.url);

    let _ = writeln!(out, "## Relations\n");
    if t.is_pr {
        let _ = writeln!(out, "- **Closes:** {}", ref_list(&t.closes, threads));
        let _ = writeln!(out, "- **Commits:** {}", t.commits.len());
        let _ = writeln!(out, "- **Reviews:** {}", t.reviews.len());
    } else {
        let _ = writeln!(out, "- **Closed by:** {}", ref_list(closed_by, threads));
    }
    let _ = writeln!(out, "- **References:** {}", ref_list(&t.refs, threads));
    let _ = writeln!(out, "- **Referenced by:** {}\n", ref_list(referenced_by, threads));

    let _ = writeln!(out, "## Description\n\n{}\n", or_empty(&t.body));

    if t.is_pr {
        let _ = writeln!(out, "## Commits\n");
        if t.commits.is_empty() {
            let _ = writeln!(out, "_(none)_\n");
        }
        for c in &t.commits {
            let _ = writeln!(out, "- `{}` {} — {}, {}", c.sha, c.title, c.author, c.date);
        }
        out.push('\n');

        let _ = writeln!(out, "## Reviews\n");
        let mut attached: HashSet<usize> = HashSet::new();
        for r in &t.reviews {
            let _ = writeln!(out, "### {} — {} on `{}` — {}\n", r.author, r.state, r.commit, r.submitted);
            if !r.body.trim().is_empty() {
                out.push_str(&quote(&r.body));
                out.push('\n');
            }
            for (i, c) in t.inline.iter().enumerate().filter(|(_, c)| c.review_id == r.id) {
                attached.insert(i);
                let _ = writeln!(out, "- `{}` (on `{}`) {}: {}", c.location, c.commit, c.author, c.body.trim().replace('\n', " "));
            }
            out.push('\n');
        }
        let orphans: Vec<&Note> = t.inline.iter().enumerate().filter(|(i, _)| !attached.contains(i)).map(|(_, c)| c).collect();
        if !orphans.is_empty() {
            let _ = writeln!(out, "### Other inline review comments\n");
            for c in orphans {
                let _ = writeln!(out, "- `{}` (on `{}`) {}: {}", c.location, c.commit, c.author, c.body.trim().replace('\n', " "));
            }
            out.push('\n');
        }
    }

    if !t.comments.is_empty() {
        let _ = writeln!(out, "## Conversation\n");
        for c in &t.comments {
            let _ = writeln!(out, "### {} — {} ([link]({}))\n\n{}\n", c.author, c.created, c.url, or_empty(&c.body));
        }
    }
    out
}

fn render_discussion(repo: &Repo, d: &Discussion) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Discussion #{}: {}\n", d.number, d.title);
    let _ = writeln!(out, "- **Repository:** {}", repo.slug());
    let _ = writeln!(out, "- **Category:** {}", d.category);
    let _ = writeln!(out, "- **Author:** {}", d.author);
    let _ = writeln!(out, "- **Created:** {}", d.created);
    let _ = writeln!(out, "- **URL:** {}\n", d.url);
    let _ = writeln!(out, "## Description\n\n{}\n", or_empty(&d.body));
    if !d.comments.is_empty() {
        let _ = writeln!(out, "## Comments\n");
        for c in &d.comments {
            let _ = writeln!(out, "### {} — {}\n\n{}\n", c.author, c.created, or_empty(&c.body));
        }
    }
    out
}

/// One page linking each item to the items closest to it.
fn render_tree(repo: &Repo, threads: &BTreeMap<u64, Thread>, discussions: &[Discussion], closed_by: &HashMap<u64, BTreeSet<u64>>) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# GitHub knowledge tree — {}\n", repo.slug());
    let _ = writeln!(out, "Each issue lists the PRs that close it; each PR lists its commits and reviews. Open the linked page for the full text.\n");

    let _ = writeln!(out, "## Issues\n");
    for t in threads.values().filter(|t| !t.is_pr) {
        let _ = writeln!(out, "- [#{} {}](issues/{}.md) — {}", t.number, t.title, t.number, t.state);
        for pr in closed_by.get(&t.number).into_iter().flatten() {
            if let Some(p) = threads.get(pr) {
                let _ = writeln!(out, "  - closed by [PR #{pr} {}](pulls/{pr}.md) — {}", p.title, p.state);
            }
        }
    }

    let _ = writeln!(out, "\n## Pull requests\n");
    for t in threads.values().filter(|t| t.is_pr) {
        let closes = if t.closes.is_empty() { String::new() } else { format!(" — closes {}", ref_list(&t.closes, threads)) };
        let _ = writeln!(out, "- [#{} {}](pulls/{}.md) — {}{closes}", t.number, t.title, t.number, t.state);
        for line in &t.tree {
            let _ = writeln!(out, "  - {line}");
        }
    }

    if !discussions.is_empty() {
        let _ = writeln!(out, "\n## Discussions\n");
        for d in discussions {
            let _ = writeln!(out, "- [#{} {}](discussions/{}.md) — {}", d.number, d.title, d.number, d.category);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Sync and cache
// ---------------------------------------------------------------------------

fn write_if_changed(path: &Path, content: &str) -> Result<()> {
    if std::fs::read_to_string(path).ok().as_deref() == Some(content) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)?;
    Ok(())
}

/// Removes documents that no sync keeps any more.
fn prune(dir: &Path, base: &Path, keep: &HashSet<String>) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    for e in read.flatten() {
        let path = e.path();
        if path.is_dir() {
            prune(&path, base, keep);
        } else {
            let inner = path.strip_prefix(base).unwrap_or(&path).to_string_lossy().replace('\\', "/");
            if !keep.contains(&inner) {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

/// Fetches the root's GitHub metadata into `<index_dir>/github-meta/`. Returns
/// None when the root has no GitHub `origin` or the feature is disabled.
///
/// Items whose `updated_at`, title or state match their signature are not
/// fetched again; an item whose reverse relations moved (someone closed or
/// mentioned it) is fetched and re-rendered even if it did not change itself.
pub fn sync(root: &Path, index_dir: &Path, cfg: &GithubSettings) -> Result<Option<SyncReport>> {
    if !cfg.enabled {
        return Ok(None);
    }
    let Some(repo) = detect_repo(root) else { return Ok(None) };
    let mut warnings = Vec::new();
    let old = load_sigs(index_dir, &repo);
    let dir = index_dir.join(DIR);

    // 1. Listings are cheap; details only for items whose signature moved.
    let mut threads = if cfg.issues || cfg.pulls {
        fetch_lists(&repo, cfg, &mut warnings)
    } else {
        BTreeMap::new()
    };
    for t in threads.values_mut() {
        match old.items.get(&t.rel()) {
            Some(p) if p.updated == t.updated && p.title == t.title && p.state == t.state => {
                t.closes = p.closes.clone();
                t.refs = p.refs.clone();
                t.tree = p.tree.clone();
            }
            _ => fetch_details(&repo, t, &mut warnings),
        }
    }
    link(&mut threads);

    // 2. Items whose reverse relations changed, or whose file went missing, are re-rendered too.
    let (closed_by, referenced_by) = reverse_links(&threads);
    let mut moved: Vec<u64> = Vec::new();
    for t in threads.values() {
        if t.detailed {
            continue;
        }
        let relations = relations_of(t.number, &closed_by, &referenced_by);
        let stale = old.items.get(&t.rel()).is_none_or(|p| p.relations != relations) || !dir.join(t.rel()).is_file();
        if stale {
            moved.push(t.number);
        }
    }
    for n in moved {
        if let Some(t) = threads.get_mut(&n) {
            fetch_details(&repo, t, &mut warnings);
        }
    }
    link(&mut threads);
    let (closed_by, referenced_by) = reverse_links(&threads);

    // 3. Discussions: rendered only when their `updatedAt` moved.
    let discussions = if cfg.discussions {
        soft(fetch_discussions(&repo), "discussions", &mut warnings)
    } else {
        Vec::new()
    };
    let wiki = if cfg.wiki {
        soft(fetch_wiki(index_dir, &repo), "wiki", &mut warnings)
    } else {
        Vec::new()
    };

    // 4. Signatures and the documents that need writing.
    let empty = BTreeSet::new();
    let mut sigs = Sigs { repo: repo.slug(), items: BTreeMap::new() };
    let mut writes: BTreeMap<String, String> = BTreeMap::new();
    let mut changed = 0;
    for t in threads.values() {
        let rel = t.rel();
        sigs.items.insert(
            rel.clone(),
            Sig {
                updated: t.updated.clone(),
                title: t.title.clone(),
                state: t.state.clone(),
                closes: t.closes.clone(),
                refs: t.refs.clone(),
                tree: t.tree.clone(),
                relations: relations_of(t.number, &closed_by, &referenced_by),
            },
        );
        if t.detailed {
            let closed = closed_by.get(&t.number).unwrap_or(&empty);
            let referenced = referenced_by.get(&t.number).unwrap_or(&empty);
            writes.insert(rel, render_thread(&repo, t, &threads, closed, referenced));
            changed += 1;
        }
    }
    for d in &discussions {
        let rel = format!("discussions/{}.md", d.number);
        let unchanged = old.items.get(&rel).is_some_and(|p| p.updated == d.updated) && dir.join(&rel).is_file();
        sigs.items.insert(rel.clone(), Sig { updated: d.updated.clone(), title: d.title.clone(), ..Sig::default() });
        if !unchanged {
            writes.insert(rel, render_discussion(&repo, d));
            changed += 1;
        }
    }
    for (name, content) in &wiki {
        let header = format!(
            "# Wiki: {page}\n\nSource: https://github.com/{}/wiki/{page}\n\n",
            repo.slug(),
            page = name.trim_end_matches(".md"),
        );
        writes.insert(format!("wiki/{name}"), format!("{header}{content}"));
    }
    writes.insert("tree.md".to_string(), render_tree(&repo, &threads, &discussions, &closed_by));

    let removed = old.items.keys().filter(|k| !sigs.items.contains_key(*k)).count();
    let issues = threads.values().filter(|t| !t.is_pr).count();
    let pulls = threads.len() - issues;

    // 5. Write what changed; everything else keeps its file and its vectors.
    let mut keep: HashSet<String> = HashSet::new();
    for rel in sigs.items.keys() {
        keep.insert(rel.clone());
    }
    for (rel, content) in &writes {
        write_if_changed(&dir.join(rel), content)?;
        keep.insert(rel.clone());
    }
    prune(&dir, &dir, &keep);
    if let Err(e) = save_sigs(index_dir, &sigs) {
        warnings.push(format!("signatures not saved: {e:#}"));
    }

    Ok(Some(SyncReport {
        repo: repo.slug(),
        issues,
        pulls,
        discussions: discussions.len(),
        wiki_pages: wiki.len(),
        changed,
        removed,
        warnings,
    }))
}

/// Title (first heading) and web link of a materialized document, for search
/// results: issues, PRs and discussions carry `- **URL:**`, wiki pages `Source:`.
pub fn describe(content: &str) -> (Option<String>, Option<String>) {
    let title = content.lines().find_map(|l| l.strip_prefix("# ")).map(|t| t.trim().to_string());
    let url = content.lines().take(20).find_map(|l| {
        l.strip_prefix("- **URL:** ").or_else(|| l.strip_prefix("Source: ")).map(|u| u.trim().to_string())
    });
    (title, url)
}

/// The cached GitHub documents as indexable files. Reads the disk only, so it
/// is safe for background refreshes.
pub fn cached_files(index_dir: &Path) -> Vec<DiscoveredFile> {
    let dir = index_dir.join(DIR);
    let mut out = Vec::new();
    collect(&dir, &dir, &mut out);
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    out
}

fn collect(base: &Path, dir: &Path, out: &mut Vec<DiscoveredFile>) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    for e in read.flatten() {
        let path = e.path();
        if path.is_dir() {
            collect(base, &path, out);
            continue;
        }
        if path.extension().and_then(|x| x.to_str()) != Some("md") {
            continue;
        }
        let Ok(md) = e.metadata() else { continue };
        let mtime_ms = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_millis() as i64);
        let inner = path.strip_prefix(base).unwrap_or(&path).to_string_lossy().replace('\\', "/");
        // Wiki pages are documentation (P0); issue and PR threads are conversation (P1).
        let tier = if inner.starts_with("wiki/") { Tier::P0 } else { Tier::P1 };
        out.push(DiscoveredFile {
            abs: path.clone(),
            rel: format!("{DIR}/{inner}"),
            size: md.len(),
            mtime_ms,
            tier,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pr_thread(n: u64, body: &str) -> Thread {
        Thread { number: n, is_pr: true, title: "t".into(), body: body.into(), detailed: true, ..Thread::default() }
    }

    #[test]
    fn describes_issue_and_wiki_documents() {
        let issue = "# Issue #3: Broken search\n\n- **State:** open\n- **URL:** https://github.com/o/r/issues/3\n";
        assert_eq!(
            describe(issue),
            (Some("Issue #3: Broken search".to_string()), Some("https://github.com/o/r/issues/3".to_string()))
        );
        let wiki = "# Wiki: Home\n\nSource: https://github.com/o/r/wiki/Home\n\nbody";
        assert_eq!(describe(wiki).1.as_deref(), Some("https://github.com/o/r/wiki/Home"));
    }

    #[test]
    fn parses_github_remotes() {
        let want = Some(Repo { owner: "IDjinn".into(), name: "rag".into() });
        assert_eq!(parse_remote("https://github.com/IDjinn/rag.git"), want);
        assert_eq!(parse_remote("git@github.com:IDjinn/rag.git"), want);
        assert_eq!(parse_remote("https://github.com/IDjinn/rag/"), want);
        assert_eq!(parse_remote("https://gitlab.com/IDjinn/rag.git"), None);
    }

    #[test]
    fn closing_keywords_link_prs_to_issues() {
        let got: Vec<u64> = closing_refs("Fixes #12 and also see #13.\nCloses: #7, resolves #9").into_iter().collect();
        assert_eq!(got, vec![7, 9, 12]);
        assert!(closing_refs("related to #5").is_empty());
    }

    #[test]
    fn mentions_skip_other_repositories_and_headings() {
        let got: Vec<u64> = mention_refs("see #4, other/repo#8, ## Title, (#11) and #4 again", 1).into_iter().collect();
        assert_eq!(got, vec![4, 11]);
        assert!(mention_refs("#3", 3).is_empty());
    }

    #[test]
    fn unfetched_threads_keep_their_stored_links() {
        let mut threads = BTreeMap::new();
        let mut kept = pr_thread(45, "Fixes #12");
        kept.detailed = false;
        kept.closes = BTreeSet::from([12]);
        threads.insert(45, kept);
        link(&mut threads);
        assert_eq!(threads[&45].closes, BTreeSet::from([12]));
        threads.insert(46, pr_thread(46, "Closes #3"));
        link(&mut threads);
        assert_eq!(threads[&46].closes, BTreeSet::from([3]));
    }

    #[test]
    fn relation_change_is_detected_from_reverse_links() {
        let mut threads = BTreeMap::new();
        threads.insert(12, Thread { number: 12, ..Thread::default() });
        let before = reverse_links(&threads);
        let sig_before = relations_of(12, &before.0, &before.1);
        threads.insert(45, pr_thread(45, "Fixes #12"));
        link(&mut threads);
        let after = reverse_links(&threads);
        assert_ne!(relations_of(12, &after.0, &after.1), sig_before);
    }

    #[test]
    fn pr_tree_links_issue_commits_and_reviews() {
        let mut threads = BTreeMap::new();
        threads.insert(12, Thread { number: 12, title: "crash on start".into(), state: "closed".into(), detailed: true, ..Thread::default() });
        let mut pr = pr_thread(45, "Fixes #12");
        pr.title = "fix start crash".into();
        pr.state = "merged".into();
        pr.commits.push(CommitRef { sha: "abc1234".into(), title: "guard null config".into(), author: "@ana".into(), date: "2026-10-01".into() });
        pr.reviews.push(Review { id: 900, author: "@bob".into(), state: "APPROVED".into(), body: String::new(), submitted: "2026-10-02".into(), commit: "abc1234".into() });
        pr.inline.push(Note {
            author: "@bob".into(),
            created: "2026-10-02".into(),
            body: "nit: rename".into(),
            url: String::new(),
            location: "src/main.rs:10".into(),
            commit: "abc1234".into(),
            review_id: 900,
        });
        pr.tree = tree_lines(&pr);
        threads.insert(45, pr);
        link(&mut threads);
        let (closed_by, referenced_by) = reverse_links(&threads);
        let repo = Repo { owner: "o".into(), name: "r".into() };
        let issue = render_thread(&repo, &threads[&12], &threads, closed_by.get(&12).unwrap(), referenced_by.get(&12).unwrap_or(&BTreeSet::new()));
        assert!(issue.contains("**Closed by:** [PR #45](../pulls/45.md)"), "{issue}");
        let pr_doc = render_thread(&repo, &threads[&45], &threads, &BTreeSet::new(), &BTreeSet::new());
        assert!(pr_doc.contains("**Closes:** [Issue #12](../issues/12.md)"), "{pr_doc}");
        assert!(pr_doc.contains("- `src/main.rs:10` (on `abc1234`) @bob: nit: rename"), "{pr_doc}");
        let tree = render_tree(&repo, &threads, &[], &closed_by);
        assert!(tree.contains("closed by [PR #45 fix start crash](pulls/45.md)"), "{tree}");
        assert!(tree.contains("commit `abc1234` guard null config"), "{tree}");
    }
}
