use crate::config::{self, Tier};
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug)]
pub struct DiscoveredFile {
    pub abs: PathBuf,
    /// POSIX-style path relative to the root.
    pub rel: String,
    pub size: u64,
    pub mtime_ms: i64,
    pub tier: Tier,
}

/// Dependency-tree files never worth embedding: anything named `*.lock` is
/// already excluded (no "lock" extension in the tier lists); these have other
/// extensions or none.
const DEPENDENCY_FILES: &[&str] = &[
    "cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lock",
    "npm-shrinkwrap.json",
    "composer.lock",
    "gemfile.lock",
    "pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "pdm.lock",
    "go.sum",
    "packages.lock.json",
    "flake.lock",
];

/// Walk a root honoring .gitignore/.ignore/.git/info/exclude (nested and global),
/// skipping hidden entries, non-whitelisted names and oversize files.
pub fn walk(root: &Path, respect_ignore: bool) -> Vec<DiscoveredFile> {
    let mut builder = WalkBuilder::new(root);
    builder.hidden(true);
    builder.git_ignore(respect_ignore);
    builder.git_global(respect_ignore);
    builder.git_exclude(respect_ignore);
    builder.ignore(respect_ignore);
    builder.parents(respect_ignore);
    // Apply gitignore rules even outside a git repository, as requested.
    builder.require_git(false);
    if respect_ignore {
        // ZCode's own ignore file (gitignore syntax), honored alongside .gitignore.
        builder.add_custom_ignore_filename(".zcodeignore");
    }
    builder.threads(std::thread::available_parallelism().map_or(4, |n| n.get()));

    let mut out: Vec<DiscoveredFile> = Vec::new();
    for entry in builder.build().flatten() {
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(tier) = config::tier_for(name) else {
            continue;
        };
        if DEPENDENCY_FILES.contains(&lower_name(name).as_str()) {
            continue;
        }
        let Ok(md) = entry.metadata() else { continue };
        let max_bytes = config::settings().index.max_file_kb * 1024;
        if md.len() == 0 || md.len() > max_bytes {
            continue;
        }
        let mtime_ms = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        out.push(DiscoveredFile {
            abs: path.to_path_buf(),
            rel,
            size: md.len(),
            mtime_ms,
            tier,
        });
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    out
}

fn lower_name(name: &str) -> String {
    name.to_ascii_lowercase()
}

/// Read a file as UTF-8 (lossy never: null on binary or unreadable).
pub fn read_text(abs: &Path) -> Option<String> {    let bytes = std::fs::read(abs).ok()?;
    let probe = &bytes[..bytes.len().min(4096)];
    if probe.contains(&0) {
        return None; // binary
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

pub fn same_state(a: (i64, u64), b: (i64, u64)) -> bool {
    (a.0 - b.0).abs() < 2 && a.1 == b.1
}
