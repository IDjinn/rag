use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// History entries kept per index in stats.json.
pub const HISTORY_LIMIT: usize = 50;

// ---------------------------------------------------------------------------
// YAML configuration file (~/.zcode/ia/config.yaml by default, RAG_CONFIG to
// override). Precedence: CLI flag > environment variable > YAML > default.
// ---------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct Settings {
    pub provider: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub embed_model: Option<String>,
    pub price_per_1m_tokens: Option<f64>,
    pub index: IndexSettings,
    pub search: SearchSettings,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct IndexSettings {
    pub batch_size: usize,
    pub concurrency: usize,
    pub max_file_kb: u64,
    pub chunk_target_chars: usize,
    pub chunk_overlap_chars: usize,
    /// Whether the MCP server keeps indexed roots up to date automatically.
    pub auto_refresh: bool,
    /// How often (seconds) the auto-refresh loop scans watched roots.
    pub auto_refresh_interval_secs: u64,
    /// Extensions classified P0 ("core": source code and documentation).
    pub p0_exts: Vec<String>,
    /// Extensions classified P1 ("support": everything else textual).
    pub p1_exts: Vec<String>,
    /// Whole file names (no extension semantics) always classified P1.
    pub p1_names: Vec<String>,
    /// Dependency-lock file names that are never indexed.
    pub dependency_files: Vec<String>,
    /// Glob patterns upgrading matching files to P0 regardless of extension.
    /// Matched (case-insensitively) against the root-relative POSIX path and
    /// the bare file name; `*` does not cross `/`, use `**/`.
    pub p0_globs: Vec<String>,
    /// Glob patterns classifying matching files as P1 when nothing above did.
    pub p1_globs: Vec<String>,
}

impl Default for IndexSettings {
    fn default() -> Self {
        Self {
            batch_size: 32,
            concurrency: 3,
            max_file_kb: 512,
            chunk_target_chars: 1100,
            chunk_overlap_chars: 160,
            auto_refresh: true,
            auto_refresh_interval_secs: 10,
            p0_exts: list(P0_EXTS),
            p1_exts: list(P1_EXTS),
            p1_names: list(P1_NAMES),
            dependency_files: list(DEPENDENCY_FILES),
            p0_globs: Vec::new(),
            p1_globs: Vec::new(),
        }
    }
}

fn list(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct SearchSettings {
    pub k: usize,
    /// Score multiplier applied to P1 (low-priority) results.
    pub p1_damp: f32,
    /// Cosine score at or above which a search counts as a "hit" in stats.
    pub hit_threshold: f32,
    pub snippet_max_chars: usize,
}

impl Default for SearchSettings {
    fn default() -> Self {
        Self { k: 5, p1_damp: 0.9, hit_threshold: 0.3, snippet_max_chars: 1600 }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            provider: "lmstudio".to_string(),
            base_url: None,
            api_key: None,
            embed_model: None,
            price_per_1m_tokens: None,
            index: IndexSettings::default(),
            search: SearchSettings::default(),
        }
    }
}

pub fn config_path() -> PathBuf {
    std::env::var("RAG_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir().join("config.yaml"))
}

fn load() -> Settings {
    let path = config_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => match serde_yaml::from_str(&text) {
            Ok(settings) => settings,
            Err(e) => {
                eprintln!("[rag] ignoring invalid config {}: {e}", path.display());
                Settings::default()
            }
        },
        Err(_) => Settings::default(),
    }
}

pub fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(load)
}

pub fn write_default_config(path: &Path) -> std::io::Result<()> {
    // Lists are written out in full so each entry is visible and editable;
    // a list present in the file replaces the built-in default entirely.
    let template = format!(
        "\
# rag configuration — environment variables (RAG_*) override these values.
provider: lmstudio                # lmstudio | ollama | openai (OpenAI-compatible)
# base_url: http://localhost:1234/v1
# api_key: \"\"                      # Bearer token for cloud providers
embed_model: qwen.qwen3-vl-embedding-2b   # empty/absent = auto-detect from loaded models
# price_per_1m_tokens: 0.0         # USD per 1M embedding tokens (shows in cost stats)

index:
  batch_size: 32                  # chunks per embedding request
  concurrency: 3                  # parallel embedding requests
  max_file_kb: 512
  chunk_target_chars: 1100
  chunk_overlap_chars: 160
  auto_refresh: true              # MCP server re-embeds changed files automatically
  auto_refresh_interval_secs: 10  # how often watched roots are scanned for changes

  # Tier classification. A list present here replaces the default entirely;
  # globs are matched (case-insensitive) against the root-relative path and
  # the file name — `*` does not cross `/`, use `**/` for directories.
  p0_exts: {p0_exts}
  p1_exts: {p1_exts}
  p1_names: {p1_names}
  dependency_files: {dep_files}   # never indexed (lockfiles, dependency sums)
  p0_globs: []                    # e.g. [\"**/migrations/**\", \"*.proto\"]
  p1_globs: []                    # e.g. [\"**/*.csv\", \"**/fixtures/**\"]

search:
  k: 5
  p1_damp: 0.9                    # score multiplier for P1 (non source/md) results
  hit_threshold: 0.3
  snippet_max_chars: 1600
",
        p0_exts = yaml_list(&IndexSettings::default().p0_exts),
        p1_exts = yaml_list(&IndexSettings::default().p1_exts),
        p1_names = yaml_list(&IndexSettings::default().p1_names),
        dep_files = yaml_list(&IndexSettings::default().dependency_files),
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, template)
}

/// One-line YAML flow list with entries quoted for safety.
fn yaml_list(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("\"{s}\"")).collect();
    format!("[{}]", inner.join(", "))
}

// ---------------------------------------------------------------------------
// Resolved accessors: env > YAML > default
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    LmStudio,
    Ollama,
    OpenAi,
}

impl Provider {
    pub fn name(self) -> &'static str {
        match self {
            Provider::LmStudio => "lmstudio",
            Provider::Ollama => "ollama",
            Provider::OpenAi => "openai",
        }
    }
    fn default_base_url(self) -> &'static str {
        match self {
            Provider::LmStudio => "http://localhost:1234/v1",
            Provider::Ollama => "http://localhost:11434/v1",
            Provider::OpenAi => "https://api.openai.com/v1",
        }
    }
}

/// RAG_PROVIDER overrides; else YAML; else lmstudio.
pub fn provider() -> Provider {
    let from_env = std::env::var("RAG_PROVIDER").unwrap_or_default().to_lowercase();
    let name = if from_env.is_empty() { settings().provider.to_lowercase() } else { from_env };
    match name.as_str() {
        "ollama" => Provider::Ollama,
        "openai" | "openai-compatible" => Provider::OpenAi,
        _ => Provider::LmStudio,
    }
}

pub fn provider_name() -> &'static str {
    provider().name()
}

pub fn base_url() -> String {
    if let Ok(url) = std::env::var("RAG_BASE_URL").or_else(|_| std::env::var("RAG_LMSTUDIO_URL")) {
        if !url.is_empty() {
            return url.trim_end_matches('/').to_string();
        }
    }
    if let Some(url) = &settings().base_url {
        if !url.is_empty() {
            return url.trim_end_matches('/').to_string();
        }
    }
    provider().default_base_url().to_string()
}

pub fn api_key() -> Option<String> {
    if let Ok(key) = std::env::var("RAG_API_KEY") {
        if !key.is_empty() {
            return Some(key);
        }
    }
    settings().api_key.clone().filter(|k| !k.is_empty())
}

/// Explicit embedding model: RAG_EMBED_MODEL overrides YAML; None = auto-detect.
pub fn embed_model_env() -> Option<String> {
    if let Ok(m) = std::env::var("RAG_EMBED_MODEL") {
        if !m.is_empty() {
            return Some(m);
        }
    }
    settings().embed_model.clone().filter(|m| !m.is_empty())
}

pub fn price_per_1m_tokens() -> Option<f64> {
    if let Ok(p) = std::env::var("RAG_PRICE_PER_1M_TOKENS") {
        if let Ok(price) = p.parse::<f64>() {
            return (price > 0.0).then_some(price);
        }
    }
    settings().price_per_1m_tokens.filter(|p| *p > 0.0)
}

fn env_flag(name: &str) -> Option<bool> {
    let raw = std::env::var(name).unwrap_or_default().to_lowercase();
    match raw.as_str() {
        "" => None,
        "0" | "false" | "off" | "no" => Some(false),
        _ => Some(true),
    }
}

/// RAG_AUTO_REFRESH overrides; else YAML; else on.
pub fn auto_refresh() -> bool {
    env_flag("RAG_AUTO_REFRESH").unwrap_or(settings().index.auto_refresh)
}

/// RAG_AUTO_REFRESH_SECS overrides; else YAML; minimum 1s.
pub fn auto_refresh_interval_secs() -> u64 {
    if let Ok(s) = std::env::var("RAG_AUTO_REFRESH_SECS") {
        if let Ok(secs) = s.parse::<u64>() {
            if secs >= 1 {
                return secs;
            }
        }
    }
    settings().index.auto_refresh_interval_secs.max(1)
}

pub fn data_dir() -> PathBuf {
    std::env::var("RAG_DATA_DIR")
        .map(PathBuf::from)
        .ok()
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".zcode")
                .join("ia")
        })
}

/// One index directory per root folder: basename + short hash of the path.
pub fn index_dir_for(root: &Path) -> PathBuf {
    let abs = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let key = if cfg!(windows) {
        abs.to_string_lossy().to_lowercase()
    } else {
        abs.to_string_lossy().into_owned()
    };
    let digest = Sha1::digest(key.as_bytes());
    let hash = hex::encode(&digest[..5]);
    let raw_name = abs
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "root".to_string());
    let mut base: String = raw_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .take(40)
        .collect();
    if base.is_empty() {
        base = "root".to_string();
    }
    data_dir().join("indexes").join(format!("{base}-{hash}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Tier {
    #[serde(rename = "P0")]
    P0,
    #[serde(rename = "P1")]
    P1,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::P0 => "P0",
            Tier::P1 => "P1",
        }
    }
}

/// P0 ("core"): source code and documentation.
pub const P0_EXTS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "rb", "php", "go", "java", "kt", "kts",
    "scala", "swift", "dart", "c", "h", "cc", "cpp", "hpp", "cs", "m", "mm", "lua", "sh", "bash",
    "zsh", "ps1", "psm1", "bat", "cmd", "sql", "graphql", "gql", "proto", "vue", "svelte", "astro",
    "gradle", "cmake", "md", "mdx", "r", "jl", "ex", "exs", "erl", "hrl", "clj", "cljs", "hs",
    "pl", "gemspec", "podspec",
];

/// P1 ("support"): everything else textual, processed after P0 and damped at search time.
pub const P1_EXTS: &[&str] = &[
    "json", "jsonc", "txt", "yml", "yaml", "toml", "ini", "cfg", "conf", "properties", "html",
    "htm", "css", "scss", "sass", "less", "xml", "xaml", "resx", "svg", "env", "gitignore",
    "gitattributes", "editorconfig", "npmrc", "nvmrc", "csproj", "sln", "props", "targets",
];

/// P1 ("support"): whole file names always classified P1.
pub const P1_NAMES: &[&str] = &[
    "dockerfile",
    "makefile",
    "license",
    "procfile",
    ".env",
    ".env.example",
    ".zcodeignore",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".npmrc",
    ".nvmrc",
];

/// Dependency-lock file names never worth embedding (anything named `*.lock`
/// is already excluded by the tier lists; these have other extensions or none).
pub const DEPENDENCY_FILES: &[&str] = &[
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

/// Classify a root-relative path into its processing tier; None means "not
/// indexable". Order: P1 names → extensions (P0 then P1) → P0 globs → P1 globs.
pub fn tier_for(rel_path: &str) -> Option<Tier> {
    tier_for_in(&settings().index, rel_path)
}

/// Pure variant of [`tier_for`] for tests and callers holding settings.
pub fn tier_for_in(idx: &IndexSettings, rel_path: &str) -> Option<Tier> {
    let lower = rel_path.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    if idx.p1_names.iter().any(|n| n == name) {
        return Some(Tier::P1);
    }
    let ext = name
        .rsplit_once('.')
        .filter(|(stem, _)| !stem.is_empty())
        .map(|(_, ext)| ext);
    if let Some(ext) = ext {
        if idx.p0_exts.iter().any(|e| e == ext) {
            return Some(Tier::P0);
        }
        if idx.p1_exts.iter().any(|e| e == ext) {
            return Some(Tier::P1);
        }
    }
    if glob_matches(&idx.p0_globs, &lower, name) {
        return Some(Tier::P0);
    }
    if glob_matches(&idx.p1_globs, &lower, name) {
        return Some(Tier::P1);
    }
    None
}

/// A pattern hits when it matches the root-relative path or the bare file
/// name (so `*.csv` reaches `sub/dir/x.csv` without a `**/` prefix).
fn glob_matches(patterns: &[String], path: &str, name: &str) -> bool {
    if patterns.is_empty() {
        return false;
    }
    let mut builder = globset::GlobSetBuilder::new();
    for p in patterns {
        if let Ok(glob) = globset::GlobBuilder::new(p).case_insensitive(true).build() {
            builder.add(glob);
        }
    }
    let Ok(set) = builder.build() else {
        return false;
    };
    set.is_match(path) || set.is_match(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers() {
        assert_eq!(tier_for("main.rs"), Some(Tier::P0));
        assert_eq!(tier_for("README.MD"), Some(Tier::P0));
        assert_eq!(tier_for("Component.tsx"), Some(Tier::P0));
        assert_eq!(tier_for("notes.md"), Some(Tier::P0));
        assert_eq!(tier_for("package.json"), Some(Tier::P1));
        assert_eq!(tier_for("Dockerfile"), Some(Tier::P1));
        assert_eq!(tier_for("config.example.json"), Some(Tier::P1));
        assert_eq!(tier_for("photo.jpg"), None);
        assert_eq!(tier_for("app.exe"), None);
        assert_eq!(tier_for(".gitignore"), Some(Tier::P1));
        // Dependency-tree files are excluded from the tier lists.
        assert_eq!(tier_for("Cargo.lock"), None);
        assert_eq!(tier_for("pnpm-lock.yaml"), Some(Tier::P1)); // skipped in files::walk
    }

    #[test]
    fn yaml_roundtrip() {
        let parsed: Settings =
            serde_yaml::from_str("provider: ollama\nindex:\n  batch_size: 8\n").unwrap();
        assert_eq!(parsed.provider, "ollama");
        assert_eq!(parsed.index.batch_size, 8);
        assert_eq!(parsed.index.concurrency, 3); // default kept
        assert_eq!(parsed.index.auto_refresh, true); // default kept
        assert_eq!(parsed.index.auto_refresh_interval_secs, 10);
        assert_eq!(parsed.search.p1_damp, 0.9);
    }

    #[test]
    fn auto_refresh_overrides() {
        let parsed: Settings = serde_yaml::from_str(
            "index:\n  auto_refresh: false\n  auto_refresh_interval_secs: 30\n",
        )
        .unwrap();
        assert!(!parsed.index.auto_refresh);
        assert_eq!(parsed.index.auto_refresh_interval_secs, 30);
        // Default settings keep auto-refresh on.
        assert!(Settings::default().index.auto_refresh);
    }

    #[test]
    fn custom_lists_replace_defaults() {
        let idx = IndexSettings { p0_exts: vec!["zig".into()], ..IndexSettings::default() };
        assert_eq!(tier_for_in(&idx, "main.zig"), Some(Tier::P0));
        // Defaults are gone once the list is replaced.
        assert_eq!(tier_for_in(&idx, "main.rs"), None);
        // Unrelated lists keep their defaults.
        assert_eq!(tier_for_in(&idx, "package.json"), Some(Tier::P1));
        assert_eq!(tier_for_in(&idx, "Dockerfile"), Some(Tier::P1));
        assert_eq!(tier_for_in(&idx, "cargo.lock"), None);
    }

    #[test]
    fn tier_globs() {
        let idx = IndexSettings {
            p0_globs: vec!["**/migrations/*.sql".into(), "*.proto".into()],
            p1_globs: vec!["data/**".into()],
            ..IndexSettings::default()
        };
        assert_eq!(tier_for_in(&idx, "db/migrations/003_init.sql"), Some(Tier::P0));
        assert_eq!(tier_for_in(&idx, "schema.proto"), Some(Tier::P0));
        // Bare-name matching: `*.proto` reaches nested files without `**/`.
        assert_eq!(tier_for_in(&idx, "sub/dir/api.proto"), Some(Tier::P0));
        assert_eq!(tier_for_in(&idx, "data/dump.csv"), Some(Tier::P1));
        // Case-insensitive.
        assert_eq!(tier_for_in(&idx, "Data/DUMP.CSV"), Some(Tier::P1));
        // Globs only rescue otherwise-unclassified files; extension rules win.
        assert_eq!(tier_for_in(&idx, "data/readme.md"), Some(Tier::P0));
        assert_eq!(tier_for_in(&idx, "photo.jpg"), None);
        assert_eq!(tier_for_in(&idx, "random.txt"), Some(Tier::P1));
    }

    #[test]
    fn dependency_files_are_configurable() {
        let mut idx = IndexSettings::default();
        assert!(!idx.dependency_files.is_empty());
        idx.dependency_files.clear();
        // Clearing the list stops the exclusion at the classification layer
        // (files::walk is what drops dependency files).
        assert_eq!(tier_for_in(&idx, "pnpm-lock.yaml"), Some(Tier::P1));
    }

    #[test]
    fn default_template_roundtrips() {
        let dir = std::env::temp_dir().join(format!("rag-config-tpl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.yaml");
        write_default_config(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let parsed: Settings = serde_yaml::from_str(&text).unwrap();
        assert_eq!(parsed.index.p0_exts, IndexSettings::default().p0_exts);
        assert_eq!(parsed.index.p1_names, IndexSettings::default().p1_names);
        assert_eq!(parsed.index.dependency_files, IndexSettings::default().dependency_files);
        assert!(parsed.index.p0_globs.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
