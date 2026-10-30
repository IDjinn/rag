//! Project modules: path rules (exclusion and tier priority) that switch on when
//! a marker file is present at the root. Rag ships a preset per widely used
//! language (dependency, cache and build output trees) and loads more from
//! plugins: one YAML file per module in `<data dir>/modules/`, or entries in the
//! `index.modules` list of config.yaml. A plugin with the same name as a preset
//! replaces it; `index.disabled_modules` switches presets off.

use crate::config::{self, IndexSettings, Tier};
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(default)]
pub struct Module {
    pub name: String,
    /// Markers that activate the module. A plain path (`Cargo.toml`) must exist
    /// at the root; a pattern with `*`, `?` or `[` matches root entry names
    /// (`*.csproj`).
    pub detect: Vec<String>,
    /// Glob patterns never indexed while the module is active.
    pub exclude: Vec<String>,
    /// Glob patterns forced to P0 while the module is active.
    pub p0: Vec<String>,
    /// Glob patterns forced to P1 while the module is active.
    pub p1: Vec<String>,
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn preset(name: &str, detect: &[&str], exclude: &[&str]) -> Module {
    Module {
        name: name.to_string(),
        detect: strings(detect),
        exclude: strings(exclude),
        ..Module::default()
    }
}

/// Built-in presets for the ten most used languages: dependencies, caches,
/// build output and vendored libraries. Each is inert in a project without its
/// marker, so a mixed repository gets every preset that applies.
pub fn builtin_modules() -> Vec<Module> {
    vec![
        preset("rust", &["Cargo.toml"], &["**/target/**"]),
        preset(
            "python",
            &["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
            &[
                "**/.venv/**",
                "**/venv/**",
                "**/site-packages/**",
                "**/__pycache__/**",
                "**/*.egg-info/**",
                ".tox/**",
                ".nox/**",
                ".mypy_cache/**",
                ".ruff_cache/**",
                ".pytest_cache/**",
                ".ipynb_checkpoints/**",
                "build/**",
                "dist/**",
            ],
        ),
        preset(
            "node",
            &["package.json"],
            &[
                "**/node_modules/**",
                "**/bower_components/**",
                "dist/**",
                "build/**",
                ".next/**",
                ".nuxt/**",
                ".svelte-kit/**",
                ".output/**",
                "coverage/**",
                "**/*.min.js",
                "**/*.min.css",
                "**/*.map",
            ],
        ),
        preset(
            "java",
            &["pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"],
            &["**/target/**", "build/**", "out/**", ".gradle/**", "bin/**"],
        ),
        preset(
            "csharp",
            &["*.sln", "*.csproj", "*.fsproj", "*.vbproj"],
            &["**/bin/**", "**/obj/**", "packages/**", ".vs/**", "TestResults/**"],
        ),
        preset(
            "cmake",
            &["CMakeLists.txt"],
            &[
                "build/**",
                "build-*/**",
                "cmake-build-*/**",
                "_deps/**",
                "**/_deps/**",
                "third_party/**",
                "thirdparty/**",
                "3rdparty/**",
                "external/**",
                "extern/**",
                "vendor/**",
                "vcpkg_installed/**",
                "conan/**",
            ],
        ),
        preset(
            "make",
            &["Makefile", "makefile", "configure", "configure.ac", "meson.build"],
            &[
                "build/**",
                "out/**",
                "bin/**",
                "obj/**",
                "autom4te.cache/**",
                "**/.deps/**",
                "**/.libs/**",
                "**/*.dSYM/**",
                "third_party/**",
                "thirdparty/**",
                "vendor/**",
            ],
        ),
        preset("go", &["go.mod"], &["vendor/**", "bin/**"]),
        preset(
            "php",
            &["composer.json"],
            &["vendor/**", "bootstrap/cache/**", "storage/framework/cache/**", "public/build/**"],
        ),
        preset(
            "ruby",
            &["Gemfile", "*.gemspec"],
            &["vendor/bundle/**", "vendor/cache/**", ".bundle/**", "tmp/**", "log/**", "coverage/**", "**/.yardoc/**"],
        ),
    ]
}

/// Directory holding plugin modules: one `*.yaml` (or `.yml`) file per module.
pub fn plugin_dir() -> PathBuf {
    config::data_dir().join("modules")
}

fn load_plugins(dir: &Path) -> Vec<Module> {
    let Ok(read) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| matches!(p.extension().and_then(|x| x.to_str()), Some("yaml" | "yml")))
        .collect();
    paths.sort();
    let mut out = Vec::new();
    for path in paths {
        let parsed = std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|text| serde_yaml::from_str::<Module>(&text).map_err(|e| e.to_string()));
        match parsed {
            Ok(m) if !m.name.is_empty() => out.push(m),
            Ok(_) => eprintln!("[rag] ignoring module {}: missing `name`", path.display()),
            Err(e) => eprintln!("[rag] ignoring module {}: {e}", path.display()),
        }
    }
    out
}

/// Every module that may apply: presets, then plugin files, then config entries.
/// A later entry with the same name replaces an earlier one.
pub fn resolve(idx: &IndexSettings) -> Vec<Module> {
    let mut mods = builtin_modules();
    for m in load_plugins(&plugin_dir()).into_iter().chain(idx.modules.iter().cloned()) {
        match mods.iter().position(|b| b.name == m.name) {
            Some(i) => mods[i] = m,
            None => mods.push(m),
        }
    }
    mods.retain(|m| !idx.disabled_modules.contains(&m.name));
    mods
}

/// Whether any of a module's markers is present at the root.
fn detected(root: &Path, markers: &[String]) -> bool {
    markers.iter().any(|marker| {
        if marker.contains(['*', '?', '[']) {
            let Ok(glob) = GlobBuilder::new(marker).case_insensitive(true).build() else {
                return false;
            };
            let matcher = glob.compile_matcher();
            std::fs::read_dir(root)
                .map(|read| read.flatten().any(|e| matcher.is_match(e.file_name())))
                .unwrap_or(false)
        } else {
            root.join(marker).is_file()
        }
    })
}

/// Compiled rules for one root: global exclusions plus those of every module
/// detected there.
pub struct Rules {
    exclude: Option<GlobSet>,
    p0: Option<GlobSet>,
    p1: Option<GlobSet>,
    /// Names of the modules active for this root (for reports).
    pub active: Vec<String>,
}

impl Rules {
    /// Rules for a root with the modules the configuration resolves to.
    pub fn for_root(idx: &IndexSettings, root: &Path) -> Self {
        Self::build(&idx.exclude_globs, &resolve(idx), root)
    }

    pub fn build(global_exclude: &[String], modules: &[Module], root: &Path) -> Self {
        let active: Vec<&Module> = modules.iter().filter(|m| detected(root, &m.detect)).collect();
        let mut exclude: Vec<String> = global_exclude.to_vec();
        let mut p0: Vec<String> = Vec::new();
        let mut p1: Vec<String> = Vec::new();
        for m in &active {
            exclude.extend(m.exclude.iter().cloned());
            p0.extend(m.p0.iter().cloned());
            p1.extend(m.p1.iter().cloned());
        }
        Self {
            exclude: build_set(&exclude),
            p0: build_set(&p0),
            p1: build_set(&p1),
            active: active.iter().map(|m| m.name.clone()).collect(),
        }
    }

    /// Whether a root-relative path (file or directory) is excluded. Directories
    /// are checked with a trailing slash so `build/**` prunes the whole tree.
    pub fn excluded(&self, rel: &str, is_dir: bool) -> bool {
        let Some(set) = &self.exclude else { return false };
        if is_dir {
            set.is_match(format!("{rel}/"))
        } else {
            hits(set, rel)
        }
    }

    /// Tier forced by an active module (P0 wins over P1); None leaves the
    /// decision to the extension rules.
    pub fn tier_override(&self, rel: &str) -> Option<Tier> {
        if self.p0.as_ref().is_some_and(|s| hits(s, rel)) {
            return Some(Tier::P0);
        }
        if self.p1.as_ref().is_some_and(|s| hits(s, rel)) {
            return Some(Tier::P1);
        }
        None
    }
}

fn build_set(patterns: &[String]) -> Option<GlobSet> {
    if patterns.is_empty() {
        return None;
    }
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        if let Ok(glob) = GlobBuilder::new(p).case_insensitive(true).build() {
            builder.add(glob);
        }
    }
    builder.build().ok()
}

/// Same convention as config globs: the root-relative path or the bare name.
fn hits(set: &GlobSet, rel: &str) -> bool {
    let name = rel.rsplit('/').next().unwrap_or(rel);
    set.is_match(rel) || set.is_match(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rag-modules-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn presets_cover_ten_languages() {
        let names: Vec<String> = builtin_modules().into_iter().map(|m| m.name).collect();
        assert_eq!(names.len(), 10);
        for lang in ["rust", "python", "node", "java", "csharp", "cmake", "make", "go", "php", "ruby"] {
            assert!(names.iter().any(|n| n == lang), "missing {lang}");
        }
    }

    #[test]
    fn presets_activate_only_with_their_marker() {
        let dir = temp_root("plain");
        let idx = IndexSettings::default();
        assert!(Rules::for_root(&idx, &dir).active.is_empty());
        std::fs::write(dir.join("Cargo.toml"), "[package]").unwrap();
        assert_eq!(Rules::for_root(&idx, &dir).active, vec!["rust".to_string()]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rust_and_node_trees_are_excluded() {
        let dir = temp_root("lang");
        std::fs::write(dir.join("Cargo.toml"), "").unwrap();
        std::fs::write(dir.join("package.json"), "{}").unwrap();
        let rules = Rules::for_root(&IndexSettings::default(), &dir);
        assert!(rules.excluded("target", true));
        assert!(rules.excluded("target/debug/build/x.rs", false));
        assert!(rules.excluded("node_modules", true));
        assert!(rules.excluded("web/node_modules/react/index.js", false));
        assert!(rules.excluded("public/app.min.js", false));
        assert!(!rules.excluded("src/main.rs", false));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn csharp_detects_by_pattern() {
        let dir = temp_root("csharp");
        std::fs::write(dir.join("App.csproj"), "<Project/>").unwrap();
        let rules = Rules::for_root(&IndexSettings::default(), &dir);
        assert_eq!(rules.active, vec!["csharp".to_string()]);
        assert!(rules.excluded("src/bin/Debug/app.dll", false));
        assert!(rules.excluded("src/obj", true));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cmake_module_excludes_vendored_libraries_and_build_trees() {
        let dir = temp_root("cmake");
        std::fs::write(dir.join("CMakeLists.txt"), "project(x)").unwrap();
        let rules = Rules::for_root(&IndexSettings::default(), &dir);
        assert!(rules.excluded("third_party/zlib/zlib.c", false));
        assert!(rules.excluded("build/CMakeCache.txt", false));
        assert!(rules.excluded("cmake-build-debug/x.cpp", false));
        assert!(rules.excluded("build/_deps/fmt-src/format.h", false));
        assert!(rules.excluded("vendor/json/json.hpp", false));
        assert!(!rules.excluded("src/main.cpp", false));
        assert!(rules.excluded("third_party", true));
        assert!(!rules.excluded("src", true));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn config_disables_and_overrides_presets() {
        let dir = temp_root("config");
        std::fs::write(dir.join("CMakeLists.txt"), "project(x)").unwrap();
        let idx = IndexSettings {
            disabled_modules: vec!["cmake".into()],
            ..IndexSettings::default()
        };
        assert!(Rules::for_root(&idx, &dir).active.is_empty());

        let idx = IndexSettings {
            modules: vec![Module {
                name: "cmake".into(),
                detect: vec!["CMakeLists.txt".into()],
                exclude: vec!["gen/**".into()],
                ..Module::default()
            }],
            ..IndexSettings::default()
        };
        let rules = Rules::for_root(&idx, &dir);
        assert!(rules.excluded("gen/x.cpp", false));
        assert!(!rules.excluded("third_party/zlib.c", false)); // preset replaced
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn module_priority_overrides_extension_rules() {
        let dir = temp_root("prio");
        std::fs::write(dir.join("CMakeLists.txt"), "project(x)").unwrap();
        let modules = vec![Module {
            name: "docs".into(),
            detect: vec!["CMakeLists.txt".into()],
            p1: vec!["docs/**".into()],
            ..Module::default()
        }];
        let rules = Rules::build(&[], &modules, &dir);
        assert_eq!(rules.tier_override("docs/guide.md"), Some(Tier::P1));
        assert_eq!(rules.tier_override("src/a.cpp"), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn global_exclude_applies_without_modules() {
        let dir = temp_root("global");
        let rules = Rules::build(&["**/generated/**".into(), "*.min.js".into()], &[], &dir);
        assert!(rules.excluded("web/generated/api.ts", false));
        assert!(rules.excluded("static/app.min.js", false));
        assert!(!rules.excluded("web/app.js", false));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
