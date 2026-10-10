//! ZCode hook integration: `rag hook` reads the JSON payload ZCode pipes to
//! hook processes on stdin and, for `SessionStart` in an indexed root, prints
//! `{"additionalContext": …}` so the session model learns the RAG index is
//! there and reaches for `rag_search` instead of plain text search.
//!
//! Strictly advisory: unknown event, parse failure, or missing index prints
//! nothing. Non-JSON stdout is a no-op for ZCode, while a `{`-prefixed body
//! must validate against its output schema — so "print nothing" is the only
//! safe silence, and the process must always exit 0.

use serde::Deserialize;
use std::io::Read;
use std::path::PathBuf;

use crate::config::Tier;
use crate::store::Store;

/// Subset of ZCode's Claude-compatible hook stdin payload (snake_case,
/// unknown fields ignored).
#[derive(Deserialize, Default)]
struct HookInput {
    #[serde(default)]
    hook_event_name: Option<String>,
    #[serde(default)]
    cwd: Option<String>,
}

/// Reads the hook payload from stdin and prints the additionalContext JSON
/// when applicable. Any failure is silent by design — a hook must never
/// surface errors into the session.
pub fn run() -> std::io::Result<()> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let Ok(input) = serde_json::from_str::<HookInput>(buf.trim()) else {
        return Ok(());
    };
    let event = input.hook_event_name.as_deref().unwrap_or_default();
    let root = input
        .cwd
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if let Some(json) = hook_context(&Store::for_root(&root), event) {
        println!("{json}");
    }
    Ok(())
}

/// `Some({"additionalContext": …})` when `event` is `SessionStart` and the
/// store's root has a live index, otherwise `None`.
fn hook_context(store: &Store, event: &str) -> Option<String> {
    if event != "SessionStart" {
        return None;
    }
    let data = store.load()?;
    let p0 = data.files.values().filter(|e| e.tier == Tier::P0).count();
    let root_display = data.root.strip_prefix("//?/").unwrap_or(&data.root);
    let context = format!(
        "RAG index active for this repo ({}): {} files (P0: {p0}, P1: {}), {} chunks, model `{}`, last indexed {}. \
         Prefer the `rag_search` MCP tool over Grep/Glob when locating code by meaning, feature, or intent — \
         it returns ranked file:line references with snippets, and usefulness verdicts can be fed back via \
         `rag_feedback`. `rag_status` shows full details.",
        root_display,
        data.files.len(),
        data.files.len() - p0,
        data.vector_count,
        data.model,
        data.updated_at,
    );
    Some(serde_json::json!({ "additionalContext": context }).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{ChunkRef, FileEntry, IndexData};
    use std::fs;

    fn seed_index(dir: &PathBuf) -> Store {
        let store = Store { dir: dir.clone() };
        let mut data = IndexData::empty("C:/tmp/root", "text-embedding-test");
        data.dim = 4;
        data.files.insert(
            "a.rs".into(),
            FileEntry {
                hash: "h1".into(),
                size: 10,
                mtime_ms: 1,
                chunk_count: 2,
                vec_offset: 0,
                chunks: vec![ChunkRef { start: 1, end: 5 }, ChunkRef { start: 5, end: 9 }],
                tier: Tier::P0,
            },
        );
        data.files.insert(
            "b.md".into(),
            FileEntry {
                hash: "h2".into(),
                size: 8,
                mtime_ms: 2,
                chunk_count: 1,
                vec_offset: 2,
                chunks: vec![ChunkRef { start: 0, end: 8 }],
                tier: Tier::P1,
            },
        );
        data.vector_count = 3;
        store.save(&data, &[0.0; 12]).unwrap();
        store
    }

    #[test]
    fn session_start_on_indexed_root_emits_context() {
        let dir = std::env::temp_dir().join(format!("rag-hook-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = seed_index(&dir);
        let json = hook_context(&store, "SessionStart").expect("indexed root should emit context");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let ctx = parsed["additionalContext"].as_str().unwrap();
        assert!(ctx.contains("RAG index active"));
        assert!(ctx.contains("2 files (P0: 1, P1: 1)"));
        assert!(ctx.contains("3 chunks"));
        assert!(ctx.contains("text-embedding-test"));
        assert!(ctx.contains("rag_search"));
        assert!(store.remove());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unindexed_root_is_silent() {
        let dir = std::env::temp_dir().join(format!("rag-hook-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store { dir: dir.clone() };
        assert!(hook_context(&store, "SessionStart").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_session_start_events_are_silent() {
        let dir = std::env::temp_dir().join(format!("rag-hook-ev-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = seed_index(&dir);
        assert!(hook_context(&store, "UserPromptSubmit").is_none());
        assert!(hook_context(&store, "PreToolUse").is_none());
        assert!(hook_context(&store, "").is_none());
        assert!(store.remove());
        let _ = fs::remove_dir_all(&dir);
    }
}
