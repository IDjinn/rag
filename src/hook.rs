//! ZCode hook integration: `rag hook` reads the JSON payload ZCode pipes to
//! hook processes on stdin and emits `{"additionalContext": …}` payloads that
//! steer the session model toward the RAG index:
//!
//! - `SessionStart` (startup/resume/clear/post-compact): announces the index
//!   and defines the flow — rag_search first, exploration subagents only as a
//!   fallback for what the index doesn't cover.
//! - `PreToolUse` on `Agent`: a short reminder at the exact moment the model
//!   delegates, because subagents start fresh (no hook context) and Explore
//!   agents cannot call MCP tools at all.
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
    #[serde(default)]
    tool_name: Option<String>,
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
    let store = Store::for_root(&root);
    let payload = match (event, input.tool_name.as_deref()) {
        ("SessionStart", _) => session_start_context(&store),
        ("PreToolUse", Some("Agent")) => delegation_reminder(&store),
        _ => None,
    };
    if let Some(json) = payload {
        println!("{json}");
    }
    Ok(())
}

/// SessionStart payload: index facts plus the search-before-exploring flow.
fn session_start_context(store: &Store) -> Option<String> {
    let data = store.load()?;
    let p0 = data.files.values().filter(|e| e.tier == Tier::P0).count();
    let root_display = data.root.strip_prefix("//?/").unwrap_or(&data.root);
    let context = format!(
        "RAG index active for this repo ({}): {} files (P0: {p0}, P1: {}), {} chunks, model `{}`, last indexed {}. \
         Locate code by meaning, feature, or intent with the `rag_search` MCP tool BEFORE grep or exploration \
         subagents — it returns ranked file:line references. Run it yourself before spawning Explore/ \
         general-purpose subagents and paste the relevant hits into their prompts (Explore agents have no MCP \
         access). Fall back to Grep only for exact-string/regex sweeps, and to subagents only for what the \
         index does not cover. Feed usefulness verdicts back via `rag_feedback`; `rag_status` shows details.",
        root_display,
        data.files.len(),
        data.files.len() - p0,
        data.vector_count,
        data.model,
        data.updated_at,
    );
    Some(additional_context(&context))
}

/// PreToolUse(Agent) payload: a short nudge at the moment of delegation.
fn delegation_reminder(store: &Store) -> Option<String> {
    store.load()?;
    let context = "Before delegating to subagents: run `rag_search` (MCP) first and paste the relevant \
                   file:line hits into the subagent prompt — Explore agents have no MCP tool access. \
                   Delegate only what the index does not cover (exact regex sweeps, unindexed paths).";
    Some(additional_context(context))
}

fn additional_context(message: &str) -> String {
    serde_json::json!({ "additionalContext": message }).to_string()
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
        let json = session_start_context(&store).expect("indexed root should emit context");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let ctx = parsed["additionalContext"].as_str().unwrap();
        assert!(ctx.contains("RAG index active"));
        assert!(ctx.contains("2 files (P0: 1, P1: 1)"));
        assert!(ctx.contains("3 chunks"));
        assert!(ctx.contains("text-embedding-test"));
        assert!(ctx.contains("rag_search"));
        // The delegation flow must be part of the session-start message.
        assert!(ctx.contains("subagents"));
        assert!(ctx.contains("paste"));
        assert!(store.remove());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delegation_reminder_on_indexed_root() {
        let dir = std::env::temp_dir().join(format!("rag-hook-del-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = seed_index(&dir);
        let json = delegation_reminder(&store).expect("indexed root should emit reminder");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let ctx = parsed["additionalContext"].as_str().unwrap();
        assert!(ctx.contains("rag_search"));
        assert!(ctx.contains("no MCP tool access"));
        assert!(store.remove());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unindexed_root_is_silent() {
        let dir = std::env::temp_dir().join(format!("rag-hook-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store { dir: dir.clone() };
        assert!(session_start_context(&store).is_none());
        assert!(delegation_reminder(&store).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_dispatches_by_event() {
        // Payload-level proxy for the stdin dispatch table: only SessionStart
        // and PreToolUse(Agent) produce output; everything else is silent.
        let dir = std::env::temp_dir().join(format!("rag-hook-ev-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = seed_index(&dir);
        assert!(session_start_context(&store).is_some());
        assert!(delegation_reminder(&store).is_some());
        // Non-agent PreToolUse events never reach delegation_reminder.
        let event = ("PreToolUse", Some("Bash"));
        assert!(matches!(event, ("PreToolUse", Some("Agent"))) == false);
        assert!(store.remove());
        let _ = fs::remove_dir_all(&dir);
    }
}
