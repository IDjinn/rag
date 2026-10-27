//! `rag install`: registers the MCP server (and hooks / usage instructions,
//! where the agent supports them) in every coding agent found on the machine.
//!
//! | agent    | MCP server                          | hooks                         | instructions        |
//! |----------|-------------------------------------|-------------------------------|---------------------|
//! | claude   | `~/.claude.json` `mcpServers`       | `~/.claude/settings.json`     | —                   |
//! | zcode    | `~/.zcode/cli/config.json`          | same file                     | —                   |
//! | opencode | `~/.config/opencode/opencode.json`  | none (no shell hooks)         | `AGENTS.md` block   |
//! | codex    | `~/.codex/config.toml`              | none (no SessionStart hook)   | `AGENTS.md` block   |
//!
//! Every edit is idempotent, preserves unrelated settings, and keeps a
//! one-time `*.rag.bak` copy of the original. A file that fails to parse is
//! reported and left untouched.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};
use std::fs;
use std::path::{Path, PathBuf};
use toml_edit::{value, Array, DocumentMut, Item, Table};

const BLOCK_BEGIN: &str = "<!-- rag:begin -->";
const BLOCK_END: &str = "<!-- rag:end -->";

const INSTRUCTIONS: &str = "## Code search (rag)\n\n\
The `rag` MCP server indexes this repo for semantic search. Use the `rag_search` tool to locate code by \
meaning, feature or intent BEFORE grep or exploration subagents; it returns ranked file:line references. \
Fall back to grep only for exact-string/regex sweeps. If the repo is not indexed yet, run `rag_index` once. \
Report whether results helped via `rag_feedback`.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Agent {
    Claude,
    ZCode,
    OpenCode,
    Codex,
}

impl Agent {
    pub const ALL: [Agent; 4] = [Agent::Claude, Agent::ZCode, Agent::OpenCode, Agent::Codex];

    pub fn name(self) -> &'static str {
        match self {
            Agent::Claude => "claude",
            Agent::ZCode => "zcode",
            Agent::OpenCode => "opencode",
            Agent::Codex => "codex",
        }
    }

    fn parse(s: &str) -> Option<Agent> {
        Agent::ALL.into_iter().find(|a| a.name() == s.to_lowercase())
    }

    /// Directory whose existence means the agent is installed.
    fn detect_dir(self, home: &Path) -> PathBuf {
        match self {
            Agent::Claude => home.join(".claude"),
            Agent::ZCode => home.join(".zcode").join("cli"),
            Agent::OpenCode => opencode_dir(home),
            Agent::Codex => codex_dir(home),
        }
    }
}

fn opencode_dir(home: &Path) -> PathBuf {
    home.join(".config").join("opencode")
}

fn codex_dir(home: &Path) -> PathBuf {
    std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".codex"))
}

pub struct InstallOpts {
    /// Agent names, or `all`; empty = every agent detected on this machine.
    pub targets: Vec<String>,
    pub dry_run: bool,
    pub uninstall: bool,
    /// Skip hooks/instructions and register only the MCP server.
    pub mcp_only: bool,
    /// Global = user-level configs in $HOME; Local = this project (cwd).
    pub scope: Scope,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    Global,
    Local,
}

pub fn run(opts: InstallOpts) -> Result<String> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("cannot resolve the home directory"))?;
    let exe = std::env::current_exe()
        .context("cannot resolve the rag executable path")?
        .to_string_lossy()
        .trim_start_matches(r"\\?\")
        .to_string();
    let cwd = std::env::current_dir().context("cannot resolve the current directory")?;
    let agents = select(&opts.targets, &home)?;
    if agents.is_empty() {
        return Ok("No supported agent detected (claude, zcode, opencode, codex). \
                   Name one explicitly, e.g. `rag install claude`, to create its config."
            .into());
    }
    let verb = match (opts.uninstall, opts.dry_run) {
        (false, false) => "Installed rag into",
        (false, true) => "Would install rag into",
        (true, false) => "Removed rag from",
        (true, true) => "Would remove rag from",
    };
    let mut out = format!("# rag install\n\n{verb}: {}\n\n", names(&agents));
    for agent in agents {
        out.push_str(&format!("## {}\n", agent.name()));
        match apply(agent, &home, &cwd, &exe, &opts) {
            Ok(lines) => lines.iter().for_each(|l| out.push_str(&format!("- {l}\n"))),
            Err(err) => out.push_str(&format!("- ERROR: {err:#}\n")),
        }
        out.push('\n');
    }
    if opts.scope == Scope::Local && !opts.uninstall {
        out.push_str(
            "Local configs embed this machine's absolute path to `rag`; \
             check them in only if teammates share that path.\n\n",
        );
    }
    if !opts.uninstall && !opts.dry_run {
        out.push_str("Restart the agents to load the new MCP server. Then index a repo: `rag index .`\n");
    }
    Ok(out)
}

fn names(agents: &[Agent]) -> String {
    agents.iter().map(|a| a.name()).collect::<Vec<_>>().join(", ")
}

fn select(targets: &[String], home: &Path) -> Result<Vec<Agent>> {
    if targets.is_empty() {
        return Ok(Agent::ALL.into_iter().filter(|a| a.detect_dir(home).is_dir()).collect());
    }
    let mut out = Vec::new();
    for t in targets {
        if t.eq_ignore_ascii_case("all") {
            return Ok(Agent::ALL.to_vec());
        }
        let agent = Agent::parse(t).ok_or_else(|| {
            anyhow!("unknown target `{t}` (expected: claude, zcode, opencode, codex, all)")
        })?;
        if !out.contains(&agent) {
            out.push(agent);
        }
    }
    Ok(out)
}

/// Config file locations for one agent at one scope.
struct Paths {
    mcp: PathBuf,
    /// `None` when the agent has no shell-hook support.
    hooks: Option<PathBuf>,
    /// `None` when the agent has no instructions file.
    instructions: Option<PathBuf>,
}

fn paths(agent: Agent, scope: Scope, home: &Path, cwd: &Path) -> Paths {
    let global = scope == Scope::Global;
    match agent {
        Agent::Claude if global => Paths {
            mcp: home.join(".claude.json"),
            hooks: Some(home.join(".claude").join("settings.json")),
            instructions: None,
        },
        // settings.local.json is gitignored by Claude Code, so the absolute
        // binary path never lands in version control.
        Agent::Claude => Paths {
            mcp: cwd.join(".mcp.json"),
            hooks: Some(cwd.join(".claude").join("settings.local.json")),
            instructions: None,
        },
        Agent::ZCode => {
            let base = if global { home.to_path_buf() } else { cwd.to_path_buf() };
            let file = base.join(".zcode").join("cli").join("config.json");
            Paths { mcp: file.clone(), hooks: Some(file), instructions: None }
        }
        Agent::OpenCode if global => {
            let dir = opencode_dir(home);
            Paths { mcp: dir.join("opencode.json"), hooks: None, instructions: Some(dir.join("AGENTS.md")) }
        }
        Agent::OpenCode => Paths {
            mcp: cwd.join("opencode.json"),
            hooks: None,
            instructions: Some(cwd.join("AGENTS.md")),
        },
        Agent::Codex if global => {
            let dir = codex_dir(home);
            Paths { mcp: dir.join("config.toml"), hooks: None, instructions: Some(dir.join("AGENTS.md")) }
        }
        Agent::Codex => Paths {
            mcp: cwd.join(".codex").join("config.toml"),
            hooks: None,
            instructions: Some(cwd.join("AGENTS.md")),
        },
    }
}

fn apply(agent: Agent, home: &Path, cwd: &Path, exe: &str, opts: &InstallOpts) -> Result<Vec<String>> {
    let install = !opts.uninstall;
    let p = paths(agent, opts.scope, home, cwd);
    let mut lines = Vec::new();
    let mut edit = |path: &Path, what: &str, f: &dyn Fn(&str) -> Result<String>| -> Result<()> {
        let changed = edit_file(path, opts.dry_run, f)?;
        let state = if changed { if opts.dry_run { "would update" } else { "updated" } } else { "already up to date" };
        lines.push(format!("{what}: {state} (`{}`)", path.display()));
        Ok(())
    };
    match agent {
        Agent::Claude => edit(&p.mcp, "MCP server", &|s| {
            json_edit(s, |v| set_mcp(v, "mcpServers", claude_mcp(exe), install))
        })?,
        Agent::ZCode => edit(&p.mcp, "MCP server", &|s| {
            json_edit(s, |v| set_mcp(v, "mcpServers", json!({ "command": exe, "args": ["serve"] }), install))
        })?,
        Agent::OpenCode => edit(&p.mcp, "MCP server", &|s| {
            json_edit(s, |v| {
                set_mcp(v, "mcp", json!({ "type": "local", "command": [exe, "serve"], "enabled": true }), install)
            })
        })?,
        Agent::Codex => edit(&p.mcp, "MCP server", &|s| codex_edit(s, exe, install))?,
    }
    if !opts.mcp_only {
        if let Some(path) = &p.hooks {
            let cmd = hook_command(exe, agent == Agent::Claude);
            edit(path, "hooks", &|s| json_edit(s, |v| set_hooks(v, &cmd, install)))?;
        }
        if let Some(path) = &p.instructions {
            edit(path, "instructions", &|s| Ok(set_block(s, install)))?;
        }
    }
    Ok(lines)
}

/// Reads `path` (missing = empty), applies `f`, and writes back when the
/// text changed. The first modification of a file keeps `<file>.rag.bak`.
fn edit_file(path: &Path, dry_run: bool, f: &dyn Fn(&str) -> Result<String>) -> Result<bool> {
    let old = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let new = f(&old).with_context(|| format!("{} left untouched", path.display()))?;
    if new == old {
        return Ok(false);
    }
    if dry_run {
        return Ok(true);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if !old.is_empty() {
        let mut bak = path.as_os_str().to_owned();
        bak.push(".rag.bak");
        let bak = PathBuf::from(bak);
        if !bak.exists() {
            fs::write(&bak, &old)?;
        }
    }
    fs::write(path, new).with_context(|| format!("writing {}", path.display()))?;
    Ok(true)
}

/// Parses JSON text (empty = `{}`), mutates it, and pretty-prints it back.
/// Returns the input unchanged when the mutation was a no-op, so untouched
/// files keep their exact formatting.
fn json_edit(text: &str, f: impl FnOnce(&mut Value) -> bool) -> Result<String> {
    let mut root: Value = if text.trim().is_empty() {
        json!({})
    } else {
        serde_json::from_str(text).context("not valid JSON")?
    };
    if !root.is_object() {
        bail!("top-level value is not a JSON object");
    }
    if !f(&mut root) {
        return Ok(text.to_string());
    }
    Ok(serde_json::to_string_pretty(&root)? + "\n")
}

fn claude_mcp(exe: &str) -> Value {
    json!({ "type": "stdio", "command": exe, "args": ["serve"], "env": {} })
}

fn hook_command(exe: &str, claude: bool) -> String {
    let flag = if claude { " --format claude" } else { "" };
    format!("\"{exe}\" hook{flag}")
}

fn object<'a>(v: &'a mut Value, key: &str) -> &'a mut Map<String, Value> {
    let map = v.as_object_mut().expect("checked object");
    let slot = map.entry(key).or_insert_with(|| json!({}));
    if !slot.is_object() {
        *slot = json!({});
    }
    slot.as_object_mut().unwrap()
}

/// Upserts (or removes) `<section>.rag`. Returns whether anything changed.
fn set_mcp(root: &mut Value, section: &str, entry: Value, install: bool) -> bool {
    let servers = object(root, section);
    if install {
        if servers.get("rag") == Some(&entry) {
            return false;
        }
        servers.insert("rag".into(), entry);
        true
    } else {
        let removed = servers.remove("rag").is_some();
        if servers.is_empty() {
            root.as_object_mut().unwrap().remove(section);
        }
        removed
    }
}

fn is_rag_hook(cmd: &str) -> bool {
    let c = cmd.to_lowercase();
    c.contains("rag") && c.split_whitespace().any(|w| w == "hook")
}

/// (event, matcher) pairs rag registers.
const HOOK_EVENTS: [(&str, Option<&str>); 2] =
    [("SessionStart", None), ("PreToolUse", Some("Agent|Task"))];

/// Drops our hook commands from `entries`; empties entries are removed.
fn strip_ours(entries: &mut Vec<Value>) -> bool {
    let before = entries.clone();
    for entry in entries.iter_mut() {
        if let Some(hooks) = entry.get_mut("hooks").and_then(Value::as_array_mut) {
            hooks.retain(|h| !h.get("command").and_then(Value::as_str).is_some_and(is_rag_hook));
        }
    }
    entries.retain(|e| e.get("hooks").and_then(Value::as_array).is_none_or(|h| !h.is_empty()));
    *entries != before
}

fn set_hooks(root: &mut Value, command: &str, install: bool) -> bool {
    let hooks = object(root, "hooks");
    let mut changed = false;
    for (event, matcher) in HOOK_EVENTS {
        let slot = hooks.entry(event).or_insert_with(|| json!([]));
        if !slot.is_array() {
            *slot = json!([]);
        }
        let entries = slot.as_array_mut().unwrap();
        let mut ours = json!({ "hooks": [{ "type": "command", "command": command }] });
        if let Some(m) = matcher {
            ours["matcher"] = json!(m);
        }
        if install {
            if entries.contains(&ours) {
                continue;
            }
            strip_ours(entries);
            entries.push(ours);
            changed = true;
        } else {
            changed |= strip_ours(entries);
        }
        if entries.is_empty() {
            hooks.remove(event);
        }
    }
    if hooks.is_empty() {
        root.as_object_mut().unwrap().remove("hooks");
    }
    changed
}

fn codex_edit(text: &str, exe: &str, install: bool) -> Result<String> {
    let mut doc: DocumentMut = text.parse().context("not valid TOML")?;
    if install {
        let servers = doc.entry("mcp_servers").or_insert_with(|| {
            let mut t = Table::new();
            t.set_implicit(true);
            Item::Table(t)
        });
        let servers = servers.as_table_mut().ok_or_else(|| anyhow!("`mcp_servers` is not a table"))?;
        let mut rag = Table::new();
        rag["command"] = value(exe);
        let mut args = Array::new();
        args.push("serve");
        rag["args"] = value(args);
        let unchanged = servers.get("rag").is_some_and(|i| i.to_string().trim() == rag.to_string().trim());
        if unchanged {
            return Ok(text.to_string());
        }
        servers.insert("rag", Item::Table(rag));
    } else {
        let Some(servers) = doc.get_mut("mcp_servers").and_then(Item::as_table_mut) else {
            return Ok(text.to_string());
        };
        if servers.remove("rag").is_none() {
            return Ok(text.to_string());
        }
        if servers.is_empty() {
            doc.remove("mcp_servers");
        }
    }
    Ok(doc.to_string())
}

/// Inserts, replaces, or removes the marker-delimited instruction block.
fn set_block(text: &str, install: bool) -> String {
    let block = format!("{BLOCK_BEGIN}\n{INSTRUCTIONS}\n{BLOCK_END}");
    let span = text.find(BLOCK_BEGIN).and_then(|s| {
        text[s..].find(BLOCK_END).map(|e| (s, s + e + BLOCK_END.len()))
    });
    match (span, install) {
        (Some((s, e)), true) => format!("{}{block}{}", &text[..s], &text[e..]),
        (None, true) => {
            let sep = if text.is_empty() { "" } else if text.ends_with("\n\n") { "" } else if text.ends_with('\n') { "\n" } else { "\n\n" };
            format!("{text}{sep}{block}\n")
        }
        (Some((s, e)), false) => {
            let head = text[..s].trim_end_matches('\n');
            let tail = text[e..].trim_start_matches('\n');
            match (head.is_empty(), tail.is_empty()) {
                (true, true) => String::new(),
                (true, false) => format!("{tail}"),
                (false, true) => format!("{head}\n"),
                (false, false) => format!("{head}\n\n{tail}"),
            }
        }
        (None, false) => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXE: &str = r"C:\tools\rag.exe";

    #[test]
    fn mcp_install_is_idempotent_and_preserves_other_keys() {
        let src = r#"{"theme":"dark","mcpServers":{"other":{"command":"x"}}}"#;
        let once = json_edit(src, |v| set_mcp(v, "mcpServers", claude_mcp(EXE), true)).unwrap();
        let twice = json_edit(&once, |v| set_mcp(v, "mcpServers", claude_mcp(EXE), true)).unwrap();
        assert_eq!(once, twice);
        let v: Value = serde_json::from_str(&once).unwrap();
        assert_eq!(v["theme"], "dark");
        assert_eq!(v["mcpServers"]["other"]["command"], "x");
        assert_eq!(v["mcpServers"]["rag"]["args"][0], "serve");
    }

    #[test]
    fn mcp_uninstall_removes_only_rag() {
        let src = r#"{"mcpServers":{"rag":{"command":"r"},"other":{"command":"x"}}}"#;
        let out = json_edit(src, |v| set_mcp(v, "mcpServers", claude_mcp(EXE), false)).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["mcpServers"].get("rag").is_none());
        assert!(v["mcpServers"].get("other").is_some());
        let only = r#"{"mcpServers":{"rag":{}}}"#;
        let out = json_edit(only, |v| set_mcp(v, "mcpServers", claude_mcp(EXE), false)).unwrap();
        assert!(serde_json::from_str::<Value>(&out).unwrap().get("mcpServers").is_none());
    }

    #[test]
    fn hooks_install_replace_and_remove() {
        let src = r#"{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"echo hi"}]}]}}"#;
        let cmd = hook_command(EXE, true);
        let once = json_edit(src, |v| set_hooks(v, &cmd, true)).unwrap();
        let twice = json_edit(&once, |v| set_hooks(v, &cmd, true)).unwrap();
        assert_eq!(once, twice);
        let v: Value = serde_json::from_str(&once).unwrap();
        assert_eq!(v["hooks"]["SessionStart"].as_array().unwrap().len(), 2);
        assert_eq!(v["hooks"]["PreToolUse"][0]["matcher"], "Agent|Task");

        // A moved binary replaces the stale entry instead of duplicating it.
        let moved = hook_command(r"D:\bin\rag.exe", true);
        let out = json_edit(&once, |v| set_hooks(v, &moved, true)).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["hooks"]["SessionStart"].as_array().unwrap().len(), 2);

        let out = json_edit(&out, |v| set_hooks(v, &moved, false)).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["hooks"]["SessionStart"].as_array().unwrap().len(), 1);
        assert!(v["hooks"].get("PreToolUse").is_none());
    }

    #[test]
    fn invalid_json_is_rejected() {
        assert!(json_edit("{oops", |_| true).is_err());
        assert!(json_edit("[1]", |_| true).is_err());
    }

    #[test]
    fn codex_toml_roundtrip_keeps_comments() {
        let src = "# my config\nmodel = \"gpt\"\n\n[mcp_servers.other]\ncommand = \"x\"\n";
        let once = codex_edit(src, EXE, true).unwrap();
        assert!(once.contains("# my config"));
        assert!(once.contains("[mcp_servers.rag]"));
        let doc: DocumentMut = once.parse().unwrap();
        assert_eq!(doc["mcp_servers"]["rag"]["command"].as_str(), Some(EXE));
        assert_eq!(codex_edit(&once, EXE, true).unwrap(), once);
        let removed = codex_edit(&once, EXE, false).unwrap();
        assert!(!removed.contains("rag"));
        assert!(removed.contains("[mcp_servers.other]"));
    }

    #[test]
    fn instruction_block_lifecycle() {
        let added = set_block("# Notes\n", true);
        assert!(added.starts_with("# Notes\n\n<!-- rag:begin -->"));
        assert_eq!(set_block(&added, true), added);
        assert_eq!(set_block(&added, false), "# Notes\n");
        assert_eq!(set_block(&set_block("", true), false), "");
    }

    #[test]
    fn target_selection() {
        let home = std::env::temp_dir().join("rag-install-sel-none");
        assert!(select(&[], &home).unwrap().is_empty());
        assert_eq!(select(&["all".into()], &home).unwrap().len(), 4);
        assert_eq!(select(&["Claude".into(), "claude".into()], &home).unwrap(), vec![Agent::Claude]);
        assert!(select(&["vim".into()], &home).is_err());
    }

    #[test]
    fn end_to_end_in_temp_home() {
        let home = std::env::temp_dir().join(format!("rag-install-e2e-{}", std::process::id()));
        let _ = fs::remove_dir_all(&home);
        fs::create_dir_all(home.join(".claude")).unwrap();
        fs::write(home.join(".claude.json"), "{\"numStartups\":3}").unwrap();
        let mut opts = InstallOpts { targets: vec![], dry_run: false, uninstall: false, mcp_only: false, scope: Scope::Global };
        let lines = apply(Agent::Claude, &home, &home, EXE, &opts).unwrap();
        assert_eq!(lines.len(), 2);
        assert!(home.join(".claude.json.rag.bak").exists());
        let v: Value = serde_json::from_str(&fs::read_to_string(home.join(".claude.json")).unwrap()).unwrap();
        assert_eq!(v["numStartups"], 3);
        assert_eq!(v["mcpServers"]["rag"]["command"], EXE);
        opts.uninstall = true;
        apply(Agent::Claude, &home, &home, EXE, &opts).unwrap();
        let v: Value = serde_json::from_str(&fs::read_to_string(home.join(".claude.json")).unwrap()).unwrap();
        assert!(v.get("mcpServers").is_none());
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn local_scope_writes_project_files_only() {
        let root = std::env::temp_dir().join(format!("rag-install-local-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let (home, cwd) = (root.join("home"), root.join("proj"));
        fs::create_dir_all(&cwd).unwrap();
        let opts = InstallOpts { targets: vec![], dry_run: false, uninstall: false, mcp_only: false, scope: Scope::Local };
        for agent in Agent::ALL {
            apply(agent, &home, &cwd, EXE, &opts).unwrap();
        }
        assert!(!home.exists(), "local scope must not touch $HOME");
        assert!(cwd.join(".mcp.json").exists());
        assert!(cwd.join(".claude/settings.local.json").exists());
        assert!(cwd.join("opencode.json").exists());
        assert!(cwd.join(".codex/config.toml").exists());
        assert!(fs::read_to_string(cwd.join("AGENTS.md")).unwrap().contains(BLOCK_BEGIN));
        let _ = fs::remove_dir_all(&root);
    }
}
