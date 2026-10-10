use crate::config;
use crate::ops;
use anyhow::Result;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, ServiceExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

const NAME: &str = "zcode-rag";
const VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct RagServer {
    /// Roots indexed through this session — auto-refreshed in the background.
    watched: Arc<Mutex<Vec<PathBuf>>>,
}

/// One flat args type: every tool's parameters are optional.
#[derive(Deserialize, Default)]
struct ToolArgs {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    force: Option<bool>,
    #[serde(default)]
    dry_run: Option<bool>,
    #[serde(default)]
    batch_size: Option<usize>,
    #[serde(default)]
    concurrency: Option<usize>,
    #[serde(default)]
    no_ignore: Option<bool>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    k: Option<usize>,
    #[serde(default)]
    judge: Option<bool>,
    #[serde(default)]
    no_boost: Option<bool>,
    #[serde(default)]
    reset: Option<bool>,
    #[serde(default)]
    verdicts: Option<Vec<ops::FeedbackVerdict>>,
}

fn tool(
    name: &'static str,
    title: &str,
    description: &str,
    properties: Value,
    required: &[&str],
) -> Tool {
    let mut t = Tool::default();
    t.name = name.into();
    t.title = Some(title.to_string());
    t.description = Some(description.to_string().into());
    let schema = json!({ "type": "object", "properties": properties, "required": required });
    t.input_schema = Arc::new(schema.as_object().cloned().expect("object schema"));
    t
}

fn opt_path() -> Value {
    json!({ "path": { "type": "string", "description": "Root folder the index belongs to (default: server cwd)" } })
}

fn tools() -> Vec<Tool> {
    let refresh_secs = config::auto_refresh_interval_secs();
    vec![
        tool(
            "rag_status",
            "RAG status",
            "Show RAG index status for a root folder: whether an index exists, LM Studio reachability, embedding model, file/chunk counts, and auto-refresh state. Read-only and free.",
            opt_path(),
            &[],
        ),
        tool(
            "rag_index",
            "RAG index (initial / force rebuild)",
            &format!(
                "Index files under a folder into the local RAG index. Call once per root to build the initial index; afterwards the server re-embeds changed files automatically every {refresh_secs}s while a session is alive, so no need to call this for incremental updates. P0 (source code + markdown) is embedded first; unchanged files are skipped. force:true rebuilds from scratch."
            ),
            json!({
                "path": { "type": "string", "description": "Root folder to index (default: server cwd)" },
                "force": { "type": "boolean", "description": "Rebuild the whole index from scratch" },
                "dry_run": { "type": "boolean", "description": "Report files/chunks per tier without embedding" }
            }),
            &[],
        ),
        tool(
            "rag_search",
            "RAG search",
            "Semantic search over indexed files (LM Studio embeddings). Cheaper than reading many files: returns only top-k relevant chunks with file:line references. P0 results outrank P1 (damped 0.9) unless no_boost.",
            json!({
                "query": { "type": "string", "description": "Natural-language query (Portuguese or English)" },
                "k": { "type": "integer", "description": "Max results (default 5)", "minimum": 1, "maximum": 20 },
                "path": { "type": "string" },
                "judge": { "type": "boolean", "description": "Append a ready-to-run jev_triage payload to measure usefulness" },
                "no_boost": { "type": "boolean", "description": "Disable P1 score dampening" }
            }),
            &["query"],
        ),
        tool(
            "rag_feedback",
            "RAG feedback (jev-judged usefulness)",
            "Record usefulness verdicts (e.g. from a jev_triage relevance check) for the most recent rag_search, feeding the usefulness statistics.",
            json!({
                "verdicts": {
                    "type": "array",
                    "description": "One verdict per result id from the last rag_search (r1, r2, …)",
                    "items": { "type": "object", "properties": { "id": { "type": "string" }, "useful": { "type": "boolean" } }, "required": ["id", "useful"] }
                },
                "path": { "type": "string" }
            }),
            &["verdicts"],
        ),
        tool(
            "rag_stats",
            "RAG statistics",
            "Show cost vs usefulness statistics for the index: embedding tokens/time spent, searches served, context delivered vs full-file reads avoided, jev-judged usefulness rate, and a verdict line.",
            json!({
                "path": { "type": "string" },
                "reset": { "type": "boolean", "description": "Zero the counters (keeps the index)" }
            }),
            &[],
        ),
        tool(
            "rag_clear",
            "RAG clear",
            "Delete the index and its statistics for a root folder.",
            opt_path(),
            &[],
        ),
    ]
}

fn parse_args(params: &CallToolRequestParams) -> Result<ToolArgs, String> {
    let value = match &params.arguments {
        Some(map) => Value::Object(map.clone()),
        None => Value::Null,
    };
    serde_json::from_value(value).map_err(|e| format!("invalid arguments for {}: {e}", params.name))
}

impl ServerHandler for RagServer {
    fn get_info(&self) -> ServerInfo {
        let capabilities = ServerCapabilities::builder().enable_tools().build();
        let mut info = ServerInfo::new(capabilities);
        info.server_info = Implementation::new(NAME, VERSION);
        info.instructions = Some(format!(
            "Local RAG over files with LM Studio embeddings. Call rag_index once per root to \
             build the initial index; afterwards changed files are re-embedded automatically \
             every {}s while this session is alive. Prefer rag_search before reading files, and \
             rag_stats reports whether the index is paying off.",
            config::auto_refresh_interval_secs()
        ));
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let mut result = ListToolsResult::default();
        result.tools = tools();
        Ok(result)
    }

    async fn call_tool(
        &self,
        params: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let outcome: Result<String, String> = match parse_args(&params) {
            Err(msg) => Err(msg),
            Ok(args) => match params.name.as_ref() {
                "rag_status" => ops::status(args.path.as_deref())
                    .await
                    .map_err(|e| format!("{e:#}")),
                "rag_index" => {
                    let root = ops::resolve_root(args.path.as_deref());
                    let dry_run = args.dry_run.unwrap_or(false);
                    let outcome = ops::index(ops::IndexOpts {
                        path: args.path.clone(),
                        force: args.force.unwrap_or(false),
                        dry_run,
                        batch: args.batch_size,
                        concurrency: args.concurrency,
                        no_ignore: args.no_ignore.unwrap_or(false),
                        quiet: false,
                        no_progress: false, // auto-downgrades to plain over stdio pipes
                    })
                    .await
                    .map_err(|e| format!("{e:#}"));
                    // Roots indexed through this session join the auto-refresh watch list.
                    if outcome.is_ok() && !dry_run {
                        register_root(&self.watched, root).await;
                    }
                    outcome
                }
                "rag_search" => match args.query.as_deref() {
                    Some(query) => ops::search(
                        query,
                        args.k,
                        args.path.as_deref(),
                        args.judge.unwrap_or(false),
                        args.no_boost.unwrap_or(false),
                    )
                    .await
                    .map_err(|e| format!("{e:#}")),
                    None => Err("missing required argument: query".to_string()),
                },
                "rag_feedback" => match args.verdicts {
                    Some(verdicts) => ops::feedback_verdicts(verdicts, args.path.as_deref())
                        .map_err(|e| format!("{e:#}")),
                    None => Err("missing required argument: verdicts".to_string()),
                },
                "rag_stats" => ops::stats(args.path.as_deref(), args.reset.unwrap_or(false))
                    .map_err(|e| format!("{e:#}")),
                "rag_clear" => ops::clear(args.path.as_deref()).await.map_err(|e| format!("{e:#}")),
                other => Err(format!("unknown tool: {other}")),
            },
        };
        let result = match outcome {
            Ok(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
            Err(text) => CallToolResult::error(vec![ContentBlock::text(text)]),
        };
        Ok(result.into())
    }
}

pub async fn serve() -> i32 {
    let watched: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
    // Config is process-fixed (OnceLock), so the enabled check is final here.
    if config::auto_refresh() {
        tokio::spawn(auto_refresh_loop(watched.clone()));
    } else {
        eprintln!(
            "[rag] auto-refresh disabled (index.auto_refresh / RAG_AUTO_REFRESH); indexes only update on explicit rag_index"
        );
    }
    let service = RagServer { watched };
    match service.serve(rmcp::transport::stdio()).await {
        Ok(running) => {
            let _ = running.waiting().await;
            0
        }
        Err(err) => {
            eprintln!("rag mcp error: {err:#}");
            1
        }
    }
}

async fn register_root(watched: &Arc<Mutex<Vec<PathBuf>>>, root: PathBuf) {
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let mut roots = watched.lock().await;
    if !roots.contains(&root) {
        roots.push(root);
    }
}

/// Roots the current tick should consider: those indexed through this session
/// plus the server cwd (its initial index may come from an earlier session or
/// the CLI), canonicalized and deduplicated.
async fn candidate_roots(watched: &Arc<Mutex<Vec<PathBuf>>>) -> Vec<PathBuf> {
    let mut roots = watched.lock().await.clone();
    roots.push(ops::resolve_root(None));
    let mut seen = std::collections::HashSet::new();
    roots
        .into_iter()
        .map(|r| std::fs::canonicalize(&r).unwrap_or(r))
        .filter(|r| seen.insert(r.clone()))
        .collect()
}

/// Display a path without the `\\?\` verbatim prefix canonicalize adds on
/// Windows.
fn display_path(p: &std::path::Path) -> std::borrow::Cow<'_, str> {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{rest}").into();
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        return rest.to_string().into();
    }
    s
}

/// Background incremental refresh: every interval, diff each watched root
/// (stat-only) and re-embed only changed/new files. Idle ticks cost one
/// directory walk and no network I/O. On embedding errors (e.g. LM Studio
/// unloaded) it backs off exponentially up to 5 minutes and resets on the
/// first success.
async fn auto_refresh_loop(watched: Arc<Mutex<Vec<PathBuf>>>) {
    let interval = config::auto_refresh_interval_secs();
    let mut backoff = 0u64;
    loop {
        let wait = if backoff > 0 { backoff } else { interval };
        tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
        let mut failed = false;
        for root in candidate_roots(&watched).await {
            if !ops::index_exists(&root) {
                continue; // the initial index stays manual
            }
            let started = std::time::Instant::now();
            match ops::index_run(ops::IndexOpts {
                path: Some(root.to_string_lossy().into_owned()),
                quiet: true,
                ..Default::default()
            })
            .await
            {
                Ok(outcome) if outcome.changed => {
                    eprintln!(
                        "[rag] auto-refresh: {} embedded, {} reused (content unchanged), {} removed — {} ({:.1}s)",
                        outcome.embedded_files,
                        outcome.reused_files,
                        outcome.removed,
                        display_path(&root),
                        started.elapsed().as_secs_f32()
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "[rag] auto-refresh failed for {}: {e:#} — next attempt in {}s",
                        display_path(&root),
                        if backoff == 0 { interval * 2 } else { (backoff * 2).min(300) }
                    );
                    failed = true;
                    break;
                }
            }
        }
        backoff = if failed {
            if backoff == 0 { interval * 2 } else { (backoff * 2).min(300) }
        } else {
            0
        };
    }
}
