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
use std::sync::Arc;

const NAME: &str = "zcode-rag";
const VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct RagServer;

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
    description: &'static str,
    properties: Value,
    required: &[&str],
) -> Tool {
    let mut t = Tool::default();
    t.name = name.into();
    t.title = Some(title.to_string());
    t.description = Some(description.into());
    let schema = json!({ "type": "object", "properties": properties, "required": required });
    t.input_schema = Arc::new(schema.as_object().cloned().expect("object schema"));
    t
}

fn opt_path() -> Value {
    json!({ "path": { "type": "string", "description": "Root folder the index belongs to (default: server cwd)" } })
}

fn tools() -> Vec<Tool> {
    vec![
        tool(
            "rag_status",
            "RAG status",
            "Show RAG index status for a root folder: whether an index exists, LM Studio reachability, embedding model, file/chunk counts. Read-only and free.",
            opt_path(),
            &[],
        ),
        tool(
            "rag_index",
            "RAG index (manual)",
            "Manually index (or incrementally refresh) files under a folder into the local RAG index using LM Studio embeddings. Manual by design: call this explicitly after meaningful changes. Unchanged files are skipped (cheap); P0 (source code + markdown) is embedded first, P1 follows. force:true rebuilds from scratch.",
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
            "Semantic search over manually indexed files (LM Studio embeddings). Cheaper than reading many files: returns only top-k relevant chunks with file:line references. P0 results outrank P1 (damped 0.9) unless no_boost.",
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
        info.instructions = Some(
            "Manual RAG over local files with LM Studio embeddings. Call rag_index once per root, \
             prefer rag_search before reading files, and rag_stats reports whether the index is \
             paying off."
                .into(),
        );
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
                "rag_index" => ops::index(
                    args.path.as_deref(),
                    args.force.unwrap_or(false),
                    args.dry_run.unwrap_or(false),
                    args.batch_size,
                    args.concurrency,
                    args.no_ignore.unwrap_or(false),
                    false, // progress auto-downgrades to plain over stdio pipes
                )
                .await
                .map_err(|e| format!("{e:#}")),
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
                "rag_clear" => ops::clear(args.path.as_deref()).map_err(|e| format!("{e:#}")),
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
    let service = RagServer;
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
