mod chunk;
mod config;
mod files;
mod lmstudio;
mod mcp;
mod ops;
mod progress;
mod simd;
mod stats;
mod store;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "rag",
    version,
    about = "Local RAG for ZCode: manual indexing with LM Studio embeddings, SIMD search, cost/utility stats."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Manual (incremental) indexing; unchanged files are skipped
    Index {
        /// Root folder to index (default: current directory)
        path: Option<String>,
        /// Rebuild from scratch, re-embedding everything
        #[arg(long)]
        force: bool,
        /// Report what would be indexed per tier without embedding
        #[arg(long)]
        dry_run: bool,
        /// Chunks per embedding request (default: config.yaml)
        #[arg(long)]
        batch_size: Option<usize>,
        /// Parallel embedding requests against LM Studio (default: config.yaml)
        #[arg(long)]
        concurrency: Option<usize>,
        /// Ignore .gitignore/.ignore rules (hidden files stay hidden)
        #[arg(long)]
        no_ignore: bool,
        /// Plain, non-interactive progress output (default when stderr is not a terminal)
        #[arg(long)]
        no_progress: bool,
    },
    /// Semantic search over an indexed root
    Search {
        /// Natural-language query (Portuguese or English)
        query: String,
        /// Max results (default: config.yaml)
        #[arg(long)]
        k: Option<usize>,
        /// Root folder the index belongs to (default: current directory)
        path: Option<String>,
        /// Append a ready-to-run jev_triage payload to measure usefulness
        #[arg(long)]
        judge: bool,
        /// Disable the P1 (low-priority) score dampening
        #[arg(long)]
        no_boost: bool,
    },
    /// Cost vs usefulness statistics
    Stats {
        /// Root folder the index belongs to (default: current directory)
        path: Option<String>,
        /// Zero the counters (keeps the index)
        #[arg(long)]
        reset: bool,
    },
    /// Index status and LM Studio reachability
    Status {
        /// Root folder the index belongs to (default: current directory)
        path: Option<String>,
    },
    /// Delete the index and its statistics for a root
    Clear {
        /// Root folder the index belongs to (default: current directory)
        path: Option<String>,
    },
    /// Record jev usefulness verdicts for the last search
    Feedback {
        /// JSON: {"verdicts":[{"id":"r1","useful":true}]}
        #[arg(long = "json")]
        json: String,
        /// Root folder the index belongs to (default: current directory)
        path: Option<String>,
    },
    /// Run as an MCP stdio server (default when no subcommand is given)
    Serve,
    /// Show the effective YAML config, or write a default one with --init
    Config {
        /// Write a commented default config if none exists
        #[arg(long)]
        init: bool,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    std::process::exit(run(cli.command).await);
}

async fn run(command: Option<Command>) -> i32 {
    match command.unwrap_or(Command::Serve) {
        Command::Serve => return mcp::serve().await,
        Command::Index { path, force, dry_run, batch_size, concurrency, no_ignore, no_progress } => {
            report(ops::index(path.as_deref(), force, dry_run, batch_size, concurrency, no_ignore, no_progress).await)
        }
        Command::Search { query, k, path, judge, no_boost } => {
            report(ops::search(&query, k, path.as_deref(), judge, no_boost).await)
        }
        Command::Config { init } => report(ops::config_cmd(init)),
        Command::Stats { path, reset } => report(ops::stats(path.as_deref(), reset)),
        Command::Status { path } => report(ops::status(path.as_deref()).await),
        Command::Clear { path } => report(ops::clear(path.as_deref())),
        Command::Feedback { json, path } => report(ops::feedback(&json, path.as_deref())),
    }
}

fn report(result: anyhow::Result<String>) -> i32 {
    match result {
        Ok(text) => {
            println!("{text}");
            0
        }
        Err(err) => {
            eprintln!("rag error: {err:#}");
            1
        }
    }
}
