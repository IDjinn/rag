use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct QueryRecord {
    pub ts: String,
    pub query: String,
    pub results: usize,
    pub top_score: f32,
    pub ms: u64,
    #[serde(default)]
    pub judged_count: Option<u64>,
    #[serde(default)]
    pub useful_count: Option<u64>,
    /// Context tokens the results added, estimated (chars/4).
    #[serde(default)]
    pub delivered_tokens: u64,
    /// Net saving: full-file tokens avoided minus tokens delivered.
    #[serde(default)]
    pub saved_tokens: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct RagStats {
    pub index_runs: u64,
    pub last_indexed_at: Option<String>,
    pub last_index_ms: Option<u64>,
    pub files_indexed: u64,
    pub files_skipped: u64,
    pub files_removed: u64,
    pub chunks_embedded: u64,
    pub embed_requests: u64,
    pub embed_tokens: u64,
    pub embed_ms: u64,
    pub embed_errors: u64,
    pub searches: u64,
    pub search_results: u64,
    pub search_hits: u64,
    pub search_ms: u64,
    pub search_errors: u64,
    pub context_tokens_delivered: u64,
    pub full_file_tokens_avoided: u64,
    pub judge_sessions: u64,
    pub judged_chunks: u64,
    pub useful_chunks: u64,
    pub p0_files: u64,
    pub p1_files: u64,
    pub history: Vec<QueryRecord>,
}

pub fn load(dir: &std::path::Path) -> RagStats {
    std::fs::read_to_string(dir.join("stats.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

pub fn save(dir: &std::path::Path, stats: &RagStats) {
    let _ = std::fs::create_dir_all(dir);
    let tmp = dir.join("stats.json.tmp");
    if Ok(())
        .and_then(|_| std::fs::write(&tmp, serde_json::to_vec(stats).unwrap_or_default()))
        .is_ok()
    {
        let _ = std::fs::rename(&tmp, dir.join("stats.json"));
    }
}

pub fn push_history(stats: &mut RagStats, record: QueryRecord) {
    stats.history.insert(0, record);
    stats.history.truncate(crate::config::HISTORY_LIMIT);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_record_defaults_missing_token_fields() {
        // stats.json written before per-search token tracking existed.
        let old = r#"{"ts":"2026-10-01T10:00:00Z","query":"q","results":3,"top_score":0.5,"ms":12}"#;
        let rec: QueryRecord = serde_json::from_str(old).unwrap();
        assert_eq!(rec.delivered_tokens, 0);
        assert_eq!(rec.saved_tokens, 0);
    }
}
