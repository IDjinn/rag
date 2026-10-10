use crate::config;
use anyhow::{bail, Context, Result};
use futures::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub struct LmClient {
    http: reqwest::Client,
    base: String,
    api_key: Option<String>,
}

impl LmClient {
    pub fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("building HTTP client")?;
        Ok(Self {
            http,
            base: config::base_url(),
            api_key: config::api_key(),
        })
    }

    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base);
        let mut request = self.http.get(&url).timeout(Duration::from_secs(5));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let resp = request.send().await.map_err(|_| {
            anyhow::anyhow!(
                "{} not reachable at {}. Start the server and load an embedding model (set RAG_PROVIDER / RAG_BASE_URL / RAG_API_KEY to point elsewhere).",
                config::provider().name(),
                self.base
            )
        })?;
        if !resp.status().is_success() {
            bail!("{} /models returned HTTP {}", config::provider().name(), resp.status());
        }
        let json: Value = resp.json().await.context("parsing /models response")?;
        Ok(json["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default())
    }

    pub async fn resolve_embed_model(&self) -> Result<String> {
        if let Some(model) = config::embed_model_env() {
            return Ok(model);
        }
        let models = self.list_models().await?;
        if let Some(model) = models.iter().find(|m| m.to_lowercase().contains("embed")) {
            return Ok(model.clone());
        }
        if !models.is_empty() {
            bail!(
                "No embedding model available in LM Studio (models: {}). Load one (e.g. \"text-embedding-nomic-embed-text-v1.5\") or set RAG_EMBED_MODEL.",
                models.join(", ")
            );
        }
        bail!("LM Studio is reachable but no models are available.");
    }

    /// Embed a batch of texts; one retry on transient failures.
    pub async fn embed_batch(&self, model: &str, texts: &[String]) -> Result<(Vec<f32>, u64)> {
        let mut last_err = None;
        for attempt in 0..2 {
            match self.try_embed(model, texts).await {
                Ok(ok) => return Ok(ok),
                Err(err) => {
                    last_err = Some(err);
                    if attempt == 0 {
                        tokio::time::sleep(Duration::from_millis(400)).await;
                    }
                }
            }
        }
        Err(last_err.unwrap())
    }

    async fn try_embed(&self, model: &str, texts: &[String]) -> Result<(Vec<f32>, u64)> {
        let url = format!("{}/embeddings", self.base);
        let body = json!({ "model": model, "input": texts });
        let mut request = self.http.post(&url).json(&body).timeout(Duration::from_secs(300));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let resp = request
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("embeddings request failed: {e}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("{} /embeddings HTTP {status}: {}", config::provider().name(), snippet(&text, 300));
        }
        let json: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing embeddings response: {}", snippet(&text, 200)))?;
        let data = json
            .get("data")
            .and_then(Value::as_array)
            .context("embeddings response missing data")?;
        if data.len() != texts.len() {
            bail!("LM Studio returned {} embeddings for a batch of {}", data.len(), texts.len());
        }
        let mut sorted: Vec<&Value> = data.iter().collect();
        sorted.sort_by_key(|d| d.get("index").and_then(Value::as_u64).unwrap_or(0));

        let mut flat = Vec::with_capacity(sorted.len() * 768);
        let mut dim: Option<usize> = None;
        for item in &sorted {
            let emb = item
                .get("embedding")
                .and_then(Value::as_array)
                .context("embedding entry missing")?;
            match dim {
                None => dim = Some(emb.len()),
                Some(d) if d != emb.len() => bail!("inconsistent embedding dimensions across batch"),
                _ => {}
            }
            flat.extend(emb.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32));
        }
        // LM Studio reports usage as 0; treat 0 as "absent" and estimate.
        let tokens = json
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .filter(|t| *t > 0)
            .unwrap_or_else(|| (texts.iter().map(String::len).sum::<usize>() as u64).div_ceil(4));
        Ok((flat, tokens))
    }
}

fn snippet(text: &str, max: usize) -> String {
    let cut: String = text.chars().take(max).collect();
    if text.chars().count() > max {
        format!("{cut} …")
    } else {
        cut
    }
}

pub struct EmbedRun {
    pub vectors: Vec<f32>,
    pub dim: usize,
    pub tokens: u64,
    pub requests: u64,
    pub ms: u64,
}

/// Embed all texts with bounded concurrency. The first batch runs alone to
/// learn the dimension; the rest go through a `buffer_unordered` stream, each
/// result written back at its recorded offset. `on_batch` is called after every
/// completed batch with the cumulative chunk count and the batch's tokens.
pub async fn embed_all(
    client: &LmClient,
    model: &str,
    texts: &[String],
    batch: usize,
    concurrency: usize,
    on_batch: &mut (dyn FnMut(usize, u64) + Send),
) -> Result<EmbedRun> {
    if texts.is_empty() {
        return Ok(EmbedRun { vectors: Vec::new(), dim: 0, tokens: 0, requests: 0, ms: 0 });
    }
    let started = Instant::now();
    let batch = batch.max(1);
    let concurrency = concurrency.max(1);

    let first = texts.len().min(batch);
    let (first_vecs, first_tokens) = client.embed_batch(model, &texts[..first]).await?;
    let dim = first_vecs.len() / first.max(1);
    let mut vectors = vec![0.0f32; texts.len() * dim];
    vectors[..first_vecs.len()].copy_from_slice(&first_vecs);

    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut i = first;
    while i < texts.len() {
        let end = (i + batch).min(texts.len());
        ranges.push((i, end));
        i = end;
    }

    let total = texts.len();
    let done = AtomicUsize::new(first);
    on_batch(first.min(total), first_tokens);
    let mut tokens = first_tokens;
    let mut requests: u64 = 1;

    let mut stream = futures::stream::iter(ranges.into_iter().map(|(s, e)| async move {
        let (v, t) = client.embed_batch(model, &texts[s..e]).await?;
        Ok::<_, anyhow::Error>((s, e, v, t))
    }))
    .buffer_unordered(concurrency);

    while let Some(result) = stream.next().await {
        let (s, e, vecs, batch_tokens) = result?;
        vectors[s * dim..e * dim].copy_from_slice(&vecs);
        tokens += batch_tokens;
        requests += 1;
        let d = done.fetch_add(e - s, Ordering::Relaxed) + (e - s);
        on_batch(d, batch_tokens);
    }

    Ok(EmbedRun { vectors, dim, tokens, requests, ms: started.elapsed().as_millis() as u64 })
}
