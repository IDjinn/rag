use crate::config::{self, Tier};
use anyhow::{Context, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChunkRef {
    pub start: usize,
    pub end: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FileEntry {
    pub hash: String,
    pub size: u64,
    pub mtime_ms: i64,
    pub chunk_count: usize,
    /// Row of the file's first chunk vector in vectors.bin.
    pub vec_offset: usize,
    pub chunks: Vec<ChunkRef>,
    pub tier: Tier,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IndexData {
    pub version: u32,
    pub root: String,
    pub model: String,
    #[serde(default = "default_provider")]
    pub provider: String,
    pub dim: usize,
    pub files: BTreeMap<String, FileEntry>,
    pub vector_count: usize,
    pub updated_at: String,
}

fn default_provider() -> String {
    "lmstudio".to_string()
}

impl IndexData {
    pub fn empty(root: &str, model: &str) -> Self {
        Self {
            version: 1,
            root: root.to_string(),
            model: model.to_string(),
            provider: crate::config::provider_name().to_string(),
            dim: 0,
            files: BTreeMap::new(),
            vector_count: 0,
            updated_at: now_iso(),
        }
    }
}

pub fn sha1_hex(text: &str) -> String {
    let digest = Sha1::digest(text.as_bytes());
    hex::encode(digest)
}

pub fn now_iso() -> String {
    // ISO-8601 UTC without pulling chrono: second precision is enough for stats.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86_400;
    let (y, m, d) = civil_from_days(days as i64);
    let rem = secs % 86_400;
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

/// Howard Hinnant's civil-from-days algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub struct Store {
    pub dir: PathBuf,
}

impl Store {
    pub fn for_root(root: &Path) -> Self {
        Self { dir: config::index_dir_for(root) }
    }

    pub fn load(&self) -> Option<IndexData> {
        let text = fs::read_to_string(self.dir.join("index.json")).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Memory-map vectors.bin for zero-copy search. The file holds raw
    /// native-endian f32 rows written by [`save`]; x86/ARM64 are little-endian,
    /// which the compile-time assertion below pins down.
    pub fn mmap_vectors(&self) -> Option<Mmap> {
        let file = fs::File::open(self.dir.join("vectors.bin")).ok()?;
        let len = file.metadata().ok()?.len();
        if len == 0 || len % 4 != 0 {
            return None;
        }
        #[cfg(target_endian = "big")]
        {
            return None; // writer stores little-endian native f32; unsupported here
        }
        #[allow(unreachable_code)]
        unsafe { Mmap::map(&file).ok() }
    }

    pub fn save(&self, data: &IndexData, vectors: &[f32]) -> Result<()> {
        fs::create_dir_all(&self.dir).context("creating index dir")?;
        let meta_tmp = self.dir.join("index.json.tmp");
        let vec_tmp = self.dir.join("vectors.bin.tmp");
        fs::write(&meta_tmp, serde_json::to_vec(data)?)
            .with_context(|| format!("writing {}", meta_tmp.display()))?;
        // Native-endian f32 raw dump (asserted little-endian at mmap time).
        let bytes = unsafe {
            std::slice::from_raw_parts(vectors.as_ptr().cast::<u8>(), vectors.len() * 4)
        };
        fs::write(&vec_tmp, bytes).with_context(|| format!("writing {}", vec_tmp.display()))?;
        fs::rename(&meta_tmp, self.dir.join("index.json"))?;
        fs::rename(&vec_tmp, self.dir.join("vectors.bin"))?;
        Ok(())
    }

    pub fn disk_bytes(&self) -> u64 {
        let mut total = 0;
        if let Ok(entries) = fs::read_dir(&self.dir) {
            for e in entries.flatten() {
                if let Ok(md) = e.metadata() {
                    total += md.len();
                }
            }
        }
        total
    }

    pub fn remove(&self) -> bool {
        fs::remove_dir_all(&self.dir).is_ok()
    }
}

#[inline]
pub fn vectors_as_slice(m: &Mmap) -> &[f32] {
    debug_assert_eq!(m.len() % 4, 0);
    let ptr = m.as_ptr();
    assert_eq!(ptr.align_offset(4), 0, "mmap not 4-byte aligned");
    unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), m.len() / 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("rag-store-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store { dir: dir.clone() };

        let mut data = IndexData::empty("C:/tmp/root", "test-model");
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
        data.vector_count = 2;
        let vectors = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        store.save(&data, &vectors).unwrap();

        let loaded = store.load().unwrap();
        assert_eq!(loaded.model, "test-model");
        assert_eq!(loaded.files["a.rs"].chunk_count, 2);

        let mmap = store.mmap_vectors().unwrap();
        let rows = vectors_as_slice(&mmap);
        assert_eq!(rows.len(), 8);
        assert_eq!(crate::simd::dot(&rows[0..4], &vectors[0..4]), 1.0);
        assert!((crate::simd::dot(&rows[0..4], &vectors[4..8]) - 0.0).abs() < 1e-6);

        assert!(store.remove());
        assert!(store.load().is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn iso_dates_look_right() {
        let ts = now_iso();
        assert_eq!(ts.len(), 20);
        assert!(ts.ends_with('Z'));
        assert_eq!(&ts[4..5], "-");
    }
}
