/// Chunk metadata with 1-based inclusive line numbers.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChunkMeta {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// Split text into overlapping, line-aligned chunks of roughly `target` chars.
/// Very long lines are hard-split at ~1200 chars on UTF-8 boundaries.
pub fn chunk_lines(text: &str, target: usize, overlap: usize) -> Vec<ChunkMeta> {
    let mut lines: Vec<&str> = Vec::with_capacity(text.len() / 40 + 1);
    for raw in text.split('\n') {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.len() <= 2000 {
            lines.push(line);
        } else {
            let bytes = line.as_bytes();
            let mut start = 0;
            while start < bytes.len() {
                let mut end = (start + 1200).min(bytes.len());
                while end > start && !line.is_char_boundary(end) {
                    end -= 1;
                }
                if end == start {
                    end = bytes.len();
                }
                lines.push(&line[start..end]);
                start = end;
            }
        }
    }

    let mut chunks: Vec<ChunkMeta> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut cur_start = 0usize;
    let mut cur_chars = 0usize;

    for (i, line) in lines.iter().enumerate() {
        cur.push(line);
        cur_chars += line.len() + 1;
        if cur_chars >= target {
            chunks.push(ChunkMeta {
                start: cur_start + 1,
                end: i + 1,
                text: cur.join("\n"),
            });
            let mut tail: Vec<&str> = Vec::new();
            let mut tail_chars = 0usize;
            let mut j = cur.len();
            while j > 0 && tail_chars + cur[j - 1].len() + 1 <= overlap {
                j -= 1;
                tail_chars += cur[j].len() + 1;
                tail.insert(0, cur[j]);
            }
            cur_start = i + 1 - tail.len();
            cur = tail;
            cur_chars = tail_chars;
        }
    }
    // Blank-only tails (e.g. the trailing empty line of "") produce no chunk.
    if !cur.is_empty() && cur.iter().any(|l| !l.is_empty()) {
        chunks.push(ChunkMeta {
            start: cur_start + 1,
            end: lines.len(),
            text: cur.join("\n"),
        });
    }
    chunks
}

pub fn chunk_default(text: &str) -> Vec<ChunkMeta> {
    let settings = &crate::config::settings().index;
    chunk_lines(text, settings.chunk_target_chars, settings.chunk_overlap_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_has_no_chunks() {
        assert!(chunk_default("").is_empty());
    }

    #[test]
    fn small_text_is_one_chunk() {
        let chunks = chunk_default("hello\nworld\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start, 1);
        assert_eq!(chunks[0].end, 3);
        assert_eq!(chunks[0].text, "hello\nworld\n");
    }

    #[test]
    fn long_text_overlaps() {
        let line = "x".repeat(60);
        let text: String = (0..100).map(|_| format!("{line}\n")).collect();
        let chunks = chunk_default(&text);
        assert!(chunks.len() >= 5, "expected several chunks, got {}", chunks.len());
        // Overlap: the next chunk starts before the previous one ended.
        for pair in chunks.windows(2) {
            assert!(pair[1].start < pair[0].end, "chunks should overlap");
        }
        // Line coverage stays inside the file (split('\n') yields one extra
        // trailing empty line for text ending in '\n').
        assert!(chunks.last().unwrap().end <= 101);
    }

    #[test]
    fn crlf_is_normalized() {
        let chunks = chunk_default("a\r\nb\r\n");
        assert_eq!(chunks[0].text, "a\nb\n");
    }

    #[test]
    fn long_line_is_hard_split() {
        let long = "z".repeat(5000);
        let chunks = chunk_default(&long);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.text.len() <= 1300);
        }
    }
}
