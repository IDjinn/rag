// Small sample used by the smoke test to validate indexing/search.

fn split_into_chunks(text: &str, target_chars: usize, overlap_chars: usize) -> Vec<String> {
    // Chunking keeps line alignment so results can cite file:line ranges.
    let lines: Vec<&str> = text.split('\n').collect();
    let mut chunks = Vec::new();
    let mut buffer: Vec<&str> = Vec::new();
    let mut size = 0usize;
    for line in lines {
        buffer.push(line);
        size += line.len() + 1;
        if size >= target_chars {
            chunks.push(buffer.join("\n"));
            let mut tail: Vec<&str> = Vec::new();
            let mut tail_size = 0usize;
            for kept in buffer.iter().rev() {
                if tail_size + kept.len() > overlap_chars {
                    break;
                }
                tail.insert(0, kept);
                tail_size += kept.len();
            }
            buffer = tail;
            size = tail_size;
        }
    }
    if !buffer.is_empty() {
        chunks.push(buffer.join("\n"));
    }
    chunks
}

fn main() {
    let demo = "the chunking function splits lines into overlapping chunks";
    for chunk in split_into_chunks(demo, 16, 4) {
        println!("{chunk}");
    }
}
