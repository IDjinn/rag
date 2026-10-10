//! Terminal rendering of the ops reports.
//!
//! Every report from `ops` is authored as markdown because the MCP tools hand
//! it to a model verbatim. When the same strings go to a terminal (the CLI
//! subcommands with stdout as a TTY), this module lays them out for humans:
//! headings become underlined titles, pipe tables become aligned columns with
//! the separator row dropped, `---` rules become horizontal lines, and inline
//! `**bold**`/backtick markers are stripped. Fenced code blocks — search
//! snippets, the judge payload — pass through verbatim.

/// Width of the horizontal rules that replace `---` lines.
const RULE_WIDTH: usize = 56;

pub fn render(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len());
    let mut lines = markdown.lines().peekable();
    let mut in_code = false;

    while let Some(line) = lines.next() {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if in_code {
            out.push_str(line);
            out.push('\n');
            continue;
        }

        let trimmed = line.trim();

        // Table block: a `|` row followed by the `|---|` separator row, then
        // consecutive `|` rows.
        if trimmed.starts_with('|') && lines.peek().is_some_and(|l| is_separator_row(l)) {
            lines.next(); // consume the separator row: layout-only in markdown
            let mut rows = vec![trimmed.to_string()];
            while let Some(next) = lines.peek() {
                if next.trim().starts_with('|') {
                    rows.push(lines.next().unwrap().trim().to_string());
                } else {
                    break;
                }
            }
            render_table(&rows, &mut out);
            continue;
        }

        let level = if trimmed.starts_with("### ") {
            Some(3)
        } else if trimmed.starts_with("## ") {
            Some(2)
        } else {
            None
        };
        if let Some(level) = level {
            let title = strip_inline(trimmed.trim_start_matches('#').trim_start());
            out.push('\n');
            out.push_str(&title);
            out.push('\n');
            let bar = if level == 2 { '━' } else { '─' };
            for _ in 0..title.chars().count() {
                out.push(bar);
            }
            out.push('\n');
            continue;
        }

        if trimmed == "---" {
            out.push_str(&"─".repeat(RULE_WIDTH));
            out.push('\n');
            continue;
        }

        out.push_str(&strip_inline(line));
        out.push('\n');
    }

    // The leading blank line before the first heading is separation from the
    // prompt, not layout — drop it.
    out.trim_start_matches('\n').to_string()
}

fn strip_inline(line: &str) -> String {
    line.replace("**", "").replace('`', "")
}

/// True when every cell of a `|` row is a markdown alignment dash (`---`,
/// `:--:`). Decides whether the row is a table's separator row.
fn is_separator_row(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') {
        return false;
    }
    let cells = split_row(trimmed);
    !cells.is_empty() && cells.iter().all(|c| {
        let c = c.trim().trim_matches(':');
        !c.is_empty() && c.chars().all(|ch| ch == '-')
    })
}

/// Splits one `| a | b |` row into cells, unescaping `\|` (which the reports
/// emit inside table cells to survive markdown) back to a literal pipe.
fn split_row(row: &str) -> Vec<String> {
    let guarded = row.replace("\\|", "\u{1}");
    let inner = guarded.trim().trim_start_matches('|').trim_end_matches('|');
    inner
        .split('|')
        .map(|c| c.replace('\u{1}', "|").trim().to_string())
        .collect()
}

/// Fixed-width column layout: header row, a `─┼─` rule, then data rows.
fn render_table(rows: &[String], out: &mut String) {
    let parsed: Vec<Vec<String>> = rows.iter().map(|r| split_row(r)).collect();
    let cols = parsed.iter().map(Vec::len).max().unwrap_or(0);
    if cols == 0 {
        return;
    }
    let mut widths = vec![0usize; cols];
    for row in &parsed {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }
    for (ri, row) in parsed.iter().enumerate() {
        let mut line = String::new();
        for i in 0..cols {
            let cell = row.get(i).map(String::as_str).unwrap_or("");
            line.push_str(cell);
            line.push_str(&" ".repeat(widths[i] - cell.chars().count()));
            if i + 1 < cols {
                line.push_str(" | ");
            }
        }
        out.push_str(line.trim_end());
        out.push('\n');
        if ri == 0 {
            let mut rule = String::new();
            for i in 0..cols {
                rule.push_str(&"─".repeat(widths[i]));
                if i + 1 < cols {
                    rule.push_str("─┼─");
                }
            }
            out.push_str(&rule);
            out.push('\n');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headings_underlined_and_inline_markers_stripped() {
        let out = render("## rag index complete\n\n- **Root:** `src/`\n");
        let mut lines = out.lines();
        assert_eq!(lines.next(), Some("rag index complete"));
        let bar = lines.next().unwrap();
        assert!(!bar.is_empty() && bar.chars().all(|c| c == '━'));
        assert!(out.contains("- Root: src/"));
        assert!(!out.contains("**"));
        assert!(!out.contains('`'));
        assert!(!out.starts_with('\n'));
    }

    #[test]
    fn section_headings_use_a_lighter_rule() {
        let out = render("### Index\n\ntext\n");
        let mut lines = out.lines();
        assert_eq!(lines.next(), Some("Index"));
        let bar = lines.next().unwrap();
        assert!(!bar.is_empty() && bar.chars().all(|c| c == '─'));
    }

    #[test]
    fn tables_render_aligned_without_markdown_pipes() {
        let md = "### Recent queries\n\n\
                  | when | query | results |\n\
                  |---|---|---|\n\
                  | 10-10 18:23 | AI container faces detect | 8 |\n\
                  | 10-10 17:05 | embedding dim | 5 |\n\
                  \nUse `rag stats --reset`.\n";
        let out = render(md);
        // The markdown separator row is gone; the aligned layout stays.
        assert!(!out.contains("|---|"));
        assert!(!out.contains("| when |"));
        assert!(out.contains("─┼─"));
        // Header cell "when" is padded to the widest cell of its column
        // ("10-10 18:23", 11 chars), then columns join with " | ".
        let expected = format!("when{}| query", " ".repeat("10-10 18:23".len() - 4 + 1));
        let header_line = out.lines().find(|l| l.starts_with("when")).unwrap();
        assert!(header_line.starts_with(&expected));
        // Rows keep every column, padded to the widest cell.
        assert!(out.lines().any(|l| l.starts_with("10-10 18:23 ")));
        assert!(out.contains("Use rag stats --reset."));
    }

    #[test]
    fn escaped_pipes_in_cells_unescape() {
        let md = "| query | results |\n|---|---|\n| a\\|b | 8 |\n";
        let out = render(md);
        assert!(out.contains("a|b"));
        // Still two columns: the escaped pipe did not create a third one.
        assert!(out.lines().any(|l| l.starts_with("a|b ")));
    }

    #[test]
    fn code_fences_pass_through_verbatim() {
        let md = "```rust\nlet s = \"a|b\"; // **not stripped** `raw`\n```\n";
        let out = render(md);
        assert!(out.contains("let s = \"a|b\"; // **not stripped** `raw`"));
        assert!(out.contains("```rust"));
    }

    #[test]
    fn hr_rules_become_terminal_rules() {
        let out = render("---\n\ntext\n");
        assert!(out.starts_with(&"─".repeat(RULE_WIDTH)));
    }
}
