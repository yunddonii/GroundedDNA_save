"""Reorder docs/PROJECT_LOG.md into a consistent layout.

Target layout (policy A, agreed 2026-05-15):

    # GroundedDNA — Project Log
    <preamble>
    ---
    ## Current state (...)
    <snapshot>
    ---
    <all dated sections, newest first by (date, version_index)>
    ---
    ## Infrastructure & repo hygiene
    <static section, always at bottom>
    ---
    <HTML "HOW TO UPDATE" comment>

Same-date tie-breaker:
    Within one date, sort by the largest version number that appears in
    the section header (e.g. "v27b" > "v27a" > "v26b" > ...). Sections
    with no version token fall back to their original file position
    (stable sort), which preserves the human-curated ordering of
    infrastructure / cache / cleanup notes.

This script is idempotent: run it again and the file is unchanged.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

LOG_PATH = Path("/home/yschoi/GroundedDNA/docs/PROJECT_LOG.md")


HEADER_END_RE       = re.compile(r"^---\s*$")
H2_RE               = re.compile(r"^##\s+(.*)$")
DATE_IN_H2_RE       = re.compile(r"(\d{4}-\d{2}-\d{2})")
VERSION_IN_H2_RE    = re.compile(r"v(\d+)([A-Za-z\-]?)", re.IGNORECASE)
CURRENT_STATE_RE    = re.compile(r"^##\s+Current state", re.IGNORECASE)
INFRA_RE            = re.compile(r"^##\s+Infrastructure", re.IGNORECASE)


def parse(text: str):
    """Split into (preamble, sections, footer_html_comment).

    A "section" is everything from a `## ` line up to (but not including)
    the next `## ` line OR the trailing HTML comment.
    """
    lines = text.splitlines(keepends=True)

    # Find the FIRST `## ` line. Everything before it (incl. its preceding
    # `---`) is the preamble.
    first_h2 = None
    for i, ln in enumerate(lines):
        if H2_RE.match(ln):
            first_h2 = i
            break
    if first_h2 is None:
        raise SystemExit("PROJECT_LOG.md has no ## section?")
    preamble = "".join(lines[:first_h2])

    # Find the trailing HTML comment block.  Treating a mid-document comment
    # as the footer would silently hide every later H2 from the sorter, so
    # fail closed if any section follows the comment.
    html_start = None
    for i, ln in enumerate(lines):
        if ln.startswith("<!--"):
            html_start = i
            break
    if html_start is not None:
        html_end = next(
            (i for i in range(html_start, len(lines)) if "-->" in lines[i]),
            None,
        )
        if html_end is None:
            raise SystemExit("PROJECT_LOG.md has an unterminated HTML footer comment")
        if any(H2_RE.match(ln) for ln in lines[html_end + 1:]):
            raise SystemExit(
                "PROJECT_LOG.md HTML footer comment is not trailing; "
                "move it after the final ## section before reordering"
            )
    footer = "".join(lines[html_start:]) if html_start is not None else ""
    body_end = html_start if html_start is not None else len(lines)

    # Split sections by ## lines within [first_h2, body_end).
    sections: list[tuple[int, str]] = []
    start = first_h2
    for i in range(first_h2 + 1, body_end):
        if H2_RE.match(lines[i]):
            sections.append((start, "".join(lines[start:i])))
            start = i
    sections.append((start, "".join(lines[start:body_end])))

    return preamble, sections, footer


def classify_and_sort(sections: list[tuple[int, str]]):
    """Return (current_state, dated_sections_sorted_desc, infra, others_pos).

    Each "dated" section gets a sort key (date_str_desc, version_num_desc,
    suffix_desc, original_index_asc).
    """
    current_state = None
    infra = None
    dated: list[dict] = []
    misc: list[tuple[int, str]] = []   # other untyped sections

    for idx, (orig_pos, sec) in enumerate(sections):
        head = sec.splitlines()[0]
        if CURRENT_STATE_RE.match(head):
            current_state = sec
            continue
        if INFRA_RE.match(head):
            infra = sec
            continue
        m_date = DATE_IN_H2_RE.search(head)
        if not m_date:
            misc.append((orig_pos, sec))
            continue
        date = m_date.group(1)
        # collect every "v<digits><letter?>" in the header; pick the LARGEST
        # version-int (then letter) so headers like "v23a / v23b" sort
        # together with the highest variant determining order.
        vers = VERSION_IN_H2_RE.findall(head)
        if vers:
            major_int = max(int(v[0]) for v in vers)
            # pick the suffix attached to the largest version
            suffix = ""
            for v in vers:
                if int(v[0]) == major_int and v[1]:
                    suffix = max(suffix, v[1].lower())
        else:
            major_int = -1   # no version: sort below versioned same-date sections
            suffix = ""
        dated.append({
            "date":     date,
            "major":    major_int,
            "suffix":   suffix,
            "orig_idx": idx,
            "section":  sec,
        })

    # Sort: date desc, major desc, suffix desc, orig_idx asc
    def key(s):
        return (
            tuple(-int(p) for p in s["date"].split("-")),  # date desc
            -s["major"],                                    # version desc
            tuple(-ord(c) for c in s["suffix"]),           # suffix desc
            s["orig_idx"],                                  # stable
        )
    dated_sorted = sorted(dated, key=key)
    return current_state, dated_sorted, infra, misc


def _strip_trailing_separator(s: str) -> str:
    """Strip a trailing `---` separator (and surrounding blank lines) so we
    can re-emit it uniformly. Keeps the body content intact."""
    # remove trailing whitespace lines
    s = s.rstrip()
    # if it now ends with `---`, strip that too plus any blank lines before
    if s.endswith("---"):
        s = s[: -3].rstrip()
    return s


def render(preamble, current_state, dated_sorted, infra, misc, footer):
    out = [preamble.rstrip("\n") + "\n\n"]
    if current_state is not None:
        out.append(_strip_trailing_separator(current_state) + "\n\n---\n\n")
    for entry in dated_sorted:
        out.append(_strip_trailing_separator(entry["section"]) + "\n\n---\n\n")
    for _, sec in misc:
        out.append(_strip_trailing_separator(sec) + "\n\n---\n\n")
    if infra is not None:
        out.append(_strip_trailing_separator(infra) + "\n\n---\n\n")
    if footer:
        out.append(footer if footer.endswith("\n") else footer + "\n")
    return "".join(out)


def main() -> int:
    text = LOG_PATH.read_text()
    preamble, sections, footer = parse(text)
    current_state, dated_sorted, infra, misc = classify_and_sort(sections)
    new_text = render(preamble, current_state, dated_sorted, infra, misc, footer)

    # idempotency guard: only write if changed
    if new_text == text:
        print(f"[reorder] {LOG_PATH} already in target order (no change).")
        return 0
    LOG_PATH.write_text(new_text)
    print(f"[reorder] rewrote {LOG_PATH} "
          f"(sections={len(sections)}, dated={len(dated_sorted)}).")
    # Sanity check
    if current_state is not None:
        print(f"[reorder] Current state pinned at top.")
    if infra is not None:
        print(f"[reorder] Infrastructure section moved to bottom.")
    # Newest 5 dated sections
    print("[reorder] top 5 newest sections:")
    for e in dated_sorted[:5]:
        head = e["section"].splitlines()[0]
        print(f"  {e['date']}  v{e['major']}{e['suffix']:<2s}  {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
