"""Turn the Markdown reports into PDFs, with no dependency beyond a browser.

A report that lives only as Markdown is awkward to attach to a ticket, and every converter worth having
(pandoc, weasyprint) drags in a toolchain. Chrome is already on every machine that can run this, and
headless Chrome renders tables — which is most of what these reports are — properly.

The Markdown here is the subset the reports actually use: headings, tables, fenced code, lists,
blockquotes, rules, and inline bold/italic/code/links. It is deliberately not a general Markdown
implementation; anything it does not recognise passes through as a paragraph rather than being dropped.
"""

import html
import re
import shutil
import subprocess
from pathlib import Path

CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

CSS = """
@page { size: A4; margin: 18mm 14mm; }
body { font: 10.5pt/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1a1a1a; }
h1 { font-size: 19pt; margin: 0 0 4mm; border-bottom: 2px solid #333; padding-bottom: 2mm; }
h2 { font-size: 14pt; margin: 7mm 0 2mm; }
h3 { font-size: 11.5pt; margin: 5mm 0 2mm; }
p, li { margin: 0 0 2.5mm; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0 5mm; font-size: 8.5pt; }
th, td { border: 1px solid #c8c8c8; padding: 1.6mm 2.2mm; text-align: left; vertical-align: top; }
th { background: #f0f0f0; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
code { font: 9pt "SF Mono", Menlo, monospace; background: #f2f2f2; padding: 0.4mm 1mm; border-radius: 2px; }
pre { background: #f6f6f6; border-left: 3px solid #bbb; padding: 2.5mm 3mm; overflow-x: auto; }
pre code { background: none; padding: 0; font-size: 8.5pt; }
blockquote { margin: 3mm 0; padding: 0 0 0 4mm; border-left: 3px solid #999; color: #444; }
hr { border: none; border-top: 1px solid #ccc; margin: 5mm 0; }
h1, h2, h3 { break-after: avoid; }
table, pre, blockquote { break-inside: avoid; }
"""


def _inline(text: str) -> str:
    """Inline markup, applied to already-escaped text."""
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\w)", r"<em>\1</em>", text)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def to_html(md: str, title: str) -> str:
    out: list[str] = []
    lines = md.splitlines()
    i, in_list = 0, False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            close_list()
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(html.escape(lines[i]))
                i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            i += 1
            continue

        # A table is a header row, a separator of dashes, then body rows.
        if "|" in stripped and i + 1 < len(lines) and re.fullmatch(r"\|?[\s:|-]+\|[\s:|-]*", lines[i + 1].strip()):
            close_list()
            head = _cells(stripped)
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_cells(lines[i]))
                i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(html.escape(c))}</th>" for c in head))
            out.append("</tr></thead><tbody>")
            for r in rows:
                r += [""] * (len(head) - len(r))
                out.append("<tr>" + "".join(f"<td>{_inline(html.escape(c))}</td>" for c in r[: len(head)]) + "</tr>")
            out.append("</tbody></table>")
            continue

        if not stripped:
            close_list()
        elif m := re.match(r"^(#{1,4})\s+(.*)", stripped):
            close_list()
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(html.escape(m.group(2)))}</h{level}>")
        elif re.fullmatch(r"[-*_]{3,}", stripped):
            close_list()
            out.append("<hr>")
        elif stripped.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(html.escape(stripped[2:]))}</li>")
        elif stripped.startswith("> "):
            close_list()
            out.append(f"<blockquote>{_inline(html.escape(stripped[2:]))}</blockquote>")
        else:
            # Wrapped prose: join with the lines that follow until a blank line.
            close_list()
            para = [stripped]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^[#>|`-]|^\*\s", lines[i + 1].strip()):
                i += 1
                para.append(lines[i].strip())
            out.append(f"<p>{_inline(html.escape(' '.join(para)))}</p>")
        i += 1

    close_list()
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f"<style>{CSS}</style></head><body>" + "\n".join(out) + "</body></html>"
    )


def find_browser() -> str | None:
    for path in CHROME:
        if Path(path).exists():
            return path
    return shutil.which("chromium") or shutil.which("google-chrome")


def convert(md_path: Path, out_dir: Path, browser: str) -> Path:
    """One Markdown file to one PDF, via a temporary HTML beside the output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / (md_path.stem + ".html")
    pdf_path = out_dir / (md_path.stem + ".pdf")
    html_path.write_text(to_html(md_path.read_text(), md_path.stem))
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    html_path.unlink(missing_ok=True)
    return pdf_path
