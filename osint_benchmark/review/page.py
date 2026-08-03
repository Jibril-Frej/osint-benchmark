"""Render the questions into one self-contained page for the human pass.

No server and no external asset: a single file that opens from disk, so questions derived
from the confidential corpora never leave the workstation. Verdicts persist in
``localStorage`` and export as JSON, which is fed back to filter the accepted pool.

Each question is shown beside both documents it rests on, its gate outcomes and its three
necessity flags. The flags are shown rather than used as a filter, because which condition
succeeded says something different in each case and that is a judgement for the reviewer.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable

from osint_benchmark.generate.item import Item

STYLE = """
body { font: 15px/1.55 system-ui, sans-serif; margin: 0; background: #f6f6f4; color: #1a1a1a; }
header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd;
         padding: 12px 20px; display: flex; gap: 16px; align-items: center; }
h1 { font-size: 16px; margin: 0; font-weight: 600; }
#counts { color: #666; font-size: 13px; }
button { font: inherit; padding: 5px 12px; border: 1px solid #bbb; border-radius: 5px;
         background: #fff; cursor: pointer; }
main { padding: 20px; max-width: 1100px; margin: 0 auto; }
article { background: #fff; border: 1px solid #ddd; border-radius: 7px; margin-bottom: 18px;
          padding: 16px 18px; }
article.keep { border-left: 5px solid #2e7d32; }
article.drop { border-left: 5px solid #c62828; }
.q { font-size: 17px; font-weight: 600; margin-bottom: 6px; }
.a { color: #14532d; margin-bottom: 10px; }
.meta { font-size: 12px; color: #666; margin-bottom: 10px; }
.flag { display: inline-block; padding: 1px 7px; border-radius: 3px; margin-right: 5px;
        font-size: 11px; border: 1px solid #ccc; }
.bad { background: #fdecea; border-color: #f5c6c2; }
.good { background: #edf7ed; border-color: #c8e6c9; }
.evidence { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px; }
.evidence section { background: #fafafa; border: 1px solid #eee; border-radius: 5px;
                    padding: 10px; max-height: 260px; overflow: auto; }
.evidence h3 { margin: 0 0 6px; font-size: 12px; text-transform: uppercase; color: #777; }
pre { white-space: pre-wrap; font: 12px/1.5 ui-monospace, monospace; margin: 0; }
.actions { margin-top: 12px; display: flex; gap: 8px; }
"""

SCRIPT = """
const items = ITEMS;
const store = window.localStorage;
function decide(id, verdict) {
  if (store.getItem(id) === verdict) { store.removeItem(id); } else { store.setItem(id, verdict); }
  paint();
}
function paint() {
  let keep = 0, drop = 0;
  for (const item of items) {
    const el = document.getElementById(item.item_id);
    const verdict = store.getItem(item.item_id);
    el.className = verdict || '';
    if (verdict === 'keep') keep++; else if (verdict === 'drop') drop++;
  }
  document.getElementById('counts').textContent =
    `${keep} kept, ${drop} dropped, ${items.length - keep - drop} undecided of ${items.length}`;
}
function exportVerdicts() {
  const out = {};
  for (const item of items) { const v = store.getItem(item.item_id); if (v) out[item.item_id] = v; }
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'verdicts.json'; a.click();
}
paint();
"""


def _flag(label: str, ok: bool) -> str:
    """Return one coloured flag."""
    return f'<span class="flag {"good" if ok else "bad"}">{html.escape(label)}</span>'


def _article(item: Item, texts: dict[str, str]) -> str:
    """Return one question's block."""
    private = " ".join(texts.get(e.doc_id, "") for e in item.private_evidence)
    public = " ".join(texts.get(e.doc_id, "") for e in item.public_evidence)
    gates = "".join(_flag(name, ok) for name, ok in sorted(item.gates.items()))
    necessity = "".join(
        _flag(f"{name} answerable", not value)
        for name, value in (
            ("closed-book", item.necessity.closed_book),
            ("public-only", item.necessity.public_only),
            ("private-only", item.necessity.private_only),
        )
        if value is not None
    )
    ids = ", ".join(f"{e.side}:{e.doc_id}" for e in item.evidence)
    return f"""<article id="{html.escape(item.item_id)}">
  <div class="q">{html.escape(item.question)}</div>
  <div class="a">Answer: {html.escape(item.answer)}</div>
  <div class="meta">{html.escape(item.question_type)} &middot; {html.escape(ids)}</div>
  <div>{gates}{necessity}</div>
  <div class="evidence">
    <section><h3>Confidential</h3><pre>{html.escape(private) or "(not available)"}</pre></section>
    <section><h3>Public</h3><pre>{html.escape(public) or "(not available)"}</pre></section>
  </div>
  <div class="actions">
    <button onclick="decide('{html.escape(item.item_id)}','keep')">Keep</button>
    <button onclick="decide('{html.escape(item.item_id)}','drop')">Drop</button>
  </div>
</article>"""


def render(items: Iterable[Item], texts: dict[str, str]) -> str:
    """Return the whole review page as one self-contained HTML document."""
    listed = list(items)
    payload = json.dumps([{"item_id": i.item_id} for i in listed])
    body = "\n".join(_article(item, texts) for item in listed)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>OSINT benchmark review</title><style>{STYLE}</style></head>
<body>
<header>
  <h1>OSINT benchmark review</h1>
  <span id="counts"></span>
  <button onclick="exportVerdicts()">Export verdicts</button>
</header>
<main>
{body}
</main>
<script>{SCRIPT.replace("ITEMS", payload)}</script>
</body></html>"""
