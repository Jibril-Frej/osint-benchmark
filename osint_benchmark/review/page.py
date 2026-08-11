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
.prov { font-size: 12px; color: #666; margin-bottom: 10px; font-family: monospace; }
.filters { position: sticky; top: 0; background: #fff; padding: 10px 0; z-index: 10;
           border-bottom: 1px solid #ddd; margin-bottom: 16px; }
.filters button { margin-right: 6px; padding: 5px 11px; border: 1px solid #bbb;
                  background: #f6f6f6; border-radius: 3px; cursor: pointer; font-size: 13px; }
.filters button.on { background: #2b6cb0; color: #fff; border-color: #2b6cb0; }
.models { font-size: 12px; color: #555; background: #f7f7f7; border-left: 3px solid #bbb;
          padding: 8px 12px; margin-bottom: 16px; }
.models p { margin: 0 0 4px; }
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
function showOnly(kind, button) {
  const bar = document.querySelector('.filters');
  if (kind === 'all') {
    for (const b of bar.querySelectorAll('button')) b.classList.add('on');
  } else {
    bar.querySelector('[data-type="all"]').classList.remove('on');
    button.classList.toggle('on');
    if (!bar.querySelector('button.on')) button.classList.add('on');
  }
  const wanted = new Set();
  for (const b of bar.querySelectorAll('button.on')) wanted.add(b.dataset.type);
  for (const el of document.querySelectorAll('article')) {
    el.style.display = wanted.has('all') || wanted.has(el.dataset.type) ? '' : 'none';
  }
}
for (const b of document.querySelectorAll('.filters button')) {
  b.addEventListener('click', () => showOnly(b.dataset.type, b));
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


def _cited(evidence: list, texts: dict[str, str]) -> str:
    """Return the text an item actually cites, which may be a span of a document.

    An item that cites offsets is pointing at a passage, and showing the whole document
    instead makes a reviewer hunt for the sentence the question was built from. The typed
    questions all cite a span; the bridge ones cite whole documents and are unaffected.
    """
    parts = []
    for cite in evidence:
        text = texts.get(cite.doc_id, "")
        if not text:
            continue
        if cite.offsets and cite.offsets[1] > cite.offsets[0]:
            start, end = cite.offsets
            opened = "… " if start else ""
            closed = " …" if end < len(text) else ""
            parts.append(f"{opened}{text[start:end]}{closed}")
        else:
            parts.append(text)
    return " ".join(parts)


def _how(item: Item) -> str:
    """Return how the item was arrived at, for a reviewer deciding whether to believe it.

    A typed question's answer is computed, so what a reviewer has to check is the
    computation: which two people, through which predicate, or which namesakes the linker
    was choosing between and by how much its choice won.
    """
    shown = {key: value for key, value in sorted(item.provenance.items()) if value}
    if not shown:
        return ""
    pairs = " &middot; ".join(
        f"{html.escape(key)}: {html.escape(str(value))}" for key, value in shown.items()
    )
    return f'<div class="prov">{pairs}</div>'


def _article(item: Item, texts: dict[str, str]) -> str:
    """Return one question's block."""
    private = _cited(item.private_evidence, texts)
    public = _cited(item.public_evidence, texts)
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
    kind = html.escape(item.question_type)
    return f"""<article id="{html.escape(item.item_id)}" data-type="{kind}">
  <div class="q">{html.escape(item.question)}</div>
  <div class="a">Answer: {html.escape(item.answer)}</div>
  <div class="meta">{html.escape(item.question_type)} &middot; {html.escape(ids)}</div>
  {_how(item)}
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


def _filters(items: list[Item]) -> str:
    """Return the type filter bar, with a count beside each type."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item.question_type] = counts.get(item.question_type, 0) + 1
    buttons = [f'<button class="on" data-type="all">all ({len(items)})</button>']
    buttons += [
        f'<button class="on" data-type="{html.escape(kind)}">{html.escape(kind)} ({count})</button>'
        for kind, count in sorted(counts.items())
    ]
    return f'<div class="filters">{"".join(buttons)}</div>'


def _models(note: str) -> str:
    """Return what produced these questions, from the run's own provenance.

    The item carries the phraser, because that is a fact about the item. Which model judged
    it and which measured its necessity are facts about the *run*, recorded in the artefact's
    provenance sidecar -- and a reviewer weighing a verdict needs to know whose verdict it
    is, so the page says rather than leaving it a file away.
    """
    if not note:
        return ""
    return (
        '<div class="models"><p><strong>How these were produced</strong></p>'
        f"<p>{html.escape(note)}</p></div>"
    )


def render(items: Iterable[Item], texts: dict[str, str], note: str = "") -> str:
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
{_models(note)}
{_filters(listed)}
{body}
</main>
<script>{SCRIPT.replace("ITEMS", payload)}</script>
</body></html>"""
