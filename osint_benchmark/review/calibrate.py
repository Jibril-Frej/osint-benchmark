"""Check the pipeline's own verdicts against a person, on a small sample.

The review page asks "is this question any good?". This asks a different and prior
question: **can the automated measurements be believed?**

Two of them decide everything downstream. The judge says the two documents support the
answer. The ablations say neither document alone suffices. Every figure the benchmark
reports rests on those, and neither has ever been checked against a human — the first
review confirmed the formulations read well, which is the part that needed checking least.

So a dozen questions, deliberately half from each side of the necessity verdict, each
showing what the pipeline concluded and asking only whether that conclusion was right.
Agreement means the numbers scale and nobody hand-checks again. Disagreement is more
useful still: it says which measurement is broken, and a sample stratified this way says
*which direction* it is broken in — a measure that is right about the questions it passes
and wrong about the ones it fails is a different problem from the reverse.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable

from osint_benchmark.generate.item import Item
from osint_benchmark.review.page import STYLE

EXTRA_STYLE = """
.claim { border-top: 1px solid #eee; margin-top: 12px; padding-top: 10px; }
.claim p { margin: 0 0 8px; }
.says { font-weight: 600; }
.agree { color: #2e7d32; } .disagree { color: #c62828; }
article.done { border-left: 5px solid #2e7d32; }
"""

SCRIPT = """
const items = ITEMS;
const store = window.localStorage;
function answer(id, claim, value) {
  const key = id + '::' + claim;
  if (store.getItem(key) === value) { store.removeItem(key); } else { store.setItem(key, value); }
  paint();
}
function paint() {
  let done = 0, agree = 0, disagree = 0;
  for (const item of items) {
    let answered = 0;
    for (const claim of ['grounded', 'necessary']) {
      const v = store.getItem(item.item_id + '::' + claim);
      if (v) { answered++; if (v === 'agree') agree++; else disagree++; }
    }
    const el = document.getElementById(item.item_id);
    el.className = answered === 2 ? 'done' : '';
    if (answered === 2) done++;
  }
  document.getElementById('counts').textContent =
    `${done} of ${items.length} checked · ${agree} agree, ${disagree} disagree`;
}
function exportVerdicts() {
  const out = {};
  for (const item of items) {
    for (const claim of ['grounded', 'necessary']) {
      const v = store.getItem(item.item_id + '::' + claim);
      if (v) { (out[item.item_id] = out[item.item_id] || {})[claim] = v; }
    }
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'calibration.json'; a.click();
}
paint();
"""


def sample(items: Iterable[Item], size: int = 12) -> list[Item]:
    """Return a sample split evenly between the two necessity verdicts.

    Stratified rather than random, because the interesting failure is asymmetric: a
    measurement that is right about what it passes and wrong about what it fails would look
    fine in a random sample dominated by whichever verdict is commoner.

    Deterministic — sorted by item id, taken from each stratum in turn — so the same sample
    comes back on a rerun and two people check the same questions.
    """
    ordered = sorted((i for i in items if i.necessity.measured), key=lambda i: i.item_id)
    needs = [i for i in ordered if i.necessity.needs_both]
    others = [i for i in ordered if not i.necessity.needs_both]
    half = size // 2
    chosen = needs[:half] + others[: size - half]
    # If one stratum is short, take the shortfall from the other rather than a small sample.
    if len(chosen) < size:
        remaining = [i for i in ordered if i not in chosen]
        chosen += remaining[: size - len(chosen)]
    return sorted(chosen, key=lambda i: i.item_id)


def _claim(item: Item, name: str, question: str, says: str) -> str:
    """Return one claim block: what the pipeline concluded, and did it get it right."""
    item_id = html.escape(item.item_id)
    return f"""<div class="claim">
  <p>{html.escape(question)}<br><span class="says">The pipeline says: {html.escape(says)}</span></p>
  <button class="agree" onclick="answer('{item_id}','{name}','agree')">Agree</button>
  <button class="disagree" onclick="answer('{item_id}','{name}','disagree')">Disagree</button>
</div>"""


def _article(item: Item, texts: dict[str, str]) -> str:
    """Return one question's block, framed as two checks rather than a keep/drop."""
    private = " ".join(texts.get(e.doc_id, "") for e in item.private_evidence)
    public = " ".join(texts.get(e.doc_id, "") for e in item.public_evidence)
    needs = item.necessity
    alone = [
        name
        for name, value in (
            ("world knowledge alone", needs.closed_book),
            ("the public record alone", needs.public_only),
            ("the confidential document alone", needs.private_only),
        )
        if value
    ]
    verdict = (
        "both documents are needed"
        if needs.needs_both
        else "answerable from " + " and ".join(alone)
    )
    return f"""<article id="{html.escape(item.item_id)}">
  <div class="q">{html.escape(item.question)}</div>
  <div class="a">Answer: {html.escape(item.answer)}</div>
  <div class="meta">{html.escape(item.item_id)}</div>
  <div class="evidence">
    <section><h3>Confidential</h3><pre>{html.escape(private) or "(not available)"}</pre></section>
    <section><h3>Public</h3><pre>{html.escape(public) or "(not available)"}</pre></section>
  </div>
  {
        _claim(
            item,
            "grounded",
            "Do these two documents actually support that answer?",
            "yes, the judge verified it",
        )
    }
  {_claim(item, "necessary", "Is that the right call about what is needed?", verdict)}
</article>"""


def render(items: Iterable[Item], texts: dict[str, str]) -> str:
    """Return the calibration page as one self-contained HTML document."""
    listed = list(items)
    payload = json.dumps([{"item_id": i.item_id} for i in listed])
    body = "\n".join(_article(item, texts) for item in listed)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>OSINT benchmark calibration</title><style>{STYLE}{EXTRA_STYLE}</style></head>
<body>
<header>
  <h1>Calibration &mdash; is the pipeline right about these?</h1>
  <span id="counts"></span>
  <button onclick="exportVerdicts()">Export</button>
</header>
<main>
<p>Two claims per question. Read both documents, then say whether the pipeline got it
right. This is not about whether the question is any good.</p>
{body}
</main>
<script>{SCRIPT.replace("ITEMS", payload)}</script>
</body></html>"""
