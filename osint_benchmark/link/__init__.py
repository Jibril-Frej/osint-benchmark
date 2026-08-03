"""Step 2: annotate documents and records with the Wikidata entities they mention.

Two different problems wearing one name. Prose needs a *linker* -- a model that reads text,
finds mentions and resolves them (:mod:`refined` for English, and mGENRE for German, which
is not ported yet). Tabular sources need *reconciliation*: they already carry a name or an
external code, and resolving it is a lookup (:mod:`reconcile`).

Keeping them apart matters because their failure modes differ. A linker mislinks
plausibly; a lookup either matches or does not.
"""
