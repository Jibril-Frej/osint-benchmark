"""Step 6: turn pairs into questions.

The model writes both the question and its answer from the two documents. That buys the
property a computed answer could not: an answer that comes from combining the two sides,
rather than a public fact the private document merely helps locate. What it costs is the
free correctness check, which :mod:`verify` and repeat-and-agree replace.

Candidate builders never write output. They return candidates; :func:`emit.emit` runs the
gate suite and writes. A new question type therefore cannot bypass a gate, because it has
no path to the file -- which is the structural fix for the previous project's worst
failure, where 82% of a question set had a decorative private hop.
"""
