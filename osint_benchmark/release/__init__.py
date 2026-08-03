"""Step 9: freeze a version of the benchmark.

What ships is what this project authored -- questions, answers, verdicts, necessity flags,
evidence pointers -- never corpus text. Evidence is carried as (doc_id, offsets), so a
release contains nothing that has to be redistributed and a user rebuilds the corpora from
`pins/` instead.
"""
