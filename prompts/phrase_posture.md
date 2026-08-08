You build questions for an intelligence-analysis benchmark.

PRIVATE REPORTING, {private_date}, from {private_origin}:
{private_evidence}

PUBLIC RECORD — a Swiss parliamentary proceeding, in German, {public_date}, {public_type}:
{public_title}
{public_evidence}

Decide how the position taken privately compares with the public one, then write ONE
question a real analyst would ask whose answer is that comparison.

The question is asked by {asker}. Write it the way that person would actually ask a
colleague.

QUESTION FORM. It must begin with Did / Does / Is / Was / Were / Has / Have, so a yes-or-no
answer settles it. Never begin with "To what extent" or "How does" — those are essay prompts
and cannot be graded.

Requirements:
- Ask about the world, not about a document. Never write "the cable", "the record", "the
  document", "according to", "as described in", or any other phrase that says where the
  information came from.
- The question must not contain its own answer.
- Vary the wording. These openings have already been used in this run and you must not
  reuse them: {used}

ANSWER, chosen strictly. It answers YOUR question, so read each label as the reply a
colleague would give:

- **Yes** — the public record states the same position as the private one.
- **No** — the public record states a position that contradicts the private one, or plainly
  does not do the thing the question asks about. Use this whenever they genuinely conflict;
  do not soften it to Mixed.
- **Mixed** — the public record contains both agreeing and disagreeing elements. This is the
  narrowest case, not the default.
- **Not enough evidence** — the two documents do not address the same question at all, so no
  comparison is possible. Use this freely; it is a correct answer, not a failure.

Reply with only a JSON object and nothing else:
{{"question": "...", "verdict": "...", "rationale": "..."}}
