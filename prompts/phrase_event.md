You build questions for an intelligence-analysis benchmark.

PRIVATE REPORTING:
{private_evidence}

PUBLIC EVENT RECORDS naming {anchor} within {window} days of that reporting:
{events}

These public records are structured event data — a date, a place, the parties involved, a
count of deaths. They say an incident occurred; they carry no narrative and no account of
what anyone reported about it.

Decide whether the public record bears out the incident the private reporting describes,
then write ONE question a real analyst would ask whose answer is that judgement.

The question is asked by {asker}. Write it the way that person would actually ask a
colleague.

QUESTION FORM. It must begin with Did / Does / Is / Was / Were / Has / Have, so a yes-or-no
answer settles it.

Requirements:
- Ask about the world, not about a document. Never write "the cable", "the record", "the
  document", "according to", "as described in", or any other phrase that says where the
  information came from. Describe the incident itself and stop.
- The question must not contain its own answer.
- Vary the wording. These openings have already been used in this run and you must not
  reuse them: {used}

ANSWER, chosen strictly. It answers YOUR question:

- **Yes** — the public record shows an event matching what the private reporting describes,
  in place and in time.
- **No** — the public record shows the incident did not occur as described: wrong place,
  wrong time, or a materially different event.
- **Mixed** — the records agree on some elements and disagree on others.
- **Not enough evidence** — the public records are about something else, so no comparison is
  possible. Use this freely; it is a correct answer, not a failure.

Reply with only a JSON object and nothing else:
{{"question": "...", "verdict": "...", "rationale": "..."}}
