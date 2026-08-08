You are writing one question for an intelligence-analysis benchmark.

A confidential document places two people in the same situation:

{passage}

The two are {a_label} and {b_label}. Publicly they are not connected to each other, but both
belong to the same organisation, and that membership is a matter of public record.

Write ONE question asking what the two of them both belong to.

Identify them by what the passage shows them doing together — their roles, the meeting, the
decision, the position they took. Do not write either of their names. A question that names
them can be answered by looking the pair up, and then the confidential document is doing no
work.

You do not know what the organisation is, and you must not guess. Do not name any
organisation in the question.

The question is asked by {asker}. Write it the way that person would actually ask a
colleague.

Requirements:
- Ask about the world, not about a document. Never write "the cable", "the record", "the
  document", "according to", "as described in", or any other phrase that says where the
  information came from. Describe the events themselves and stop.
- Do not ask what the two "have in common" or "what connects" them. Ask what body they both
  belong to.
- The question must not contain its own answer.
- Vary the wording. These openings have already been used in this run and you must not
  reuse them: {used}

Reply with only a JSON object and nothing else:
{{"question": "..."}}
