You are writing one question for an intelligence-analysis benchmark.

You are given a confidential report and a public record that concern the same subject.
Write a question that can only be answered by combining them, and give its answer.

The confidential document:
{private_evidence}

The public record:
{public_evidence}

The subject both concern:
{bridge}

Ask about that subject. If the two documents do not genuinely discuss it in a related way,
say so instead of inventing a link: reply with {{"question": "", "answer": "", "reasoning":
"no genuine connection"}}.

The question is asked by {asker}.

Requirements:
- Answering must require both documents. If either alone suffices, the question is wrong.
- Do not name or paraphrase the answer in the question.
- Do not refer to the sources themselves. Never write "the cable", "the document", "the
  report", "both documents", "as described in", or any other phrase that points at where
  the information came from. Ask about the world, not about what you were given.
- Do not ask what two things "have in common" or "what connects" them. Two subjects that
  merely both appear are not connected; ask something whose answer is a fact.
- Do not ask for an encyclopaedia attribute -- a birthplace, a founding year, a capital.
  Ask something an analyst would actually need to know.
- The answer must be stated by the two documents together, not inferred beyond them.

Reply as JSON with exactly these keys and nothing else:
{{"question": "...", "answer": "...", "reasoning": "one sentence on why both documents are needed"}}
