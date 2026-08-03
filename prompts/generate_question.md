You are writing one question for an intelligence-analysis benchmark.

You are given a confidential report and a public record that concern the same subject.
Write a question that can only be answered by combining them, and give its answer.

The confidential document:
{private_evidence}

The public record:
{public_evidence}

What connects them:
{bridge}

The question is asked by {asker}.

Requirements:
- Answering must require both documents. If either alone suffices, the question is wrong.
- Do not name or paraphrase the answer in the question.
- Do not refer to the sources themselves. No "the cable", no "according to the record".
- Do not ask for an encyclopaedia attribute -- a birthplace, a founding year, a capital.
  Ask something an analyst would actually need to know.
- The answer must be stated by the two documents together, not inferred beyond them.

Reply as JSON with exactly these keys and nothing else:
{{"question": "...", "answer": "...", "reasoning": "one sentence on why both documents are needed"}}
