You are writing one question for an intelligence-analysis benchmark.

You are given a confidential report and a public record that concern the same subject. Write
a question that can only be answered by combining them, and give its answer.

The confidential document:
{private_evidence}

The public record:
{public_evidence}

The subject both concern:
{bridge}

Work in this order.

**First**, decide what each document tells you about that subject that the other does not.
Be specific: name the fact, not the topic. If one of them says nothing particular about the
subject, or if both say the same thing, there is nothing to build on — reply with empty
question and answer fields, and say why.

**Then** write a question whose answer requires both of those facts together, and test it
before you commit to it:

- Could someone holding only the confidential document answer it? Then it is wrong.
- Could someone holding only the public record answer it? Then it is wrong.
- Could someone answer it from general knowledge, without either? Then it is wrong.

A question that fails any of those is not a harder version of the right question. It is the
wrong question, and rewriting it is the work.

The question is asked by {asker}.

Requirements:
- Do not name or paraphrase the answer in the question.
- Do not refer to the sources themselves. Never write "the cable", "the document", "the
  report", "both documents", "as described in", or any other phrase that points at where
  the information came from. Ask about the world, not about what you were given.
- Do not ask what two things "have in common" or "what connects" them. Two subjects that
  merely both appear are not connected; ask something whose answer is a fact.
- Do not ask for an encyclopaedia attribute — a birthplace, a founding year, a capital.
  Ask something an analyst would actually need to know.
- The answer must be a specific fact stated by the two documents together, not a general
  characterisation that either document would support on its own.

Reply as JSON with exactly these keys, in this order, and nothing else:
{{"from_confidential": "the specific fact only the confidential document gives",
"from_public": "the specific fact only the public record gives", "question": "...",
"answer": "..."}}
