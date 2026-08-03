"""Model access: what to say (``prompts``), how to say it (``settings``), and to whom.

Both halves are deliberately outside the Python. Prompts are files in ``prompts/`` and
parameters are files in ``config/``, so the whole model surface of the project can be
reviewed by reading two directories rather than by grepping for f-strings.
"""
