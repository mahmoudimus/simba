"""Intent-primed doctrine + mandated preflight (spec 28).

The doctrine-triggers store + the cheap (embedding-match, no-LLM) intent
classifier that primes the right doctrine from a stated intent at
``UserPromptSubmit``, plus the ``simba preflight`` machinery the PreToolUse gate
enforces.
"""
