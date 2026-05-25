# Correction Memory

Correction memory remains a runtime concern, not a training mechanism.

In the AST-first architecture it may store accepted, rejected, ignored, or
manual decisions for a concrete correction in a concrete context. It must not
modify RuRoBERTa weights and must not expand the strict project scope beyond
Russian spelling and punctuation.

The future runtime boundary is a simple ScopeGuard. Memory-selected behavior
will still pass through runtime scope checks before edits are applied.
