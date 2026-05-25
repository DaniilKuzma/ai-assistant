# API

The public user workflow is unchanged during this cleanup:

1. User enters text.
2. User clicks `Исправить`.
3. The application returns corrected text plus edits.

The internal target API is changing toward direct edit tagging:

- generated examples will come from `src.grammar_gen`;
- label/edit schemas will live in `src.schema`;
- deterministic-first runtime orchestration will live in `src.runtime`.

Existing inference modules are retained only as transition code until the direct
tagger runtime lands.
