# Syntax Punctuation Layer

`syntax_punctuation` is a YAML-backed RuleLayer for controlled punctuation-only
examples. It emits direct gap labels and never rewrites word tokens.

Covered groups:

- final marks: `DOT`, `QUESTION`, `EXCLAMATION`, `ELLIPSIS`;
- syntax dash: noun/numeral/infinitive predicates, `это`/`вот`/`значит`,
  and no-dash/no-comma guards;
- homogeneous members, generalizing words with `COLON`/`DASH`, and
  homogeneous/heterogeneous definition guards;
- detached definitions, appositions, adverbials, clarifying time/place, and
  fixed-expression guards;
- comparative turns and controlled `как` hard negatives;
- introductory words/constructions, address, interjection, and particle cases;
- compound, subordinate, and asyndetic complex sentence patterns;
- extra punctuation deletion through `DELETE_PUNCTUATION`.

Outside this layer:

- quotes, brackets, citations, and full direct speech;
- paired punctuation and span-level quote/bracket labels;
- comma+dash combinations, because one gap currently carries only one label.

