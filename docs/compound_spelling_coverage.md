# Compound Spelling Coverage

`compound_spelling_layer` is a controlled online-generation layer. It trains
direct labels for bounded Russian spelling cases and does not claim full
semantic disambiguation for unrestricted text.

| Area | Rule id | Status | Notes |
| --- | --- | --- | --- |
| Service words: `также/так же`, `тоже/то же`, `чтобы/что бы`, `зато/за то`, causal/interrogative pairs | `compound_service_words` | Controlled/generated, model-assisted at runtime | Legacy direct labels are used for old overlap pairs; new pairs use `SPAN_REPLACE_BY_LEXICON`. Hard negatives cover comparison and preposition/pronoun readings. |
| Derived prepositions: `в течение`, `вследствие`, `насчёт`, `ввиду`, `наподобие`, `навстречу`, `несмотря на`, `невзирая на` | `compound_prepositions` | Controlled/generated, model-assisted at runtime | Executable examples and runtime lexicon entries exist; river/current, bank-account, and physical-looking contexts are hard negatives. |
| Pronouns and particles: `кое-`, `-то`, `-либо`, `-нибудь`, `ну-ка`, `всё-таки` | `compound_pronouns_particles` | Controlled/generated | Existing specialized labels are used for `кое-` and indefinite-pronoun particles. Other particle cases use generic span replacement. |
| Adverbs: `по-...-ому`, `по-...-ски`, repeated adverbs, `во-первых` | `compound_adverbs` | Controlled/generated | `по-русски` uses the legacy direct label; other controlled cases use generic span replacement. |
| Compound nouns/adjectives: appositions, first parts, coordinate/directional/color adjectives | `compound_nouns_adjectives` | Controlled/generated, model-assisted at runtime | The layer covers selected executable examples such as `врач-терапевт`, `мини-отель`, `социально-экономический`, `северо-западный`, `ярко-красный`. Broad dictionary coverage remains model-assisted. |
| `пол-` / `полу-` | `compound_pol_polu` | Controlled/generated, model-assisted at runtime | Covers vowel, `л`, proper-name, and `полу-` controlled cases. |
| `не` with verbs, nouns, adjectives, adverbs, participles | `compound_ne_spellings` | Controlled/generated, model-assisted at runtime | Verbs use `SPLIT_NE_VERB`. Other controlled examples use generic span replacement. Opposition, dependent words, and `далеко/вовсе/отнюдь/ничуть`-style contexts remain hard negatives. |

Runtime entries live in
`lexicon/layers/compound_spelling/corrections.yaml`. Ambiguous lexicon matches
remain no-op, and `replace` operations are reserved for single-token
`DICT_REPLACE` corrections.
