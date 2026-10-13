# Changelog

## 0.2.3 (unreleased)

- Split the coherence machinery (`coherenceUMass`, `coherenceCV`,
  `topTermsByPMI`) out of `topics` into a new `coherence` module; the public
  `import nlp` facade is unchanged, but code importing `nlp/topics` directly
  now imports `nlp/coherence` for those three symbols.
- Extracted the shared `topScored` ranking helper (score descending, key
  ascending, truncate to N) and removed five duplicated copies across
  `analysis`, `topics`, and `coherence`; unified the two `extractKeywords`
  method tails.
- Flattened the sentence-tokenizer scan loop and simplified `SparseVector`
  `pairs`/`values` (no behavior change; overlarge differential checks pass).
- Trimmed the edge-case suite to the pins that exist nowhere else (every
  removed assertion verified as covered by another test file).
- Documented the full public API surface in the README, corrected stale
  normalizer comments (accent table coverage, ASCII-only punctuation, HTML
  quote scanning, URL prefix policy, unused `ChainedNormalizer`), and added
  coverage for the previously untested `uniqueTerms` and ranking sharp edges.
- Fixed `normalizeReplaceEmails("a@.")` reporting an email span whose domain
  was emptied by the trailing-dot trim; the domain must still contain a dot
  separating a non-empty host from a non-empty TLD after the trim (D2).
  Also fixed the same root cause for `"a@b."`.
- Fixed `normalizeStripHtml` truncating a tag at a `>` inside a quoted
  attribute value (`<a href="x>y">z</a>`); quoted `>` is now literal text
  until the matching quote closes (D3).
- Resolved the custom `removeStopWords` overload's doc/behavior mismatch: the
  doc now states that tokens are lowercased but the caller's set is used
  verbatim and must contain lowercase entries (D4).
- `transformLda` on an empty or fully out-of-vocabulary document now returns
  the uniform prior (sums to 1) for both no-evidence inputs (D5). As a
  consequence `transformLda` now validates `iterations` before the empty-doc
  short-circuit, so `transformLda(m, @[], iterations = 0)` raises `ValueError`.

## 0.2.1

- Built-in linguistic data (stopwords, irregular plurals/verbs, no-change
  words) is embedded as compile-time constants in `src/nlp/private/lexicon.nim`;
  no data files are shipped or read at install, compile, or run time. This
  removes the install-layout failure where `staticRead("../../data/...")`
  resolved outside the installed package.
- Fixed BM25 returning no results for terms present in about half the corpus
  or more: the probabilistic IDF collapsed to zero (or negative) and the
  `score > 0` filter silently dropped every hit. The weight is now strictly
  positive (Lucene-style +1 inside the logarithm).
- Parameter validation no longer lets NaN slip past comparison-based guards
  (`fitLda` alpha/beta, `searchBm25` k1/b).
- Fixed a potential integer overflow when deriving an LDA fold-in seed from
  long documents; seeds are hashed in unsigned arithmetic.
- Hoisted per-token probability-buffer allocations out of the Gibbs loops in
  `fitLda` and `transformLda`.
- Stemmers refactored to table-driven implementations (verified equivalent:
  341,400 words, 0 output diffs); smart variant renamed `smartStemTokens` so
  `import nlp; stemTokens(tokens)` is no longer ambiguous (D6).
- Test suite split into per-feature files with an edge-case/defect regression
  file; added `scripts/install-smoke.sh`, which installs the package into an
  isolated `NIMBLE_DIR` and compiles a standalone consumer.
- Extracted shared model/analysis helpers (vocabulary construction, ranking,
  stopword language dispatch, co-occurrence, token accumulation).
