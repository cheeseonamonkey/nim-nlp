# Changelog

## 0.3.0

- **BREAKING**: `ngrams` now returns `seq[seq[string]]` (arrays of component
  tokens, e.g. `@[@["the", "quick"]]`) instead of space-joined strings, and
  lives in the new `nlp/ngrams` module alongside `bigrams`, `trigrams`, and
  `ngramRange`. Update call sites that pinned the old joined-string shape.
- Added n-gram analysis to `analysis`: `ngramFrequencies`, `topNgrams`,
  `ngramCosineSimilarity` (count-bag cosine over shingles, captures word
  order), and `ngramJaccardSimilarity`.
- Added `nlp/distance`: rune-aware `levenshtein`, `levenshteinSimilarity`,
  `jaro`, `jaroWinkler`, and `ngramDistance` (built on `ngramTokenize` +
  `jaccardSimilarity`).
- Added the `ContractionExpander` normalizer and `normalizeExpandContractions`
  convenience (`don't` → `do not`), backed by a 93-entry table in
  `private/lexicon.nim` (99 entries, including leading-apostrophe forms
  like `'tis` and `'em`); whole-token matching leaves possessives and
  `o'clock` untouched.
- Added `nlp/hashing`: stateless `HashingVectorizer` (`newHashingVectorizer`,
  `transform`, `transformHashing`) with optional signed (+1/-1) hashing; pairs
  with the existing `cosineSimilarity` on `SparseVector`.
- Exposed the generic bases and derived everything from them (no behavior
  change): `windows`/`joinGrams` (backing `ngrams` and `NGramTokenizer`),
  `countItems`/`cosineCounts`/`jaccardSets`/`overlapSets` (backing all
  frequency and similarity measures), `topRanked` (backing `topScored` and
  `topNgrams`), `levenshteinRunes`/`jaroRunes` (backing the string
  distances), `hashBucket`/`hashSign` (backing `transform`).
- `ngramDistance` is now case-sensitive (previously lowercased via
  `ngramTokenize`), consistent with the word-level similarities.
- **Fixed**: smart stemmer `measure` never returned its count (always 0),
  so step-3/4/5 table rules, the `-eed → -ee` rewrite, final-`-e` strip,
  and double-`l` cleanup never fired. Smart outputs change accordingly
  (e.g. `houses` → `hous`, `-ies` plurals → `-i` like Porter: `cities` →
  `citi`); the `cities` divergence pin in `t_edgecases` now asserts
  convergence, while `flies` (`fli` vs `fly`) still diverges by design.
- `Normalizer` concept + composable, nestable `TextPipeline` (`addStep`,
  `normalize`); normalizer scanners and the smart/porter step internals
  decomposed into named, pinned helpers (no behavior change).
- Tokenizer, LDA, and coherence internals decomposed (`flushToken`,
  `isSentenceBoundary`, `fitLda` phases, PMI/co-occurrence helpers;
  no math or RNG-order change).
- Case folding is now Unicode-aware (`unicode.toLower`) everywhere:
  tokenizers, `Lowercaser`, contraction matching, stemmers, and stopwords.
  ASCII behavior is unchanged; `"CAFÉ"` now folds to `"café"` (previously
  `"cafÉ"`), so e.g. `isStopWord("À", "french")` is true. Char-level
  ASCII fast paths that are provably ASCII-domain (contraction key
  building, expansion capitalization) are intentionally kept.

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
