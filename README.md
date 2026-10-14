# nim-nlp

A lightweight, dependency-free natural-language processing toolkit for Nim.

`nim-nlp` provides practical text preprocessing, linguistic normalization,
classic stemming, corpus analysis, TF-IDF, BM25 retrieval, and sparse vectors.
It is implemented in pure Nim and targets Nim 2.x.

## Features

- Word, whitespace, sentence, simple-pattern, and Unicode character n-gram tokenizers
- Word n-grams as token arrays (`ngrams`, `bigrams`, `trigrams`, `ngramRange`)
- Lowercasing, accent stripping, whitespace, punctuation, number, HTML, URL, email, and English contraction-expansion normalization
- Rune-aware edit distance and fuzzy similarity (Levenshtein, Jaro, Jaro-Winkler, n-gram distance)
- English, Spanish, French, and German stopwords
- Porter and irregular-form-aware smart stemming
- Term/document frequencies, statistics, keywords, and similarity measures (word-bag and order-sensitive n-gram cosine/Jaccard)
- Bag-of-words / CountVectorizer with cosine search
- Stateless HashingVectorizer (feature hashing) with cosine similarity
- TF-IDF vectorization and cosine search
- BM25 retrieval with configurable `k1` and `b`
- LDA topic modeling (collapsed Gibbs) with perplexity and coherence (UMass / c_v)
- Sparse vectors and vocabularies
- Compile-time constant linguistic data (no shipped data files)
- No runtime data files and no external runtime dependencies

## Installation

The repository is named `nim-nlp`; the Nimble package name and the import name
are both `nlp`.

```bash
nimble install nlp
```

## Quick start

```nim
import nlp

let documents = @[
  wordTokenize("the cat sat on the mat"),
  wordTokenize("the dog sat on the log"),
  wordTokenize("cats and dogs are great pets")
]

let keywords = extractKeywords(documents, topN = 5)

let tfidf = fitTfidf(documents)
let tfidfResults = tfidf.searchTfidf(wordTokenize("cat mat"))

let bm25 = fitBm25(documents)
let bm25Results = bm25.searchBm25(wordTokenize("cats pets"))
```

## Preprocessing

```nim
let tokens = "The quick brown foxes jumped"
  .wordTokenize()
  .removeStopWords("english")
  .smartStemTokens()
# @["quick", "brown", "fox", "jump"]

let normalized = "  Café 42  ".normalizeStripAccents()
# "  Cafe 42  "

let cleaned = "Hi <b>Bob</b>, see https://example.com or bob@example.com".normalizeStripHtml()
  .normalizeReplaceUrls()
  .normalizeReplaceEmails()
# "Hi Bob, see <url> or <email>"

let expanded = "I don't think it's ready".normalizeExpandContractions()
# "I do not think it is ready"

var pipe = TextPipeline()
pipe.addStep(Lowercaser())
pipe.addStep(ContractionExpander())
let piped = pipe.normalize("I DON'T know")
# "i do not know"

let bow = fitCounts(documents)
let bowResults = bow.searchCounts(wordTokenize("cat mat"))

let lda = fitLda(documents, numTopics = 2, seed = 42)
let topics = lda.topTerms(0, n = 5)                # phi-ranked labeling
let topicsPmi = lda.topTermsByPMI(documents, 0, 5) # PMI-reranked labeling
let ppl = lda.perplexity()                          # training perplexity
let umass = lda.coherenceUMass(documents)           # UMass coherence
let cv = lda.coherenceCV(documents)                 # c_v coherence (PMI)
let dist = lda.transformLda(wordTokenize("cats and pets")) # fold-in for unseen doc
```

## Analysis

```nim
let frequencies = termFrequencies(wordTokenize("one fish two fish"))
let stats = textStats(wordTokenize("one fish two fish"))
let similarity = cosineSimilarityBags(
  wordTokenize("quick brown fox"),
  wordTokenize("quick red fox")
)

let grams = bigrams(wordTokenize("the quick brown fox"))
# @[@["the", "quick"], @["quick", "brown"], @["brown", "fox"]]
let gramFreqs = ngramFrequencies(wordTokenize("one fish two fish one fish"), 2)
let topGrams = topNgrams(wordTokenize("one fish two fish one fish"), 2, topK = 3)
let shingleSim = ngramCosineSimilarity(  # order-sensitive: 0.0 here, bag cosine is 1.0
  wordTokenize("dog bites man"),
  wordTokenize("man bites dog"), 2)

let dist = levenshtein("kitten", "sitting")        # 3
let fuzzy = jaroWinkler("martha", "marhta")         # ~0.961
let hashed = transformHashing(wordTokenize("cat mat")) # hashed SparseVector
let wins = windows(@[1, 2, 3, 4], 2)                 # @[@[1, 2], @[2, 3], @[3, 4]]
```

## Public API

`import nlp` re-exports all of the following. Overloads and labeled
parameters are omitted here; see the module doc comments for full signatures.

**`tokenize`**
- `wordTokenize`, `whitespaceTokenize`, `ngramTokenize`, `splitSentences` — convenience tokenizers for words, whitespace, character n-grams, and sentences.
- `WordTokenizer`, `WhitespaceTokenizer`, `PatternTokenizer`, `NGramTokenizer`, `SentenceTokenizer`, `Tokenizer` — configurable tokenizer types, each with `tokenize`.
- `flushToken`, `isSentenceBoundary` — the shared token-drain and boundary-test helpers the tokenizers derive from.

**`ngrams`**
- `ngrams`, `bigrams`, `trigrams`, `ngramRange` — word-level n-grams returned as arrays of component tokens (`seq[seq[string]]`).
- `windows` — the generic sliding-window base everything above derives from; `joinGrams` — escape hatch back to joined-string form.

**`distance`**
- `levenshtein`, `levenshteinSimilarity` — rune-aware edit distance and its normalized similarity.
- `jaro`, `jaroWinkler` — Jaro and Jaro-Winkler fuzzy similarity (configurable `prefixWeight` and `maxPrefix`).
- `levenshteinRunes`, `jaroRunes` — the rune-level bases the string versions delegate to (decode once, reuse).
- `ngramDistance` — case-sensitive character-n-gram Jaccard distance.

**`hashing`**
- `HashingVectorizer` / `newHashingVectorizer` — stateless feature-hashing vectorizer (`dim`, `signed`); hashed ids are stable within a process but not guaranteed across Nim versions, so re-hash corpus and queries together.
- `transform`, `transformHashing` — hash token sequences into `SparseVector`s; compare with `cosineSimilarity`.
- `hashBucket`, `hashSign` — the exposed bucket/sign primitives (building blocks for MinHash/LSH-style sketches).

**`normalizers`**
- `normalizeLowercase`, `normalizeStripAccents`, `normalizeWhitespace`, `normalizeStripPunctuation`, `normalizeRemoveNumbers`, `normalizeStripHtml`, `normalizeReplaceUrls`, `normalizeReplaceEmails` — one-call text normalizers.
- `Lowercaser`, `AccentStripper`, `WhitespaceNormalizer`, `PunctuationStripper`, `NumberNormalizer`, `HtmlStripper`, `UrlReplacer`, `EmailReplacer`, `CustomNormalizer`, `ContractionExpander` — normalizer types, each with a `normalize` overload (`normalizeExpandContractions` is the contraction convenience; possessives and `o'clock` are left untouched).
- `Normalizer`, `TextPipeline`, `addStep` — the `Normalizer` concept every type above satisfies, plus a composable, nestable pipeline (`normalize` on a pipeline runs its steps in order).
- `chain` — compose two normalizer procs into one.

**`stopwords`**
- `isStopWord`, `removeStopWords` — language-aware filtering, or filtering against a custom `StopWords` set.
- `englishStopWords`, `spanishStopWords`, `frenchStopWords`, `germanStopWords` — built-in sets.

**`porter_stemmer` / `smart_stemmer`**
- `stemWord`, `stemTokens` — Porter stemming; `step1ab`, `step1c`, `step2`, `step3`, `step4`, `step5ab` — the reference-algorithm step groupings `stem` derives from.
- `SmartStemmer`, `stem`, `smartStemWord`, `smartStemTokens` — irregular-form-aware stemming; `irregularPlurals`, `irregularVerbs`, and `noChangeWords` expose the lookup data.
- `lookupIrregular`, `detectPlural`, `smartStepEdIng`, `smartStepY`, `smartStep3/4/5`, `smartStepFinalE`, `smartStepDoubleL` — the smart pipeline stages.

**`analysis`**
- `termFrequencies`, `documentFrequency`, `idfSmoothed`, `sortedTermsByFrequency` — frequency helpers.
- `jaccardSimilarity`, `cosineSimilarityBags`, `overlapCoefficient` — similarity measures.
- `ngramFrequencies`, `topNgrams` — n-gram counting and ranking (frequency descending, deterministic tie-break).
- `ngramCosineSimilarity`, `ngramJaccardSimilarity` — order-sensitive shingle similarity over word n-grams.
- `countItems`, `cosineCounts`, `jaccardSets`, `overlapSets` — the generic counting/similarity bases every measure above derives from.
- `textStats` / `TextStats` — token, unique-term, hapax, and average-length statistics.
- `uniqueTerms`, `extractKeywords` — corpus vocabulary and keyword extraction.

**`bow`**
- `fitCounts`, `transformCounts`, `searchCounts` — bag-of-words count model (`CountModel`) with cosine search.

**`embeddings`**
- `fitTfidf`, `transformTfidf`, `searchTfidf` — TF-IDF model (`TfidfModel`).
- `fitBm25`, `searchBm25` — BM25 retrieval (`BM25Model`, configurable `k1` and `b`).

**`topics`**
- `fitLda`, `transformLda` — LDA topic model (`LdaModel`): training and fold-in inference for unseen documents.
- `docTopicDistribution`, `topicWordDistribution` — per-document and per-topic distributions.
- `topTerms`, `topTermsByPMI` — topic labeling by phi or by PMI reranking.
- `perplexity`, `coherenceUMass`, `coherenceCV` — model evaluation.
- `sortedPairKey`, `meanPmiScores`, `meanPairwisePmi`, `normalizeUnit`, `umassPairScore` — the coherence building blocks.

**`types`**
- `SparseVector` / `newSparseVector` — index-to-value vectors with `set`, `get`, `containsIdx`, `clear`, `len`, `pairs`, `values`, and `cosineSimilarity`.
- `Vocabulary` / `newVocabulary` — term-to-id maps with `add`, `id`, `contains`, `containsIdx`, `[]`, `clear`, and `len`.
- `Term` — the string alias used for terms throughout the API.

## Linguistic data

The built-in linguistic data (stopword lists, irregular plurals, irregular
verbs, no-change words, and the English contraction table) is compiled into the library as constants in
`src/nlp/private/lexicon.nim`. That file is the editable source of truth: change
it and rebuild to change the built-in lists.

No data files are shipped or required at install, compile, or run time. The
project does not bundle an arbitrary global English frequency corpus; frequency
data varies by corpus and language variety.

## Development

```bash
nimble test                 # run the split test suite
./scripts/install-smoke.sh  # install into an isolated NIMBLE_DIR and compile a
                            # standalone consumer against the installed package
```

Contributions are welcome. Please keep the library dependency-free, add tests
for behavior changes, and run `nimble check` and `nimble test` before opening a
pull request.

The public module is named `nlp`; TF-IDF remains available through APIs such as
`fitTfidf`, `transformTfidf`, and `searchTfidf`.

## License

MIT
