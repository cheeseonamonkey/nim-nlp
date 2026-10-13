# nim-nlp

A lightweight, dependency-free natural-language processing toolkit for Nim.

`nim-nlp` provides practical text preprocessing, linguistic normalization,
classic stemming, corpus analysis, TF-IDF, BM25 retrieval, and sparse vectors.
It is implemented in pure Nim and targets Nim 2.x.

## Features

- Word, whitespace, sentence, simple-pattern, and Unicode character n-gram tokenizers
- Lowercasing, accent stripping, whitespace, punctuation, number, HTML, URL, and email normalization
- English, Spanish, French, and German stopwords
- Porter and irregular-form-aware smart stemming
- Term/document frequencies, statistics, keywords, and similarity measures
- Bag-of-words / CountVectorizer with cosine search
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
```

## Linguistic data

The built-in linguistic data (stopword lists, irregular plurals, irregular
verbs, and no-change words) is compiled into the library as constants in
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
