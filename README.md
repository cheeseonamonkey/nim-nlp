# nim-nlp

A lightweight, dependency-free natural-language processing toolkit for Nim.

`nim-nlp` provides practical text preprocessing, linguistic normalization,
classic stemming, corpus analysis, TF-IDF, BM25 retrieval, and sparse vectors.
It is implemented in pure Nim and targets Nim 2.x.

## Features

- Word, whitespace, sentence, pattern, and Unicode character n-gram tokenizers
- Lowercasing, accent stripping, whitespace, punctuation, and number normalization
- English, Spanish, French, and German stopwords
- Porter and irregular-form-aware smart stemming
- Term/document frequencies, statistics, keywords, and similarity measures
- TF-IDF vectorization and cosine search
- BM25 retrieval with configurable `k1` and `b`
- Sparse vectors and vocabularies
- CSV-maintained linguistic data embedded at compile time
- No runtime data files and no external runtime dependencies

## Installation

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

The editable source lists are in `data/`:

- `stopwords.csv`
- `irregular_plurals.csv`
- `irregular_verbs.csv`
- `no_change_words.csv`

They are embedded during compilation and parsed with Nim's standard `parsecsv`
module. The project does not bundle an arbitrary global English frequency corpus;
frequency data varies by corpus and language variety.

## Development

```bash
nimble test
```

The public module is named `nlp`; TF-IDF remains available through APIs such as
`fitTfidf`, `transformTfidf`, and `searchTfidf`.

## License

MIT
