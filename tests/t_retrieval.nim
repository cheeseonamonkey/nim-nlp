## TF-IDF, BM25, and count-vectorizer tests for the nlp library.

import ../src/nlp
import std/[tables, sets]

proc testEmbeddings() =
  echo "\n=== Embeddings ==="

  let docs = @[
    wordTokenize("the cat sat on the mat"),
    wordTokenize("the dog sat on the log"),
    wordTokenize("cats and dogs are great pets"),
    wordTokenize("the quick brown fox jumps over the lazy dog"),
    wordTokenize("a journey of a thousand miles begins with a single step")
  ]

  # TF-IDF
  var tfidf = fitTfidf(docs)
  assert tfidf.numDocs == 5, "tfidf numDocs"
  assert tfidf.vocabulary.len > 0, "tfidf vocab"

  # TF-IDF transform
  let vec = tfidf.transformTfidf(wordTokenize("the cat is on the mat"))
  assert vec.len > 0, "tfidf transform"

  # TF-IDF empty transform
  let emptyVec = tfidf.transformTfidf(@[])
  assert emptyVec.len == 0, "tfidf transform empty"

  # TF-IDF search
  let results = tfidf.searchTfidf(wordTokenize("cat mat"), topK = 3)
  assert results.len > 0, "tfidf search"
  assert results[0][0] == 0, "tfidf search ranking"

  # TF-IDF empty search
  assert tfidf.searchTfidf(@[]).len == 0, "tfidf search empty"
  assert tfidf.searchTfidf(wordTokenize("cat"), topK = 0).len == 0, "tfidf search topK=0"

  # BM25
  var bm25 = fitBm25(docs)
  assert bm25.numDocs == 5, "bm25 numDocs"

  let bm25Results = bm25.searchBm25(wordTokenize("cats pets"), topK = 3)
  assert bm25Results.len > 0, "bm25 search"
  let once = bm25.searchBm25(@["cat"])
  let twice = bm25.searchBm25(@["cat", "cat"])
  assert twice[0][1] > once[0][1], "bm25 query frequency"

  # BM25 empty search
  assert bm25.searchBm25(@[]).len == 0, "bm25 search empty"
  assert bm25.searchBm25(wordTokenize("cat"), topK = 0).len == 0, "bm25 search topK=0"
  var invalidBm25Raised = false
  try:
    discard bm25.searchBm25(@["cat"], b = 1.5)
  except ValueError:
    invalidBm25Raised = true
  assert invalidBm25Raised, "bm25 parameter validation"

  # Empty model
  var emptyTfidf = fitTfidf(@[])
  assert emptyTfidf.numDocs == 0, "tfidf empty model"
  assert emptyTfidf.searchTfidf(wordTokenize("cat")).len == 0, "tfidf empty model search"

  var emptyBm25 = fitBm25(@[])
  assert emptyBm25.numDocs == 0, "bm25 empty model"
  assert emptyBm25.searchBm25(wordTokenize("cat")).len == 0, "bm25 empty model search"

  # Model with empty docs
  var sparseTfidf = fitTfidf(@[@[], @["hello"], @[]])
  assert sparseTfidf.numDocs == 3, "tfidf sparse docs"
  assert sparseTfidf.vocabulary.len == 1, "tfidf sparse vocab"

  echo "  ALL PASS"

proc testCounts() =
  echo "\n=== Count Vectorizer ==="

  let model = fitCounts(@[@["cat", "cat"], @["dog"]])
  assert model.numDocs == 2, "count model numDocs"
  let vector = model.transformCounts(@["cat", "unknown"])
  assert vector.get(model.vocabulary.id("cat")) == 1.0, "count transform"
  let results = model.searchCounts(@["cat"])
  assert results == @[(0, 1.0)], "count search"

  echo "  ALL PASS"

proc testRetrievalEdges() =
  echo "\n=== Retrieval edges ==="

  let docs = @[
    wordTokenize("the cat sat on the mat"),
    wordTokenize("the dog sat on the log"),
    wordTokenize("cats and dogs are great pets"),
    wordTokenize("the quick brown fox jumps over the lazy dog"),
    wordTokenize("a journey of a thousand miles begins with a single step")
  ]

  # Count vectorizer and TF-IDF share the same vocabulary construction.
  assert fitCounts(docs).vocabulary.len == fitTfidf(docs).vocabulary.len,
    "counts and tfidf share vocabulary size"

  # Equal scores are ordered by ascending docId.
  let tied = fitTfidf(@[@["cat"], @["cat"]]).searchTfidf(@["cat"], topK = 2)
  assert tied.len == 2, "tfidf tie length"
  assert tied[0][0] < tied[1][0], "tfidf tie docId ascending"

  # Fitting on no documents yields an empty search.
  assert fitTfidf(@[]).searchTfidf(@["cat"]).len == 0, "tfidf empty fit search"
  assert fitBm25(@[]).searchBm25(@["cat"]).len == 0, "bm25 empty fit search"
  assert fitCounts(@[]).searchCounts(@["cat"]).len == 0, "counts empty fit search"

  # A query made only of unknown terms returns nothing.
  let model = fitTfidf(@[@["cat"], @["dog"]])
  assert model.searchTfidf(@["unknown", "absent"]).len == 0, "tfidf unknown query"
  let countModel = fitCounts(@[@["cat"], @["dog"]])
  assert countModel.searchCounts(@["unknown"]).len == 0, "counts unknown query"

  echo "  ALL PASS"

when isMainModule:
  testEmbeddings()
  testCounts()
  testRetrievalEdges()
