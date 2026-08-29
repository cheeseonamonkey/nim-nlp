## Bag-of-Words / CountVectorizer for the nlp library.
##
## Provides:
## - CountModel: fitted term-count model with cosine-similarity search
##
## Reuses `Vocabulary` and `SparseVector` from `types` and helpers from
## `analysis`/`embeddings`. Purely count-based (no IDF); TF-IDF/BM25 remain
## in `embeddings`.
##
## Design: additive and dependency-free. No new data files, no new runtime
## dependencies. Mirrors the TF-IDF/BM25 call shape (`fit*`, `transform*`,
## `search*`, `topK`) for familiarity.

import std/[tables, algorithm]
import types
import analysis

type
  CountModel* = object
    ## A fitted bag-of-words count model.
    vocabulary*: Vocabulary
    numDocs*: int
    docVectors*: seq[SparseVector]
    tokenizedDocs*: seq[seq[string]]
    documentTermFrequencies*: seq[Table[string, int]]
    documentFrequencies*: Table[string, int]

proc buildVocab(docs: seq[seq[string]]): Vocabulary =
  result = newVocabulary()
  for doc in docs:
    for term in doc:
      discard result.add(term)

proc buildTermFrequencyTables(
  docs: seq[seq[string]]
): seq[Table[string, int]] =
  result = newSeq[Table[string, int]](docs.len)
  for i, doc in docs:
    result[i] = termFrequencies(doc)

proc topScores(
  scores: var seq[(int, float)], topK: int
): seq[(int, float)] =
  scores.sort(proc(a, b: (int, float)): int =
    if a[1] == b[1]: cmp(a[0], b[0])
    else: cmp(b[1], a[1]))
  if scores.len > topK: scores[0..<topK] else: scores

proc fitCounts*(
  documents: seq[seq[string]],
): CountModel =
  ## Fit a bag-of-words count model on pre-tokenized documents.
  ## Returns an empty model for an empty document list.
  if documents.len == 0:
    result.vocabulary = newVocabulary()
    result.numDocs = 0
    return
  let vocab = buildVocab(documents)
  let df = documentFrequency(documents)
  let termFreqs = buildTermFrequencyTables(documents)

  result.vocabulary = vocab
  result.numDocs = documents.len
  result.tokenizedDocs = documents
  result.documentTermFrequencies = termFreqs
  result.documentFrequencies = df
  result.docVectors = newSeq[SparseVector](documents.len)
  for i, freqs in termFreqs:
    var vec = newSparseVector()
    for term, tf in freqs:
      if vocab.contains(term):
        let termId = vocab.termToId[term]
        vec.set(termId, tf.float)
    result.docVectors[i] = vec

proc transformCounts*(
  model: CountModel,
  tokens: seq[string]
): SparseVector =
  ## Transform pre-tokenized text into a count vector.
  ## Returns an empty vector for empty input or unknown terms only.
  result = newSparseVector()
  if tokens.len == 0: return
  let freqs = termFrequencies(tokens)
  for term, tf in freqs:
    if model.vocabulary.contains(term):
      let termId = model.vocabulary.termToId[term]
      result.set(termId, tf.float)

proc searchCounts*(
  model: CountModel,
  query: seq[string],
  topK: int = 5
): seq[(int, float)] =
  ## Search the corpus using cosine similarity over count vectors.
  ## Returns empty results for empty query or empty model.
  if topK <= 0 or model.numDocs == 0 or query.len == 0: return
  let queryVec = model.transformCounts(query)
  if queryVec.len == 0: return
  var scores = newSeq[(int, float)]()
  for i, docVec in model.docVectors:
    let score = cosineSimilarity(queryVec, docVec)
    if score > 0: scores.add((i, score))
  result = topScores(scores, topK)
