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

import std/tables
import types
import analysis
import private/model_utils

type
  CountModel* = object
    ## A fitted bag-of-words count model.
    vocabulary*: Vocabulary
    numDocs*: int
    docVectors*: seq[SparseVector]
    tokenizedDocs*: seq[seq[string]]
    documentTermFrequencies*: seq[Table[string, int]]
    documentFrequencies*: Table[string, int]

proc fitCounts*(
  documents: seq[seq[string]],
): CountModel =
  ## Fit a bag-of-words count model on pre-tokenized documents.
  ## Returns an empty model for an empty document list.
  if documents.len == 0:
    result.vocabulary = newVocabulary()
    result.numDocs = 0
    return
  let vocab = buildVocabulary(documents)
  let df = documentFrequency(documents)
  let termFreqs = termFrequencyTables(documents)

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
  result = rankTopK(scores, topK)
