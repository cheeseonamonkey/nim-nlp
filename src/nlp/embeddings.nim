## Text embedding and retrieval models for the nlp library.
##
## Provides:
## - TfidfModel: TF-IDF vectorization with cosine similarity search
## - BM25Model: BM25 scoring with configurable k1, b parameters

import std/[tables, math, algorithm]
import types
import analysis

type
  TfidfModel* = object
    ## A fitted TF-IDF model for document vectorization and search.
    vocabulary*: Vocabulary
    numDocs*: int
    docVectors*: seq[SparseVector]
    idfValues*: Table[int, float]
    tokenizedDocs*: seq[seq[string]]
    documentTermFrequencies*: seq[Table[string, int]]
    documentFrequencies*: Table[string, int]

  BM25Model* = object
    ## A fitted BM25 model for document scoring.
    vocabulary*: Vocabulary
    numDocs*: int
    tokenizedDocs*: seq[seq[string]]
    documentTermFrequencies*: seq[Table[string, int]]
    documentFrequencies*: Table[string, int]
    avgDocLength*: float

# ============================================================
# Internal helpers
# ============================================================

proc maxTf(freqs: Table[string, int]): int =
  ## Get the maximum term frequency. Returns 0 for empty table.
  for _, tf in freqs:
    if tf > result: result = tf

proc buildVocab(docs: seq[seq[string]]): Vocabulary =
  ## Build vocabulary from pre-tokenized documents.
  result = newVocabulary()
  for doc in docs:
    for term in doc:
      discard result.add(term)

proc computeIdf(df: int, numDocs: int): float =
  ## Smoothed IDF: log(1 + N / df)
  if df == 0: 0.0 else: ln(1.0 + numDocs.float / df.float)

proc computeProbIdf(df: int, numDocs: int): float =
  ## Probabilistic IDF: log((N - df + 0.5) / (df + 0.5))
  let n = numDocs.float
  let d = df.float
  if d >= n: 0.0 else: ln((n - d + 0.5) / (d + 0.5))

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

# ============================================================
# TF-IDF
# ============================================================

proc fitTfidf*(
  documents: seq[seq[string]],
): TfidfModel =
  ## Fit a TF-IDF model on pre-tokenized documents.
  ## Returns an empty model for empty document list.
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

  # Compute IDF
  for termId in 0..<vocab.len:
    let term = vocab[termId]
    let docFreq = df.getOrDefault(term, 0)
    result.idfValues[termId] = computeIdf(docFreq, documents.len)

  # Compute document vectors
  result.docVectors = newSeq[SparseVector](documents.len)
  for i, freqs in termFreqs:
    let mtf = maxTf(freqs)
    var vec = newSparseVector()

    for term, tf in freqs:
      if vocab.contains(term):
        let termId = vocab.termToId[term]
        # Augmented TF + IDF
        let tfWeight = 0.5 + 0.5 * (tf.float / mtf.float)
        vec.set(termId, tfWeight * result.idfValues[termId])

    result.docVectors[i] = vec

proc transformTfidf*(
  model: TfidfModel,
  tokens: seq[string]
): SparseVector =
  ## Transform pre-tokenized text into a TF-IDF vector.
  ## Returns an empty vector for empty input.
  result = newSparseVector()
  if tokens.len == 0: return
  let freqs = termFrequencies(tokens)
  let mtf = maxTf(freqs)

  for term, tf in freqs:
    if model.vocabulary.contains(term):
      let termId = model.vocabulary.termToId[term]
      let tfWeight = 0.5 + 0.5 * (tf.float / mtf.float)
      result.set(termId, tfWeight * model.idfValues[termId])

proc searchTfidf*(
  model: TfidfModel,
  query: seq[string],
  topK: int = 5
): seq[(int, float)] =
  ## Search the corpus using TF-IDF cosine similarity.
  ## Returns empty results for empty query or empty model.
  if topK <= 0 or model.numDocs == 0 or query.len == 0: return
  let queryVec = model.transformTfidf(query)
  if queryVec.len == 0: return
  var scores = newSeq[(int, float)]()
  for i, docVec in model.docVectors:
    let score = cosineSimilarity(queryVec, docVec)
    if score > 0: scores.add((i, score))
  result = topScores(scores, topK)

# ============================================================
# BM25
# ============================================================

proc fitBm25*(
  documents: seq[seq[string]],
): BM25Model =
  ## Fit a BM25 model on pre-tokenized documents.
  ## Returns an empty model for empty document list.
  if documents.len == 0:
    result.vocabulary = newVocabulary()
    result.numDocs = 0
    result.avgDocLength = 0.0
    return
  let vocab = buildVocab(documents)
  var totalLen = 0
  for doc in documents:
    totalLen += doc.len

  result.vocabulary = vocab
  result.numDocs = documents.len
  result.tokenizedDocs = documents
  result.documentTermFrequencies = buildTermFrequencyTables(documents)
  result.documentFrequencies = documentFrequency(documents)
  if documents.len > 0:
    result.avgDocLength = totalLen.float / documents.len.float

proc searchBm25*(
  model: BM25Model,
  query: seq[string],
  topK: int = 5,
  k1: float = 1.2,
  b: float = 0.75
): seq[(int, float)] =
  ## Search the corpus using BM25 scoring.
  ## Returns empty results for empty query, empty model, or zero avgDocLength.
  if topK <= 0 or model.numDocs == 0 or query.len == 0: return
  if model.avgDocLength == 0.0: return
  if k1 < 0.0 or b < 0.0 or b > 1.0:
    raise newException(ValueError, "BM25 requires k1 >= 0 and 0 <= b <= 1")
  var queryFreqs = initTable[string, int]()
  for term in query:
    queryFreqs[term] = queryFreqs.getOrDefault(term, 0) + 1

  var scores = newSeq[(int, float)]()

  for docId in 0..<model.numDocs:
    var score = 0.0
    let docLen = model.tokenizedDocs[docId].len.float

    for term, qf in queryFreqs:
      if not model.vocabulary.contains(term): continue
      let docFreq = model.documentFrequencies.getOrDefault(term, 0)
      let idf = computeProbIdf(docFreq, model.numDocs)
      let tf = model.documentTermFrequencies[docId].getOrDefault(term, 0)

      if tf > 0:
        let lengthNorm = 1.0 - b + b * (docLen / model.avgDocLength)
        let tfComponent = (tf.float * (k1 + 1.0)) / (tf.float + k1 * lengthNorm)
        let queryTfComponent = (qf.float * (k1 + 1.0)) / (qf.float + k1)
        score += idf * tfComponent * queryTfComponent

    if score > 0: scores.add((docId, score))

  result = topScores(scores, topK)
