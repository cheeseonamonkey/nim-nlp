## Text analysis utilities for the nlp library.
##
## Provides:
## - Term frequency analysis
## - Document frequency analysis
## - Text similarity (Jaccard, cosine on bags, overlap)
## - Text statistics
## - Keyword extraction

import std/[tables, math, algorithm, sets, sequtils, strutils, unicode]
import private/model_utils
import ngrams

# ============================================================
# Generic base primitives
# ============================================================

proc countItems*[K](items: openArray[K]): Table[K, int] =
  ## Generic counter: count occurrences of each item.
  ## Empty input -> empty table.
  for x in items:
    result[x] = result.getOrDefault(x, 0) + 1

proc cosineCounts*[K](fa, fb: Table[K, int]): float =
  ## Cosine similarity over integer count tables.
  ## Returns 0.0 if either table is empty (zero norm).
  ## (Float-vector sibling: `cosineSimilarity` on SparseVector in `types`.)
  var dotProduct = 0.0
  var normA = 0.0
  var normB = 0.0
  for key, freq in fa:
    normA += freq.float * freq.float
    if fb.hasKey(key):
      dotProduct += freq.float * fb[key].float
  for _, freq in fb:
    normB += freq.float * freq.float
  if normA == 0.0 or normB == 0.0: 0.0
  else: dotProduct / (sqrt(normA) * sqrt(normB))

proc jaccardSets*[T](a, b: HashSet[T]): float =
  ## Jaccard similarity over sets: |A n B| / |A u B|.
  ## Returns 0.0 if the union is empty.
  let intersection = a * b
  let union = a + b
  if union.len == 0: 0.0
  else: intersection.len.float / union.len.float

proc overlapSets*[T](a, b: HashSet[T]): float =
  ## Overlap coefficient over sets: |A n B| / min(|A|, |B|).
  ## Returns 0.0 if the smaller set is empty.
  let minSize = min(a.len, b.len)
  if minSize == 0: 0.0
  else: (a * b).len.float / minSize.float

# ============================================================
# Frequency analysis
# ============================================================

proc termFrequencies*(tokens: seq[string]): Table[string, int] =
  ## Count term frequencies in a token sequence.
  countItems(tokens)

proc documentFrequency*(documents: seq[seq[string]]): Table[string, int] =
  ## Count how many documents contain each term.
  for doc in documents:
    var seen = initHashSet[string]()
    for tok in doc:
      if tok notin seen:
        seen.incl(tok)
        result[tok] = result.getOrDefault(tok, 0) + 1

proc idfSmoothed*(df: int, numDocs: int): float =
  ## Smoothed inverse document frequency: ln(1 + N / df).
  ## Returns 0.0 when the term appears in no document.
  if df == 0: 0.0 else: ln(1.0 + numDocs.float / df.float)

proc sortedTermsByFrequency*(freqs: Table[string, int]): seq[(string, int)] =
  ## Sort terms by frequency descending (tie-break: term ascending).
  var pairs = newSeq[(string, int)]()
  for term, freq in freqs:
    pairs.add((term, freq))
  topScored(pairs, pairs.len)

# ============================================================
# Similarity measures
# ============================================================

proc jaccardSimilarity*(a, b: seq[string]): float =
  ## Jaccard similarity between two token sequences.
  ## Returns 0.0 for empty inputs.
  jaccardSets(a.toHashSet(), b.toHashSet())

proc cosineSimilarityBags*(a, b: seq[string]): float =
  ## Cosine similarity between two token sequences (as bags of words).
  ## Returns 0.0 for empty inputs.
  cosineCounts(termFrequencies(a), termFrequencies(b))

proc overlapCoefficient*(a, b: seq[string]): float =
  ## Overlap coefficient: |A ∩ B| / min(|A|, |B|).
  ## Returns 0.0 for empty inputs.
  overlapSets(a.toHashSet(), b.toHashSet())

# ============================================================
# Text statistics
# ============================================================

type
  TextStats* = object
    ## Basic statistics about a text or corpus.
    numTokens*: int
    numUniqueTerms*: int
    avgTokenLength*: float
    vocabularyRichness*: float  # unique terms / total tokens
    hapaxLegomena*: int         # terms appearing only once
    topTerm*: string            # most frequent term

proc textStats*(tokens: seq[string]): TextStats =
  ## Compute basic statistics for a token sequence.
  ## Returns zeros for empty input.
  result.numTokens = tokens.len
  let freqs = termFrequencies(tokens)
  result.numUniqueTerms = freqs.len
  var totalLen = 0
  for tok in tokens:
    totalLen += tok.len
  if tokens.len > 0:
    result.avgTokenLength = totalLen.float / tokens.len.float
    result.vocabularyRichness = freqs.len.float / tokens.len.float
  # Hapax legomena (terms appearing only once)
  var maxFreq = 0
  for term, freq in freqs:
    if freq == 1: result.hapaxLegomena.inc
    if freq > maxFreq or (freq == maxFreq and
        (result.topTerm.len == 0 or term < result.topTerm)):
      maxFreq = freq
      result.topTerm = term

proc uniqueTerms*(documents: seq[seq[string]]): HashSet[string] =
  ## Get the set of all unique terms across multiple documents.
  for doc in documents:
    for tok in doc:
      result.incl(tok)

# ============================================================
# Keyword extraction
# ============================================================

proc extractKeywords*(
  documents: seq[seq[string]],
  topN: int = 10,
  methodName: string = "tfidf"
): seq[(string, float)] =
  ## Extract keywords from a corpus.
  ##
  ## Methods:
  ## - `"tfidf"`: TF-IDF based extraction
  ## - `"freq"`: Simple frequency-based extraction
  if documents.len == 0 or topN <= 0:
    return newSeq[(string, float)]()
  var scores: Table[string, float]
  case methodName.toLower():
  of "tfidf":
    let df = documentFrequency(documents)
    scores = initTable[string, float]()
    for doc in documents:
      let tf = termFrequencies(doc)
      if tf.len == 0: continue
      let mtf = max(tf.values.toSeq())
      for term, freq in tf:
        let idf = idfSmoothed(df.getOrDefault(term, 1), documents.len)
        let tfWeight = 0.5 + 0.5 * (freq.float / mtf.float)
        scores[term] = scores.getOrDefault(term, 0.0) + tfWeight * idf
  of "freq":
    scores = initTable[string, float]()
    for doc in documents:
      for term in doc:
        scores[term] = scores.getOrDefault(term, 0.0) + 1.0
  else:
    return newSeq[(string, float)]()
  var pairs = newSeq[(string, float)]()
  for term, score in scores:
    pairs.add((term, score))
  result = topScored(pairs, topN)

# ============================================================
# N-gram analysis
# ============================================================

proc ngramFrequencies*(tokens: seq[string], n: int): Table[seq[string], int] =
  ## Count n-gram frequencies in a token sequence.
  ## Keys are token arrays, preserving token boundaries.
  ## Returns an empty table when n <= 0 or `tokens.len < n`.
  countItems(ngrams(tokens, n))

proc topNgrams*(
  tokens: seq[string], n: int, topK = 10
): seq[(seq[string], int)] =
  ## Top-K most frequent n-grams, frequency descending.
  ## Ties break by the joined form ascending ("\x1F" separator,
  ## comparison only) for determinism.
  ## Returns @[] when topK <= 0 or there are no n-grams.
  if topK <= 0: return @[]
  let freqs = ngramFrequencies(tokens, n)
  var pairs = newSeq[(seq[string], int)]()
  for gram, freq in freqs:
    pairs.add((gram, freq))
  result = topRanked(pairs, topK,
    proc(a, b: seq[string]): int = cmp(a.join("\x1F"), b.join("\x1F")))

proc ngramCosineSimilarity*(a, b: seq[string], n: int): float =
  ## Cosine similarity between two token sequences as bags of n-grams.
  ## Order-sensitive: unlike `cosineSimilarityBags`, shingles capture
  ## word order. Returns 0.0 for empty inputs or when either side
  ## yields no n-grams.
  cosineCounts(ngramFrequencies(a, n), ngramFrequencies(b, n))

proc ngramJaccardSimilarity*(a, b: seq[string], n: int): float =
  ## Jaccard similarity between the n-gram sets of two token sequences.
  ## Returns 0.0 for empty inputs or when either side yields no n-grams.
  jaccardSets(ngrams(a, n).toHashSet(), ngrams(b, n).toHashSet())
