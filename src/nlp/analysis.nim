## Text analysis utilities for the nlp library.
##
## Provides:
## - Term frequency analysis
## - Document frequency analysis
## - Text similarity (Jaccard, cosine on bags, overlap)
## - Text statistics
## - Keyword extraction

import std/[tables, math, algorithm, sets, sequtils, strutils]

# ============================================================
# Frequency analysis
# ============================================================

proc termFrequencies*(tokens: seq[string]): Table[string, int] =
  ## Count term frequencies in a token sequence.
  for tok in tokens:
    result[tok] = result.getOrDefault(tok, 0) + 1

proc documentFrequency*(documents: seq[seq[string]]): Table[string, int] =
  ## Count how many documents contain each term.
  for doc in documents:
    var seen = initHashSet[string]()
    for tok in doc:
      if tok notin seen:
        seen.incl(tok)
        result[tok] = result.getOrDefault(tok, 0) + 1

proc sortedTermsByFrequency*(freqs: Table[string, int]): seq[(string, int)] =
  ## Sort terms by frequency descending.
  var pairs = newSeq[(string, int)]()
  for term, freq in freqs:
    pairs.add((term, freq))
  pairs.sort(proc(a, b: (string, int)): int =
    if a[1] == b[1]: cmp(a[0], b[0])
    else: cmp(b[1], a[1]))
  result = pairs

# ============================================================
# Similarity measures
# ============================================================

proc jaccardSimilarity*(a, b: seq[string]): float =
  ## Jaccard similarity between two token sequences.
  ## Returns 0.0 for empty inputs.
  if a.len == 0 or b.len == 0: return 0.0
  let setA = a.toHashSet()
  let setB = b.toHashSet()
  let intersection = setA * setB
  let union = setA + setB
  if union.len == 0: 0.0
  else: intersection.len.float / union.len.float

proc cosineSimilarityBags*(a, b: seq[string]): float =
  ## Cosine similarity between two token sequences (as bags of words).
  ## Returns 0.0 for empty inputs.
  if a.len == 0 or b.len == 0: return 0.0
  let freqA = termFrequencies(a)
  let freqB = termFrequencies(b)
  var dotProduct = 0.0
  var normA = 0.0
  var normB = 0.0
  for term, freq in freqA:
    normA += freq.float * freq.float
    if freqB.hasKey(term):
      dotProduct += freq.float * freqB[term].float
  for _, freq in freqB:
    normB += freq.float * freq.float
  if normA == 0.0 or normB == 0.0: 0.0
  else: dotProduct / (sqrt(normA) * sqrt(normB))

proc overlapCoefficient*(a, b: seq[string]): float =
  ## Overlap coefficient: |A ∩ B| / min(|A|, |B|).
  ## Returns 0.0 for empty inputs.
  if a.len == 0 or b.len == 0: return 0.0
  let setA = a.toHashSet()
  let setB = b.toHashSet()
  let intersection = setA * setB
  let minSize = min(setA.len, setB.len)
  if minSize == 0: 0.0
  else: intersection.len.float / minSize.float

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
  case methodName.toLowerAscii():
  of "tfidf":
    let df = documentFrequency(documents)
    let n = documents.len.float
    var scores = initTable[string, float]()
    for doc in documents:
      let tf = termFrequencies(doc)
      if tf.len == 0: continue
      let mtf = max(tf.values.toSeq())
      for term, freq in tf:
        let idf = ln(1.0 + n / df.getOrDefault(term, 1).float)
        let tfWeight = 0.5 + 0.5 * (freq.float / mtf.float)
        scores[term] = scores.getOrDefault(term, 0.0) + tfWeight * idf
    var pairs = newSeq[(string, float)]()
    for term, score in scores:
      pairs.add((term, score))
    pairs.sort(proc(a, b: (string, float)): int =
      if a[1] == b[1]: cmp(a[0], b[0])
      else: cmp(b[1], a[1]))
    if pairs.len > topN:
      result = pairs[0..<topN]
    else:
      result = pairs
  of "freq":
    var combined = initTable[string, int]()
    for doc in documents:
      for term in doc:
        combined[term] = combined.getOrDefault(term, 0) + 1
    var pairs = newSeq[(string, float)]()
    for term, freq in combined:
      pairs.add((term, freq.float))
    pairs.sort(proc(a, b: (string, float)): int =
      if a[1] == b[1]: cmp(a[0], b[0])
      else: cmp(b[1], a[1]))
    if pairs.len > topN:
      result = pairs[0..<topN]
    else:
      result = pairs
  else:
    result = newSeq[(string, float)]()
