## Shared model plumbing for the nlp library.
##
## Vocabulary construction, per-document term-frequency tables, and score
## ranking are needed by several modules; they live here so the logic is
## defined once.

import std/[tables, algorithm]
import ../types

proc topRanked*[K, V](
  pairs: seq[(K, V)], topN: int, keyCmp: proc (a, b: K): int
): seq[(K, V)] =
  ## Generic ranker: sort by value descending, tie-break via `keyCmp`,
  ## truncate to `topN`. (Same truncation semantics as `topScored`.)
  ## Note: `K` is intentionally unconstrained (`[K, V: SomeNumber]` would
  ## wrongly require `K` itself to be numeric) — only the values are
  ## ordered via `cmp`, so `V` just needs a `cmp` overload.
  result = pairs
  result.sort(proc(a, b: (K, V)): int =
    if a[1] == b[1]: keyCmp(a[0], b[0])
    else: cmp(b[1], a[1]))
  if result.len > topN:
    result = result[0..<topN]

proc topScored*[T: SomeNumber](
  pairs: seq[(string, T)], topN: int
): seq[(string, T)] =
  ## Sort `(term, score)` pairs by score descending (tie-break: term
  ## ascending) and truncate to `topN`.
  topRanked(pairs, topN, proc(a, b: string): int = cmp(a, b))

proc buildVocabulary*(docs: seq[seq[string]]): Vocabulary =
  ## Build a vocabulary from pre-tokenized documents.
  result = newVocabulary()
  for doc in docs:
    for term in doc:
      discard result.add(term)

proc termFrequencyTables*(
  docs: seq[seq[string]]
): seq[Table[string, int]] =
  ## Build one term-frequency table per document.
  result = newSeq[Table[string, int]](docs.len)
  for i, doc in docs:
    for term in doc:
      result[i][term] = result[i].getOrDefault(term, 0) + 1

proc rankTopK*(
  scores: seq[(int, float)], topK: int
): seq[(int, float)] =
  ## Sort by score descending (tie-break: docId ascending) and truncate.
  result = scores
  result.sort(proc(a, b: (int, float)): int =
    if a[1] == b[1]: cmp(a[0], b[0])
    else: cmp(b[1], a[1]))
  if result.len > topK:
    result = result[0..<topK]

