## Shared model plumbing for the nlp library.
##
## Vocabulary construction, per-document term-frequency tables, and top-K
## ranking are needed by both `bow` and `embeddings`. They are implemented
## once here instead of being duplicated verbatim in each module.

import std/[tables, algorithm]
import ../types
import ../analysis

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
    result[i] = termFrequencies(doc)

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
