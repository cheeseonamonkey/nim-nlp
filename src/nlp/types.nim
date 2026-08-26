## Core types for the nlp library.

import std/[tables, sets, math]

export tables.Table
export sets.HashSet

type
  Term* = string

  SparseVector* = object
    ## Sparse vector: index -> value mapping.
    data*: Table[int, float]

  Vocabulary* = object
    ## Maps terms to integer IDs and back.
    termToId*: Table[Term, int]
    idToTerm*: seq[Term]

# ============================================================
# SparseVector
# ============================================================

proc newSparseVector*(): SparseVector =
  SparseVector(data: initTable[int, float]())

proc set*(v: var SparseVector, idx: int, val: float) =
  ## Set a value; zero and invalid indices remove/ignore entries.
  ## NaN is ignored so it cannot poison later similarity calculations.
  if idx < 0 or val != val:
    return
  if val == 0.0:
    v.data.del(idx)
  else:
    v.data[idx] = val

proc get*(v: SparseVector, idx: int): float =
  ## Get a value (0.0 if not present).
  v.data.getOrDefault(idx, 0.0)

proc containsIdx*(v: SparseVector, idx: int): bool =
  ## Check if a non-zero value exists at the given index.
  v.data.hasKey(idx)

proc clear*(v: var SparseVector) =
  ## Remove all entries from the vector.
  v.data.clear()

proc len*(v: SparseVector): int = v.data.len

proc pairs*(v: SparseVector): seq[(int, float)] =
  result = newSeq[(int, float)](v.data.len)
  var i = 0
  for k, val in v.data:
    result[i] = (k, val)
    inc i

proc values*(v: SparseVector): seq[float] =
  result = newSeq[float](v.data.len)
  var i = 0
  for _, val in v.data:
    result[i] = val
    inc i

proc cosineSimilarity*(a, b: SparseVector): float =
  ## Cosine similarity between two sparse vectors.
  ## Returns 0.0 if either vector is empty or norms are zero.
  var dot = 0.0
  var normA = 0.0
  var normB = 0.0
  for idx, val in a.data:
    normA += val * val
    if b.data.hasKey(idx):
      dot += val * b.data[idx]
  for _, val in b.data:
    normB += val * val
  if normA == 0.0 or normB == 0.0: return 0.0
  result = dot / (sqrt(normA) * sqrt(normB))
  # Guard against NaN from floating point imprecision
  if result != result: return 0.0

# ============================================================
# Vocabulary
# ============================================================

proc newVocabulary*(): Vocabulary =
  Vocabulary(termToId: initTable[Term, int](), idToTerm: newSeq[Term]())

proc add*(v: var Vocabulary, term: Term): int =
  ## Add a term to the vocabulary, return its ID.
  if not v.termToId.hasKey(term):
    result = v.idToTerm.len
    v.termToId[term] = result
    v.idToTerm.add(term)
  else:
    result = v.termToId[term]

proc id*(v: Vocabulary, term: Term): int =
  ## Get the ID for a term (-1 if not found).
  v.termToId.getOrDefault(term, -1)

proc contains*(v: Vocabulary, term: Term): bool = v.termToId.hasKey(term)

proc containsIdx*(v: Vocabulary, idx: int): bool =
  ## Check if an index is valid in the vocabulary.
  idx >= 0 and idx < v.idToTerm.len

proc `[]`*(v: Vocabulary, idx: int): Term =
  ## Get the term at an index (\"\" if out of bounds).
  if idx >= 0 and idx < v.idToTerm.len:
    result = v.idToTerm[idx]
  else:
    result = ""

proc clear*(v: var Vocabulary) =
  ## Remove all terms from the vocabulary.
  v.termToId.clear()
  v.idToTerm.setLen(0)

proc len*(v: Vocabulary): int = v.idToTerm.len
