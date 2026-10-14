## Feature hashing (HashingVectorizer) for the nlp library.
##
## Provides:
## - HashingVectorizer: stateless token -> sparse-vector mapping with no
##   vocabulary and no fit step
##
## Design: additive and dependency-free. Uses `std/hashes` only. Complements
## `CountModel` (bow) for unbounded vocabularies / streaming use. Pair the
## output with `cosineSimilarity` from `types` for retrieval.
##
## Stability note: hashed indices are stable within a process/run but are NOT
## guaranteed across Nim versions (std/hashes makes no cross-version promise).
## Re-hash the corpus and queries together with the same library build, and
## never persist hashed ids to disk.

import std/hashes
import types

type
  HashingVectorizer* = object
    ## Stateless feature hasher.
    ## `dim` is the number of hash buckets; `signed` enables the +/-1 trick
    ## (Weinberger et al.) which halves collision bias: each token adds +1.0
    ## or -1.0 to its bucket depending on a second independent hash.
    dim*: int
    signed*: bool

proc newHashingVectorizer*(dim = 65536, signed = true): HashingVectorizer =
  ## Create a hashing vectorizer. Raises ValueError if dim <= 0.
  if dim <= 0:
    raise newException(ValueError, "HashingVectorizer dim must be > 0, got " & $dim)
  HashingVectorizer(dim: dim, signed: signed)

proc hashBucket*(tok: string, dim: int): int =
  ## Map a token to a bucket in 0..<dim; raises ValueError if dim <= 0.
  ## Masks the sign bit first so `low(int)` cannot overflow (no abs() call).
  if dim <= 0:
    raise newException(ValueError, "hashBucket dim must be > 0, got " & $dim)
  int((uint64(hash(tok)) and 0x7FFFFFFFFFFFFFFF'u64) mod dim.uint64)

proc hashSign*(tok: string): float =
  ## Second independent hash decides the sign: odd -> -1.0, else +1.0.
  ## Exposed for future MinHash/LSH use alongside `hashBucket`.
  if (uint64(hash(tok & "\x00")) and 1'u64) == 1'u64: -1.0
  else: 1.0

proc transform*(v: HashingVectorizer, tokens: seq[string]): SparseVector =
  ## Hash pre-tokenized text into a sparse vector.
  ## Returns an empty vector for empty input. Counts accumulate per bucket,
  ## so hash collisions sum (signed collisions may cancel to zero, in which
  ## case the bucket is dropped via `set`).
  result = newSparseVector()
  if tokens.len == 0: return
  for tok in tokens:
    let bucket = hashBucket(tok, v.dim)
    let delta = if v.signed: hashSign(tok) else: 1.0
    result.set(bucket, result.get(bucket) + delta)

proc transformHashing*(
  tokens: seq[string],
  dim = 65536,
  signed = true
): SparseVector =
  ## Convenience wrapper: `newHashingVectorizer(dim, signed).transform(tokens)`.
  newHashingVectorizer(dim, signed).transform(tokens)
