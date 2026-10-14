## HashingVectorizer tests for the nlp library.

import ../src/nlp/hashing
import ../src/nlp/types
import std/tables

proc testConstructor() =
  echo "\n=== HashingVectorizer constructor ==="

  let v = newHashingVectorizer()
  assert v.dim == 65536, "default dim"
  assert v.signed == true, "default signed"

  let custom = newHashingVectorizer(dim = 1024, signed = false)
  assert custom.dim == 1024, "custom dim"
  assert custom.signed == false, "custom signed"

  doAssertRaises(ValueError):
    discard newHashingVectorizer(dim = 0)
  doAssertRaises(ValueError):
    discard newHashingVectorizer(dim = -1)

  echo "  ALL PASS"

proc testTransformBasics() =
  echo "\n=== HashingVectorizer transform basics ==="

  let v = newHashingVectorizer()

  # Empty input -> empty vector (mirrors transformCounts/transformTfidf).
  let empty = v.transform(@[])
  assert empty.len == 0, "empty input -> empty vector"
  assert transformHashing(@[]).len == 0, "convenience empty -> empty vector"

  # Determinism: same tokens twice -> identical vectors.
  let toks = @["the", "cat", "sat", "on", "the", "mat"]
  let a = v.transform(toks)
  let b = v.transform(toks)
  assert a.len == b.len, "determinism same length"
  for idx, val in a.data:
    assert b.data.hasKey(idx), "determinism same buckets: " & $idx
    assert b.data[idx] == val, "determinism same values: " & $idx

  # Convenience wrapper matches explicit construction.
  let c = transformHashing(toks)
  assert c.len == a.len, "convenience matches explicit length"
  for idx, val in a.data:
    assert c.data.getOrDefault(idx, 0.0) == val, "convenience matches explicit: " & $idx

  # Unsigned determinism as well.
  let u1 = transformHashing(toks, signed = false)
  let u2 = transformHashing(toks, signed = false)
  assert u1.len == u2.len, "unsigned determinism length"
  for idx, val in u1.data:
    assert u2.data.getOrDefault(idx, 0.0) == val, "unsigned determinism value: " & $idx

  echo "  ALL PASS"

proc testBounds() =
  echo "\n=== HashingVectorizer bounds ==="

  let dim = 64
  let v = newHashingVectorizer(dim = dim)
  let corpus = @[
    @["the", "quick", "brown", "fox"],
    @["lazy", "dog", "jumps", "over"],
    @["hello", "world", "nim", "café", "naïve"],
    @["😀", "emoji", "tokens", "123", "!@#"]
  ]
  for doc in corpus:
    let vec = v.transform(doc)
    assert vec.len > 0, "non-empty doc -> non-empty vector"
    assert vec.len <= doc.len, "buckets bounded by token count"
    for idx, _ in vec.data:
      assert idx >= 0 and idx < dim, "bucket in range: " & $idx

  # Unsigned over the same corpus: every bucket in range too.
  let u = newHashingVectorizer(dim = dim, signed = false)
  for doc in corpus:
    let vec = u.transform(doc)
    for idx, _ in vec.data:
      assert idx >= 0 and idx < dim, "unsigned bucket in range: " & $idx

  echo "  ALL PASS"

proc testCollisions() =
  echo "\n=== HashingVectorizer collisions (dim=1) ==="

  # dim=1 forces every token into bucket 0.
  let signed = newHashingVectorizer(dim = 1, signed = true)
  let toks = @["a", "b", "c", "d"]
  let svec = signed.transform(toks)
  assert svec.len <= 1, "dim=1 signed at most one bucket"
  for idx, _ in svec.data:
    assert idx == 0, "dim=1 signed bucket is 0"
  # Signed mass magnitude cannot exceed token count (cancellation only shrinks).
  assert abs(svec.get(0)) <= toks.len.float, "dim=1 signed mass bounded"

  # Unsigned dim=1: mass equals token count exactly (pure sum).
  let unsigned = newHashingVectorizer(dim = 1, signed = false)
  let uvec = unsigned.transform(toks)
  assert uvec.len == 1, "dim=1 unsigned exactly one bucket"
  assert uvec.get(0) == toks.len.float, "dim=1 unsigned mass == token count"

  # Repeated tokens accumulate in one bucket with unsigned hashing.
  let rep = unsigned.transform(@["cat", "cat", "cat"])
  assert rep.get(0) == 3.0 or rep.len <= 3, "repeated tokens accumulate"
  # With dim=1 there is exactly one bucket, so it must hold the full count.
  assert rep.len == 1 and rep.get(0) == 3.0, "dim=1 repeated tokens sum to 3"

  echo "  ALL PASS"

proc testSimilarity() =
  echo "\n=== HashingVectorizer similarity ==="

  let v = newHashingVectorizer()

  # Identical docs -> cosine 1.0 (self-similarity invariant).
  let toks = @["the", "cat", "sat", "on", "the", "mat"]
  let a = v.transform(toks)
  let same = v.transform(toks)
  assert abs(cosineSimilarity(a, same) - 1.0) < 1e-9, "identical docs cosine ~= 1.0"

  # Empty vectors have zero similarity (matches types.cosineSimilarity).
  let empty = v.transform(@[])
  assert cosineSimilarity(empty, a) == 0.0, "empty vs non-empty cosine == 0.0"
  assert cosineSimilarity(empty, empty) == 0.0, "empty vs empty cosine == 0.0"

  # Near-duplicate docs score higher than unrelated docs (large dim, no
  # forced collisions — probabilistic but overwhelmingly safe with 65536
  # buckets and short docs; pinned on ordering, not exact values).
  let query = v.transform(@["cat", "sat", "mat"])
  let related = v.transform(@["cat", "sat", "mat", "dog"])
  let unrelated = v.transform(@["quantum", "xylophone", "zephyr", "quokka"])
  assert cosineSimilarity(query, related) > cosineSimilarity(query, unrelated),
    "related outranks unrelated"

  echo "  ALL PASS"

proc testHashPrimitives() =
  echo "\n=== hashBucket/hashSign ==="

  # Determinism: same token twice -> same bucket.
  assert hashBucket("hello", 64) == hashBucket("hello", 64), "hashBucket deterministic"
  # Bounds over tokens incl. unicode/emoji.
  for tok in @["hello", "world", "caf\xC3\xA9", "na\xC3\xAFve", "\xF0\x9F\x98\x80", "", "a"]:
    let b = hashBucket(tok, 64)
    assert b >= 0 and b < 64, "hashBucket in range: " & tok
  # dim=1 forces bucket 0.
  assert hashBucket("anything", 1) == 0, "hashBucket dim=1 -> 0"
  # ValueError on non-positive dim.
  doAssertRaises(ValueError):
    discard hashBucket("hello", 0)
  doAssertRaises(ValueError):
    discard hashBucket("hello", -3)
  # hashSign returns exactly +/-1.0 and is deterministic.
  for tok in @["hello", "world", "cat", "dog"]:
    let s = hashSign(tok)
    assert s == 1.0 or s == -1.0, "hashSign +/-1.0: " & tok
    assert hashSign(tok) == s, "hashSign deterministic: " & tok
  # Both signs occur across a small corpus (unsigned/signed paths both live).
  var seenPos = false
  var seenNeg = false
  for tok in @["a", "b", "c", "d", "e", "f", "g", "h"]:
    if hashSign(tok) == 1.0: seenPos = true
    else: seenNeg = true
  assert seenPos and seenNeg, "hashSign takes both signs"

  echo "  ALL PASS"

when isMainModule:
  testConstructor()
  testHashPrimitives()
  testTransformBasics()
  testBounds()
  testCollisions()
  testSimilarity()
