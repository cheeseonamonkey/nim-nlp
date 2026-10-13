## Core type tests for the nlp library.

import ../src/nlp
import std/[tables, sets]

proc testTypes() =
  echo "\n=== Types ==="

  # SparseVector
  var v = newSparseVector()
  v.set(0, 1.5)
  v.set(1, 2.5)
  assert v.get(0) == 1.5, "sparse set/get"
  assert v.get(1) == 2.5, "sparse set/get 2"
  assert v.get(999) == 0.0, "sparse get missing"
  v.set(-1, 4.0)
  v.set(2, 0.0 / 0.0)
  assert v.len == 2, "sparse rejects invalid values"
  assert v.get(-1) == 0.0, "sparse rejects invalid index"
  assert v.get(2) == 0.0, "sparse rejects NaN"
  assert v.len == 2, "sparse len"

  # SparseVector - clear
  v.clear()
  assert v.len == 0, "sparse clear"

  # SparseVector - containsIdx
  v.set(5, 1.0)
  assert v.containsIdx(5), "sparse containsIdx true"
  assert not v.containsIdx(3), "sparse containsIdx false"

  # SparseVector - cosine with empty
  let a = newSparseVector()
  let b = newSparseVector()
  assert cosineSimilarity(a, b) == 0.0, "cosine empty vectors"

  # SparseVector - cosine one empty
  var c = newSparseVector()
  c.set(0, 1.0)
  assert cosineSimilarity(c, b) == 0.0, "cosine one empty"

  # Vocabulary
  var vocab = newVocabulary()
  let id1 = vocab.add("hello")
  let id2 = vocab.add("world")
  assert id2 != id1, "vocab different ids"
  let id3 = vocab.add("hello")
  assert id1 == id3, "vocab dedup"
  assert vocab.len == 2, "vocab len"
  assert vocab.id("hello") == id1, "vocab lookup"
  assert vocab[id1] == "hello", "vocab reverse lookup"

  # Vocabulary - bounds checking
  assert vocab[-1] == "", "vocab out of bounds negative"
  assert vocab[999] == "", "vocab out of bounds positive"

  # Vocabulary - contains
  assert vocab.contains("hello"), "vocab contains true"
  assert not vocab.contains("nonexistent"), "vocab contains false"

  # Vocabulary - containsIdx
  assert vocab.containsIdx(0), "vocab containsIdx true"
  assert vocab.containsIdx(1), "vocab containsIdx true 2"
  assert not vocab.containsIdx(-1), "vocab containsIdx negative"
  assert not vocab.containsIdx(999), "vocab containsIdx positive"

  # Vocabulary - clear
  vocab.clear()
  assert vocab.len == 0, "vocab clear"

  echo "  ALL PASS"

when isMainModule:
  testTypes()
