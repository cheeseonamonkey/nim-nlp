## Distance tests for the nlp library.

import ../src/nlp/distance
import std/[math, unicode]

proc approx(a, b: float, tol = 1e-6): bool =
  abs(a - b) <= tol

proc testLevenshtein() =
  echo "=== levenshtein ==="
  assert levenshtein("kitten", "sitting") == 3, "kitten/sitting == 3"
  assert levenshtein("", "") == 0, "both empty == 0"
  assert levenshtein("", "abc") == 3, "empty vs abc"
  assert levenshtein("abc", "") == 3, "abc vs empty"
  assert levenshtein("abc", "abc") == 0, "identical == 0"
  assert levenshtein("a", "b") == 1, "single substitution"
  assert levenshtein("a", "") == 1, "single vs empty"
  assert levenshtein("", "a") == 1, "empty vs single"
  assert levenshtein("flaw", "lawn") == 2, "flaw/lawn == 2"
  assert levenshtein("gumbo", "gambol") == 2, "gumbo/gambol == 2"
  assert levenshtein("saturday", "sunday") == 3, "saturday/sunday == 3"
  # Case-sensitive
  assert levenshtein("ABC", "abc") == 3, "case-sensitive"
  # Rune-aware: é is one unit, not two bytes
  assert levenshtein("café", "cafe") == 1, "unicode café/cafe == 1"
  assert levenshtein("😀", "😀") == 0, "emoji identical"
  assert levenshtein("😀", "a") == 1, "emoji vs ascii == 1"
  assert levenshtein("é", "") == 1, "single rune vs empty"
  # Symmetry
  assert levenshtein("kitten", "sitting") == levenshtein("sitting", "kitten"),
    "levenshtein symmetry"
  assert levenshtein("café", "cafe") == levenshtein("cafe", "café"),
    "levenshtein unicode symmetry"

proc testLevenshteinSimilarity() =
  echo "=== levenshteinSimilarity ==="
  assert levenshteinSimilarity("", "") == 1.0, "both empty -> 1.0"
  assert levenshteinSimilarity("", "abc") == 0.0, "one empty -> 0.0"
  assert levenshteinSimilarity("abc", "") == 0.0, "one empty -> 0.0 other side"
  assert levenshteinSimilarity("abc", "abc") == 1.0, "identical -> 1.0"
  assert approx(levenshteinSimilarity("kitten", "sitting"), 1.0 - 3.0 / 7.0),
    "kitten/sitting similarity"
  assert approx(levenshteinSimilarity("café", "cafe"), 0.75),
    "unicode similarity 1 - 1/4"
  # Range check
  for pair in [("hello", "hallo"), ("nim", "nimble"), ("", "x"), ("a", "b")]:
    let s = levenshteinSimilarity(pair[0], pair[1])
    assert s >= 0.0 and s <= 1.0, "levSim range [0,1]"
  # Symmetry
  assert approx(levenshteinSimilarity("abc", "abd"), levenshteinSimilarity("abd", "abc")),
    "levSim symmetry"

proc testJaro() =
  echo "=== jaro ==="
  assert jaro("", "") == 1.0, "both empty -> 1.0"
  assert jaro("", "abc") == 0.0, "one empty -> 0.0"
  assert jaro("abc", "") == 0.0, "one empty -> 0.0 other side"
  assert jaro("abc", "abc") == 1.0, "identical -> 1.0"
  assert jaro("a", "a") == 1.0, "single identical -> 1.0"
  assert jaro("a", "b") == 0.0, "single different -> 0.0"
  assert approx(jaro("martha", "marhta"), 0.9444444444444445),
    "jaro martha/marhta"
  assert approx(jaro("dixon", "dicksonx"), 0.7666666666666666),
    "jaro dixon/dicksonx"
  # Symmetry
  assert approx(jaro("martha", "marhta"), jaro("marhta", "martha")),
    "jaro symmetry"
  assert approx(jaro("dixon", "dicksonx"), jaro("dicksonx", "dixon")),
    "jaro symmetry 2"
  # Range check
  for pair in [("hello", "hallo"), ("nim", "nimble"), ("a", "b"), ("abc", "xyz")]:
    let s = jaro(pair[0], pair[1])
    assert s >= 0.0 and s <= 1.0, "jaro range [0,1]"
  # Unicode identical -> 1.0
  assert jaro("café", "café") == 1.0, "jaro unicode identical"
  assert jaro("😀😀", "😀😀") == 1.0, "jaro emoji identical"

proc testJaroWinkler() =
  echo "=== jaroWinkler ==="
  assert approx(jaroWinkler("martha", "marhta"), 0.9611111111111111),
    "jw martha/marhta"
  assert jaroWinkler("abc", "abc") == 1.0, "jw identical -> 1.0"
  assert jaroWinkler("", "") == 1.0, "jw both empty -> 1.0"
  assert jaroWinkler("", "abc") == 0.0, "jw one empty -> 0.0"
  assert jaroWinkler("abc", "") == 0.0, "jw one empty -> 0.0 other side"
  # Common prefix boosts: "dixon"/"dicksonx" has prefix "di" (l=2)
  let j = jaro("dixon", "dicksonx")
  let expected = j + 2.0 * 0.1 * (1.0 - j)
  assert approx(jaroWinkler("dixon", "dicksonx"), expected),
    "jw dixon/dicksonx prefix boost"
  # No common prefix -> equals jaro
  assert approx(jaroWinkler("abc", "xyz"), jaro("abc", "xyz")),
    "jw no prefix equals jaro"
  # maxPrefix = 0 disables the boost
  assert approx(jaroWinkler("martha", "marhta", maxPrefix = 0), jaro("martha", "marhta")),
    "jw maxPrefix=0 equals jaro"
  # Custom prefix weight
  let j2 = jaro("martha", "marhta")
  assert approx(jaroWinkler("martha", "marhta", prefixWeight = 0.2), j2 + 3.0 * 0.2 * (1.0 - j2)),
    "jw custom weight"
  # Range check
  for pair in [("hello", "hallo"), ("nim", "nimble"), ("a", "b"), ("café", "cafe")]:
    let s = jaroWinkler(pair[0], pair[1])
    assert s >= 0.0 and s <= 1.0, "jw range [0,1]"
  # Errors
  try:
    discard jaroWinkler("a", "b", prefixWeight = -0.1)
    assert false, "negative prefixWeight should raise"
  except ValueError:
    assert true, "negative prefixWeight raises ValueError"
  try:
    discard jaroWinkler("a", "b", maxPrefix = -1)
    assert false, "negative maxPrefix should raise"
  except ValueError:
    assert true, "negative maxPrefix raises ValueError"

proc testRuneBases() =
  echo "=== rune bases ==="
  # Equivalence with the string versions, incl. unicode.
  assert levenshteinRunes("kitten".toRunes(), "sitting".toRunes()) == levenshtein("kitten", "sitting"),
    "levenshteinRunes kitten/sitting"
  assert levenshteinRunes("café".toRunes(), "cafe".toRunes()) == 1,
    "levenshteinRunes café/cafe == 1"
  assert levenshteinRunes("".toRunes(), "".toRunes()) == 0, "levenshteinRunes both empty"
  assert levenshteinRunes("".toRunes(), "abc".toRunes()) == 3, "levenshteinRunes empty vs abc"
  assert levenshteinRunes("😀".toRunes(), "a".toRunes()) == 1, "levenshteinRunes emoji vs ascii"
  assert approx(jaroRunes("martha".toRunes(), "marhta".toRunes()), jaro("martha", "marhta")),
    "jaroRunes martha/marhta equivalence"
  assert jaroRunes("café".toRunes(), "café".toRunes()) == 1.0,
    "jaroRunes unicode identical"
  assert jaroRunes("".toRunes(), "".toRunes()) == 1.0, "jaroRunes both empty"
  assert jaroRunes("".toRunes(), "abc".toRunes()) == 0.0, "jaroRunes one empty"

proc testNgramDistance() =
  echo "=== ngramDistance ==="
  assert ngramDistance("hello", "hello") == 0.0, "identical -> 0.0"
  assert ngramDistance("", "") == 0.0, "both empty -> 0.0"
  assert ngramDistance("a", "a") == 0.0, "short identical -> 0.0"
  # Symmetry
  assert approx(ngramDistance("hello", "hallo"), ngramDistance("hallo", "hello")),
    "ngramDistance symmetry"
  assert approx(ngramDistance("night", "nacht"), ngramDistance("nacht", "night")),
    "ngramDistance symmetry 2"
  # Range check
  for pair in [("hello", "hallo"), ("nim", "nimble"), ("abc", "xyz"), ("", "x")]:
    let d = ngramDistance(pair[0], pair[1])
    assert d >= 0.0 and d <= 1.0, "ngramDistance range [0,1]"
  # Similar strings are closer than dissimilar ones
  assert ngramDistance("hello", "hallo") < ngramDistance("hello", "xyz"),
    "similar < dissimilar"
  # Completely disjoint bigram sets -> 1.0
  assert ngramDistance("ab", "cd") == 1.0, "disjoint bigrams -> 1.0"
  # Case-sensitive: upper vs lower share no bigrams -> max distance.
  assert ngramDistance("ABC", "abc") > 0.0, "ngramDistance case-sensitive ABC/abc > 0"
  assert ngramDistance("ABC", "abc") == 1.0, "ngramDistance ABC/abc disjoint -> 1.0"
  # Unicode identical -> 0.0
  assert ngramDistance("café", "café") == 0.0, "unicode identical -> 0.0"
  # Trigram variant stays in range and is symmetric
  let t1 = ngramDistance("hello", "hallo", 3)
  let t2 = ngramDistance("hallo", "hello", 3)
  assert t1 >= 0.0 and t1 <= 1.0, "trigram range"
  assert approx(t1, t2), "trigram symmetry"

proc testDistance() =
  testLevenshtein()
  testLevenshteinSimilarity()
  testJaro()
  testJaroWinkler()
  testRuneBases()
  testNgramDistance()
  echo "  ALL PASS"

when isMainModule:
  testDistance()
