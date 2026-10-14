## String distance and similarity measures for the nlp library.
##
## Provides:
## - Levenshtein edit distance and similarity
## - Jaro and Jaro-Winkler similarity
## - Character n-gram distance (via tokenize + analysis)
##
## All procedures are rune-aware (Unicode-correct): inputs are decoded
## with `toRunes` before comparison, so e.g. "é" counts as one unit.

import std/[unicode, math]
import tokenize
import analysis

# ============================================================
# Levenshtein
# ============================================================

proc levenshteinRunes*(ra, rb: seq[Rune]): int =
  ## Two-row DP Levenshtein core on decoded runes (case-sensitive).
  ## Empty-input guards live here so both entry points share them.
  let m = ra.len
  let n = rb.len
  if m == 0: return n
  if n == 0: return m
  var prev = newSeq[int](n + 1)
  var curr = newSeq[int](n + 1)
  for j in 0 .. n:
    prev[j] = j
  for i in 1 .. m:
    curr[0] = i
    for j in 1 .. n:
      let cost = if ra[i - 1] == rb[j - 1]: 0 else: 1
      curr[j] = min(min(prev[j] + 1, curr[j - 1] + 1), prev[j - 1] + cost)
    swap(prev, curr)
  result = prev[n]

proc levenshtein*(a, b: string): int =
  ## Classic Levenshtein edit distance on runes (case-sensitive).
  ## Uses O(min(lenA, lenB)) memory via a two-row DP.
  ## Decodes once and delegates to `levenshteinRunes <#levenshteinRunes>`_.
  levenshteinRunes(a.toRunes(), b.toRunes())

proc levenshteinSimilarity*(a, b: string): float =
  ## Normalized Levenshtein similarity: `1 - dist / max(lenA, lenB)`.
  ## Both empty -> 1.0; exactly one empty -> 0.0.
  let ra = a.toRunes()
  let rb = b.toRunes()
  if ra.len == 0 and rb.len == 0: return 1.0
  if ra.len == 0 or rb.len == 0: return 0.0
  let dist = levenshteinRunes(ra, rb)
  result = 1.0 - dist.float / max(ra.len, rb.len).float

# ============================================================
# Jaro / Jaro-Winkler
# ============================================================

proc jaroRunes*(ra, rb: seq[Rune]): float =
  ## Jaro match/transposition core on decoded runes.
  ## Empty-input guards live here so both entry points share them.
  ## Both empty -> 1.0; one empty -> 0.0; no matches -> 0.0.
  let lenA = ra.len
  let lenB = rb.len
  if lenA == 0 and lenB == 0: return 1.0
  if lenA == 0 or lenB == 0: return 0.0
  var window = max(lenA, lenB) div 2 - 1
  if window < 0: window = 0
  var matchedA = newSeq[bool](lenA)
  var matchedB = newSeq[bool](lenB)
  var matches = 0
  for i in 0 ..< lenA:
    let lo = max(0, i - window)
    let hi = min(lenB - 1, i + window)
    for j in lo .. hi:
      if not matchedB[j] and ra[i] == rb[j]:
        matchedA[i] = true
        matchedB[j] = true
        inc matches
        break
  if matches == 0: return 0.0
  var transpositions = 0
  var k = 0
  for i in 0 ..< lenA:
    if matchedA[i]:
      while not matchedB[k]: inc k
      if ra[i] != rb[k]: inc transpositions
      inc k
  let m = matches.float
  let t = transpositions.float / 2.0
  result = (m / lenA.float + m / lenB.float + (m - t) / m) / 3.0

proc jaro*(a, b: string): float =
  ## Standard Jaro similarity on runes.
  ## Both empty -> 1.0; one empty -> 0.0; no matches -> 0.0.
  ## Decodes once and delegates to `jaroRunes <#jaroRunes>`_.
  jaroRunes(a.toRunes(), b.toRunes())

proc jaroWinkler*(a, b: string, prefixWeight = 0.1, maxPrefix = 4): float =
  ## Jaro-Winkler similarity: `jaro + l * prefixWeight * (1 - jaro)`,
  ## where `l` is the common rune-prefix length capped at `maxPrefix`.
  ## The standard bound (`result <= 1.0`) is meaningful for
  ## `prefixWeight <= 0.25`; larger weights can push results above 1.0.
  ## Raises ValueError for negative `prefixWeight` or `maxPrefix`.
  ## Decodes once; `jaro` and the prefix scan share the rune sequences.
  if prefixWeight < 0.0:
    raise newException(ValueError, "prefixWeight must be >= 0.0")
  if maxPrefix < 0:
    raise newException(ValueError, "maxPrefix must be >= 0")
  let ra = a.toRunes()
  let rb = b.toRunes()
  let j = jaroRunes(ra, rb)
  var prefixLen = 0
  let limit = min(min(ra.len, rb.len), maxPrefix)
  while prefixLen < limit and ra[prefixLen] == rb[prefixLen]:
    inc prefixLen
  result = j + prefixLen.float * prefixWeight * (1.0 - j)

# ============================================================
# N-gram distance
# ============================================================

proc ngramDistance*(a, b: string, n = 2): float =
  ## Character n-gram distance: `1 - jaccard(char grams of a, char grams
  ## of b)` using case-SENSITIVE grams (`NGramTokenizer` with
  ## `lowercase = false`), consistent with the case-sensitive word-level
  ## similarities. Returns 0.0 for identical strings (this also covers
  ## short/empty inputs whose n-gram sets would both be empty).
  if a == b: return 0.0
  let ga = NGramTokenizer(n: n, lowercase: false).tokenize(a)
  let gb = NGramTokenizer(n: n, lowercase: false).tokenize(b)
  result = 1.0 - jaccardSimilarity(ga, gb)
