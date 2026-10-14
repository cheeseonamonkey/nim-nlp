## Word n-gram generation tests for the nlp library.

import ../src/nlp/ngrams

proc testNgrams() =
  echo "=== Word N-grams ==="

  # Basic bigrams pin
  let g2 = ngrams(@["the", "quick", "brown", "fox"], 2)
  assert g2 == @[@["the", "quick"], @["quick", "brown"], @["brown", "fox"]],
    "ngrams bigrams pin"

  # Trigrams
  let g3 = ngrams(@["the", "quick", "brown", "fox"], 3)
  assert g3 == @[@["the", "quick", "brown"], @["quick", "brown", "fox"]],
    "ngrams trigrams pin"

  # bigrams wrapper
  assert bigrams(@["the", "quick", "brown", "fox"]) == g2, "bigrams wrapper"

  # trigrams wrapper
  assert trigrams(@["the", "quick", "brown", "fox"]) == g3, "trigrams wrapper"

  # n=1 wraps each token
  assert ngrams(@["the", "quick"], 1) == @[@["the"], @["quick"]], "ngrams n=1 wraps"
  assert bigrams(@["only"]) == @[], "bigrams single token empty"
  assert trigrams(@["a", "b"]) == @[], "trigrams too short empty"

  # Edge cases: n<=0 / empty / too short
  assert ngrams(@["the", "quick"], 0) == @[], "ngrams n=0"
  assert ngrams(@["the", "quick"], -1) == @[], "ngrams n<0"
  assert ngrams(@[], 2) == @[], "ngrams empty tokens"
  assert ngrams(@["hello"], 2) == @[], "ngrams too short"
  assert ngrams(@["a", "b"], 5) == @[], "ngrams n>len"
  assert bigrams(@[]) == @[], "bigrams empty"
  assert trigrams(@[]) == @[], "trigrams empty"

  # Full-length n returns the whole sequence as one gram
  assert ngrams(@["a", "b", "c"], 3) == @[@["a", "b", "c"]], "ngrams n==len"

  # ngramRange flattened shorter-first
  let r = ngramRange(@["a", "b", "c"], 1, 2)
  assert r == @[@["a"], @["b"], @["c"], @["a", "b"], @["b", "c"]],
    "ngramRange 1..2 flattened"

  # ngramRange single width equals ngrams
  assert ngramRange(@["a", "b", "c"], 2, 2) == ngrams(@["a", "b", "c"], 2),
    "ngramRange single width"

  # ngramRange invalid range
  assert ngramRange(@["a", "b", "c"], 3, 2) == @[], "ngramRange minN>maxN"
  assert ngramRange(@[], 1, 3) == @[], "ngramRange empty tokens"
  assert ngramRange(@["a"], 2, 3) == @[], "ngramRange all too short"

  # Unicode tokens pass through untouched
  let uni = ngrams(@["café", "naïve", "😀"], 2)
  assert uni == @[@["café", "naïve"], @["naïve", "😀"]], "ngrams unicode"

  # KEY accuracy pin: tokens containing spaces stay single elements
  assert ngrams(@["a b", "c"], 2) == @[@["a b", "c"]],
    "ngrams space-containing token boundary"
  let amb = ngrams(@["ab", "c d"], 2)
  assert amb == @[@["ab", "c d"]], "ngrams space token second position"
  assert amb != @[@["ab", "c"], @["c", "d"]], "ngrams no phantom split"
  assert amb[0].len == 2, "ngrams gram arity preserved"

  # Generic sliding-window base
  assert windows(@[1, 2, 3, 4], 2) == @[@[1, 2], @[2, 3], @[3, 4]],
    "windows ints n=2"
  assert windows(@[1, 2, 3], 1) == @[@[1], @[2], @[3]], "windows n=1"
  assert windows(@[1, 2, 3], 3) == @[@[1, 2, 3]], "windows n==len"
  assert windows(@[1, 2], 0) == @[], "windows n=0"
  assert windows(@[1, 2], -1) == @[], "windows n<0"
  assert windows(@[1], 2) == @[], "windows len<n"
  assert windows(newSeq[int](), 2) == @[], "windows empty"
  assert windows(@["a", "b", "c"], 2) == @[@["a", "b"], @["b", "c"]],
    "windows strings"
  # ngrams derives from windows
  assert ngrams(@["a", "b", "c"], 2) == windows(@["a", "b", "c"], 2),
    "ngrams equals windows"

  # joinGrams escape hatch back to string form
  assert joinGrams(ngrams(@["the", "quick", "brown"], 2)) ==
    @["the quick", "quick brown"], "joinGrams default sep"
  assert joinGrams(@[@["a", "b"], @["c", "d"]], "_") == @["a_b", "c_d"],
    "joinGrams custom sep"
  assert joinGrams(@[]) == @[], "joinGrams empty"
  assert joinGrams(@[@["solo"]]) == @["solo"], "joinGrams single"

  # Derivation equivalence: char windows rebuild ngramTokenize output
  let chars = @['h', 'e', 'l', 'l', 'o']
  var rebuilt: seq[string] = @[]
  for w in windows(chars, 2):
    var g = ""
    for c in w: g.add(c)
    rebuilt.add(g)
  assert rebuilt == @["he", "el", "ll", "lo"],
    "windows derives char bigrams"

  echo "  ALL PASS"

when isMainModule:
  testNgrams()
