## Word-level n-grams for the nlp library.
##
## Each n-gram is returned as an array of its component tokens
## (`seq[seq[string]]`), preserving token boundaries. Character n-grams
## (strings) stay in `tokenize` (`NGramTokenizer`, `ngramTokenize`).

proc windows*[T](items: openArray[T], n: int): seq[seq[T]] =
  ## Generic sliding window: consecutive length-`n` slices of `items`.
  ## Returns @[] when n <= 0 or `items.len < n`.
  if n <= 0 or items.len < n: return
  for i in 0..(items.len - n):
    result.add(@items[i ..< i + n])

proc joinGrams*(grams: seq[seq[string]], sep = " "): seq[string] =
  ## Join each gram with `sep` (escape hatch back to string form).
  for gram in grams:
    var s = ""
    for i, tok in gram:
      if i > 0: s.add(sep)
      s.add(tok)
    result.add(s)

proc ngrams*(tokens: seq[string], n: int): seq[seq[string]] =
  ## Generate word n-grams from a token sequence.
  ## n=2 on @["the", "quick", "brown"] -> @[@["the", "quick"], @["quick", "brown"]].
  ## Returns @[] when n <= 0, `tokens` is empty, or `tokens.len < n`.
  ## n=1 returns each token wrapped in its own array.
  windows(tokens, n)

proc bigrams*(tokens: seq[string]): seq[seq[string]] =
  ## Word bigrams (n=2) from a token sequence.
  ngrams(tokens, 2)

proc trigrams*(tokens: seq[string]): seq[seq[string]] =
  ## Word trigrams (n=3) from a token sequence.
  ngrams(tokens, 3)

proc ngramRange*(tokens: seq[string], minN, maxN: int): seq[seq[string]] =
  ## Word n-grams for every n in minN..maxN, flattened (shorter grams first).
  ## Returns @[] when the range is empty or invalid (minN > maxN).
  if minN > maxN: return
  for n in minN..maxN:
    for gram in ngrams(tokens, n):
      result.add(gram)
