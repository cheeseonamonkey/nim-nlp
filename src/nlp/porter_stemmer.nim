## Porter Stemmer for the nlp library.
##
## Based on the classic Porter Stemming Algorithm by Martin Porter (1980).
## https://tartarus.org/martin/PorterStemmer/def.txt

import std/[strutils, sequtils]

type
  PorterStemmer* = object
    ## Classic Porter stemmer for English

  SuffixRule = tuple[suffix: string, repl: string, prevIn: set[char]]
    ## One suffix-stripping rule.  ``prevIn``, when non-empty, restricts the
    ## replacement to stems whose last character is in the given set.

# ============================================================
# Internal helpers
# ============================================================

proc isConsonant(s: string, i: int): bool =
  case s[i]
  of 'a', 'e', 'i', 'o', 'u': false
  of 'y':
    if i == 0: true
    else: not isConsonant(s, i - 1)
  else: true

proc measure(s: string): int =
  var i = 0
  var count = 0
  var consonantRun = false
  while i < s.len:
    if isConsonant(s, i):
      consonantRun = true
    else:
      if consonantRun:
        inc count
        consonantRun = false
    inc i
  result = count

proc hasVowel(s: string): bool =
  for i in 0..<s.len:
    if not isConsonant(s, i): return true
  false

proc endsDoubleConsonant(s: string): bool =
  if s.len < 2: false
  elif s[^1] != s[^2]: false
  else: isConsonant(s, s.len - 1)

proc cvc(s: string): bool =
  if s.len < 3: false
  elif isConsonant(s, s.len - 1): false
  elif not isConsonant(s, s.len - 2): false
  elif not isConsonant(s, s.len - 3): false
  elif s[^1] in {'w', 'x', 'y'}: false
  else: true

proc applySuffixRules(s: string, rules: openArray[SuffixRule]): string =
  ## Apply the first matching suffix rule.  Even when the matching rule's guard
  ## (`prevIn`) rejects the stem, the scan stops and the input is returned
  ## unchanged; later rules are deliberately not tried.
  for r in rules:
    if s.endsWith(r.suffix):
      let stem = s[0 ..< s.len - r.suffix.len]
      if r.prevIn == {} or (stem.len > 0 and stem[^1] in r.prevIn):
        return stem & r.repl
      return s
  s

const
  ## Step 2 rules, in the original first-match order.
  porterStep2Rules: array[20, SuffixRule] = [
    ("ational", "ate", {}), ("tional", "tion", {}), ("enci", "ence", {}),
    ("anci", "ance", {}), ("izer", "ize", {}), ("abli", "able", {}),
    ("alli", "al", {}), ("entli", "ent", {}), ("eli", "e", {}),
    ("ousli", "ous", {}), ("ization", "ize", {}), ("ation", "ate", {}),
    ("ator", "ate", {}), ("alism", "al", {}), ("iveness", "ive", {}),
    ("fulness", "ful", {}), ("ousness", "ous", {}), ("aliti", "al", {}),
    ("iviti", "ive", {}), ("biliti", "ble", {})]

  ## Step 3 rules, in the original first-match order.
  porterStep3Rules: array[7, SuffixRule] = [
    ("icate", "ic", {}), ("ative", "", {}), ("alize", "al", {}),
    ("iciti", "ic", {}), ("ical", "ic", {}), ("ful", "", {}),
    ("ness", "", {})]

  ## Step 4 rules, in the original first-match order.  The "ion" rule is
  ## guarded on the character preceding the suffix.
  porterStep4Rules: array[19, SuffixRule] = [
    ("al", "", {}), ("ance", "", {}), ("ence", "", {}), ("er", "", {}),
    ("ic", "", {}), ("able", "", {}), ("ible", "", {}), ("ant", "", {}),
    ("ement", "", {}), ("ment", "", {}), ("ent", "", {}),
    ("ion", "", {'s', 't'}), ("ou", "", {}), ("ism", "", {}),
    ("ate", "", {}), ("iti", "", {}), ("ous", "", {}), ("ive", "", {}),
    ("ize", "", {})]

# ============================================================
# Porter steps
# ============================================================

proc step1a(s: string): string =
  result = s
  if result.endsWith("sses"): result = result[0..^3]
  elif result.endsWith("ies"): result = result[0..^3]
  elif result.endsWith("ss"): discard
  elif result.endsWith("s"): result = result[0..^2]

proc step1b(s: string): string =
  result = s
  if result.endsWith("eed"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem & "ee"
  elif result.endsWith("ed"):
    let stem = result[0..^3]
    if hasVowel(stem):
      result = stem
      if result.endsWith("at") or result.endsWith("bl") or result.endsWith("iz"):
        result.add("e")
      elif endsDoubleConsonant(result) and result[^1] notin {'l', 's', 'z'}:
        result = result[0..^2]
      elif measure(result) == 1 and cvc(result):
        result.add("e")
  elif result.endsWith("ing"):
    let stem = result[0..^4]
    if hasVowel(stem):
      result = stem
      if result.endsWith("at") or result.endsWith("bl") or result.endsWith("iz"):
        result.add("e")
      elif endsDoubleConsonant(result) and result[^1] notin {'l', 's', 'z'}:
        result = result[0..^2]
      elif measure(result) == 1 and cvc(result):
        result.add("e")

proc step1c(s: string): string =
  result = s
  if result.endsWith("y") and hasVowel(result[0..^2]):
    result = result[0..^2] & "i"

proc step2(s: string): string =
  result = s
  if result.len < 3: return
  if measure(result[0..^2]) == 0: return
  result = applySuffixRules(result, porterStep2Rules)

proc step3(s: string): string =
  result = s
  if result.len < 3: return
  if measure(result[0..^2]) == 0: return
  result = applySuffixRules(result, porterStep3Rules)

proc step4(s: string): string =
  result = s
  if result.len < 3: return
  if measure(result[0..^2]) < 2: return
  result = applySuffixRules(result, porterStep4Rules)

proc step5a(s: string): string =
  result = s
  if result.endsWith("e"):
    let stem = result[0..^2]
    if measure(stem) > 1: result = stem
    elif measure(stem) == 1 and not cvc(stem): result = stem

proc step5b(s: string): string =
  result = s
  if measure(result) > 1 and endsDoubleConsonant(result) and result[^1] == 'l':
    result = result[0..^2]

# ============================================================
# Public API
# ============================================================

proc stem*(ps: PorterStemmer, word: string): string =
  ## Stem a single English word using the Porter algorithm.
  ## Returns the word as-is for empty strings or words <= 2 characters.
  if word.len == 0: return ""
  if word.len <= 2: return word.toLowerAscii()
  result = word.toLowerAscii()
  result = step1a(result)
  result = step1b(result)
  result = step1c(result)
  result = step2(result)
  result = step3(result)
  result = step4(result)
  result = step5a(result)
  result = step5b(result)

proc stemWord*(word: string): string =
  ## Quick stem a single word.
  PorterStemmer().stem(word)

proc stemTokens*(tokens: seq[string]): seq[string] =
  ## Stem a sequence of tokens. Returns empty for empty input.
  if tokens.len == 0: return
  let ps = PorterStemmer()
  tokens.mapIt(ps.stem(it))
