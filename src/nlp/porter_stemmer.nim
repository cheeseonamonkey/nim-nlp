## Porter Stemmer for the nlp library.
##
## Based on the classic Porter Stemming Algorithm by Martin Porter (1980).
## https://tartarus.org/martin/PorterStemmer/def.txt

import std/[strutils, sequtils]

type
  PorterStemmer* = object
    ## Classic Porter stemmer for English

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

proc replaceEnding(s: string, oldLen: int, newSuffix: string): string =
  ## Replace the last `oldLen` characters with `newSuffix`.
  ## If oldLen >= s.len, returns just newSuffix.
  if oldLen >= s.len:
    return newSuffix
  s[0..^(oldLen + 1)] & newSuffix

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
  if result.endsWith("ational"): result = replaceEnding(result, 7, "ate")
  elif result.endsWith("tional"): result = replaceEnding(result, 6, "tion")
  elif result.endsWith("enci"): result = replaceEnding(result, 4, "ence")
  elif result.endsWith("anci"): result = replaceEnding(result, 4, "ance")
  elif result.endsWith("izer"): result = replaceEnding(result, 4, "ize")
  elif result.endsWith("abli"): result = replaceEnding(result, 4, "able")
  elif result.endsWith("alli"): result = replaceEnding(result, 4, "al")
  elif result.endsWith("entli"): result = replaceEnding(result, 5, "ent")
  elif result.endsWith("eli"): result = replaceEnding(result, 3, "e")
  elif result.endsWith("ousli"): result = replaceEnding(result, 5, "ous")
  elif result.endsWith("ization"): result = replaceEnding(result, 7, "ize")
  elif result.endsWith("ation"): result = replaceEnding(result, 5, "ate")
  elif result.endsWith("ator"): result = replaceEnding(result, 4, "ate")
  elif result.endsWith("alism"): result = replaceEnding(result, 5, "al")
  elif result.endsWith("iveness"): result = replaceEnding(result, 7, "ive")
  elif result.endsWith("fulness"): result = replaceEnding(result, 7, "ful")
  elif result.endsWith("ousness"): result = replaceEnding(result, 7, "ous")
  elif result.endsWith("aliti"): result = replaceEnding(result, 5, "al")
  elif result.endsWith("iviti"): result = replaceEnding(result, 5, "ive")
  elif result.endsWith("biliti"): result = replaceEnding(result, 6, "ble")

proc step3(s: string): string =
  result = s
  if result.len < 3: return
  if measure(result[0..^2]) == 0: return
  if result.endsWith("icate"): result = replaceEnding(result, 5, "ic")
  elif result.endsWith("ative"): result = replaceEnding(result, 5, "")
  elif result.endsWith("alize"): result = replaceEnding(result, 5, "al")
  elif result.endsWith("iciti"): result = replaceEnding(result, 5, "ic")
  elif result.endsWith("ical"): result = replaceEnding(result, 4, "ic")
  elif result.endsWith("ful"): result = replaceEnding(result, 3, "")
  elif result.endsWith("ness"): result = replaceEnding(result, 4, "")

proc step4(s: string): string =
  result = s
  if result.len < 3: return
  if measure(result[0..^2]) < 2: return
  if result.endsWith("al"): result = replaceEnding(result, 2, "")
  elif result.endsWith("ance"): result = replaceEnding(result, 4, "")
  elif result.endsWith("ence"): result = replaceEnding(result, 4, "")
  elif result.endsWith("er"): result = replaceEnding(result, 2, "")
  elif result.endsWith("ic"): result = replaceEnding(result, 2, "")
  elif result.endsWith("able"): result = replaceEnding(result, 4, "")
  elif result.endsWith("ible"): result = replaceEnding(result, 4, "")
  elif result.endsWith("ant"): result = replaceEnding(result, 3, "")
  elif result.endsWith("ement"): result = replaceEnding(result, 5, "")
  elif result.endsWith("ment"): result = replaceEnding(result, 4, "")
  elif result.endsWith("ent"): result = replaceEnding(result, 3, "")
  elif result.endsWith("ion"):
    if result.len >= 4 and result[^4] in {'s', 't'}:
      result = replaceEnding(result, 3, "")
  elif result.endsWith("ou"): result = replaceEnding(result, 2, "")
  elif result.endsWith("ism"): result = replaceEnding(result, 3, "")
  elif result.endsWith("ate"): result = replaceEnding(result, 3, "")
  elif result.endsWith("iti"): result = replaceEnding(result, 3, "")
  elif result.endsWith("ous"): result = replaceEnding(result, 3, "")
  elif result.endsWith("ive"): result = replaceEnding(result, 3, "")
  elif result.endsWith("ize"): result = replaceEnding(result, 3, "")

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
