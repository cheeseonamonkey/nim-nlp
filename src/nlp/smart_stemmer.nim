## Smart stemmer for the nlp library.
##
## Combines pattern-based suffix stripping with irregular-form data loaded
## from CSV files under ``data/``.  The files are embedded at compile time,
## so users do not need to arrange runtime data paths.

import std/[tables, strutils, sequtils, sets]
import ./csv_data

const
  irregularPluralsData = staticRead("../../data/irregular_plurals.csv")
  irregularVerbsData = staticRead("../../data/irregular_verbs.csv")
  noChangeWordsData = staticRead("../../data/no_change_words.csv")

proc loadPairTable(data, filename: string): Table[string, string] =
  result = initTable[string, string]()
  for row in csvRows(data, filename):
    # The first row is the CSV header.
    if row.len >= 2 and row[0].len > 0 and row[0] != "plural" and row[0] != "form":
      result[row[0].toLowerAscii()] = row[1].toLowerAscii()

proc loadWordSet(data, filename: string): HashSet[string] =
  result = initHashSet[string]()
  for row in csvRows(data, filename):
    if row.len >= 1 and row[0].len > 0 and row[0] != "word":
      result.incl(row[0].toLowerAscii())

let
  ## Irregular plural forms and their singular equivalents.
  irregularPlurals* = loadPairTable(irregularPluralsData, "irregular_plurals.csv")
  ## Irregular verb forms and their infinitives.
  irregularVerbs* = loadPairTable(irregularVerbsData, "irregular_verbs.csv")
  ## Words whose singular and plural forms are identical.
  noChangeWords* = loadWordSet(noChangeWordsData, "no_change_words.csv")

# ============================================================
# Internal helpers
# ============================================================

proc isVowel(ch: char): bool =
  ch in {'a', 'e', 'i', 'o', 'u'}

proc measure(s: string): int =
  ## Count VC sequences (Porter's 'measure').
  var count = 0
  var consonantRun = false
  for ch in s:
    if isVowel(ch):
      consonantRun = false
    else:
      if not consonantRun:
        inc count
        consonantRun = true

proc hasVowel(s: string): bool =
  for ch in s:
    if isVowel(ch): return true
  false

proc endsDoubleConsonant(s: string): bool =
  if s.len < 2: false
  elif s[^1] != s[^2]: false
  elif isVowel(s[^1]): false
  else: true

proc cvc(s: string): bool =
  if s.len < 3: false
  elif isVowel(s[^1]): false
  elif not isVowel(s[^2]): false
  elif isVowel(s[^3]): false
  elif s[^1] in {'w', 'x', 'y'}: false
  else: true

# ============================================================
# Smart plural detection
# ============================================================

proc detectPlural(w: string): string =
  ## Detect and singularize plural forms.
  ## Returns the singular form, or the input unchanged.
  # Check irregular forms first
  if irregularPlurals.hasKey(w):
    return irregularPlurals[w]

  # Words that don't change in plural
  if w in noChangeWords:
    return w

  # Pattern-based plural handling (longest suffix first)

  # -ices → -ex (appendices → appendix, codices → codex)
  if w.endsWith("ices") and w.len > 5:
    let stem = w[0..^4]
    if stem.endsWith("pend") or stem.endsWith("dex"):
      return stem & "ex"

  # -ies → -y (cities → city, puppies → puppy)
  if w.endsWith("ies") and w.len > 4:
    let stem = w[0..^4]
    if stem.len > 0 and not isVowel(stem[^1]):
      return stem & "y"

  # -ves → -f or -fe (wolves → wolf, knives → knife)
  if w.endsWith("ves") and w.len > 4:
    let stem = w[0..^4]
    if stem.endsWith("ar") or stem.endsWith("ol") or stem.endsWith("ea"):
      return stem & "f"
    if stem.endsWith("ni") or stem.endsWith("li") or stem.endsWith("ri"):
      return stem & "fe"
    return stem & "f"

  # -oes → -o (heroes → hero, potatoes → potato)
  if w.endsWith("oes") and w.len > 4:
    return w[0..^3]

  # -ches/-shes/-xes/-zes/-sses → remove -es
  if w.endsWith("ches") or w.endsWith("shes") or w.endsWith("xes") or
     w.endsWith("zes") or w.endsWith("sses"):
    if w.len > 4:
      return w[0..^3]

  # -ses → -sis (analyses → analysis, diagnoses → diagnosis)
  if w.endsWith("ses") and w.len > 4:
    let stem = w[0..^3]
    if stem.endsWith("aly") or stem.endsWith("agno") or stem.endsWith("hesi"):
      return stem & "is"

  # Default: just remove -s
  if w.endsWith("s") and not w.endsWith("ss") and w.len > 3:
    return w[0..^2]

  result = w

# ============================================================
# Main stem function
# ============================================================

type
  SmartStemmer* = object
    ## A smart English stemmer combining pattern rules with irregular form lookup.

proc stem*(s: SmartStemmer, word: string): string =
  ## Stem a single English word using smart pattern rules + Porter-style suffix stripping.
  ## Handles irregular plurals, irregular verbs, and regular suffixes.
  if word.len <= 2: return word.toLowerAscii()

  let w = word.toLowerAscii()

  # Check irregular forms first
  if irregularPlurals.hasKey(w):
    return irregularPlurals[w]
  if irregularVerbs.hasKey(w):
    return irregularVerbs[w]

  # Words that don't change
  if w in noChangeWords:
    return w

  # Handle plurals first
  result = detectPlural(w)

  # Track if original word ended in 'y' (for step 2)
  let originalEndedInY = w.endsWith("y")
  # Track if we already handled -y via plural detection (e.g., cities → city)
  let alreadyHandledY = w.endsWith("ies") and result.endsWith("y")

  # Apply Porter-style suffix stripping
  # Step 1: Handle -eed, -ed, -ing
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

  # Step 2: Handle -y (skip if already handled by plural detection)
  if not alreadyHandledY and originalEndedInY and result.endsWith("y") and hasVowel(result[0..^2]):
    result = result[0..^2] & "i"

  # Step 3: Handle -ational, -tional, etc.
  if result.endsWith("ational"):
    let stem = result[0..^8]
    if measure(stem) > 0: result = stem & "ate"
  elif result.endsWith("tional"):
    let stem = result[0..^7]
    if measure(stem) > 0: result = stem & "tion"
  elif result.endsWith("enci"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem & "ence"
  elif result.endsWith("anci"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem & "ance"
  elif result.endsWith("izer"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem & "ize"
  elif result.endsWith("abli"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem & "able"
  elif result.endsWith("alli"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem & "al"
  elif result.endsWith("entli"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "ent"
  elif result.endsWith("eli"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem & "e"
  elif result.endsWith("ousli"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "ous"
  elif result.endsWith("ization"):
    let stem = result[0..^8]
    if measure(stem) > 0: result = stem & "ize"
  elif result.endsWith("ation"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "ate"
  elif result.endsWith("ator"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem & "ate"
  elif result.endsWith("alism"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "al"
  elif result.endsWith("iveness"):
    let stem = result[0..^8]
    if measure(stem) > 0: result = stem & "ive"
  elif result.endsWith("fulness"):
    let stem = result[0..^8]
    if measure(stem) > 0: result = stem & "ful"
  elif result.endsWith("ousness"):
    let stem = result[0..^8]
    if measure(stem) > 0: result = stem & "ous"
  elif result.endsWith("aliti"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "al"
  elif result.endsWith("iviti"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "ive"
  elif result.endsWith("biliti"):
    let stem = result[0..^7]
    if measure(stem) > 0: result = stem & "ble"
  elif result.endsWith("logi"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem & "log"

  # Step 4: Handle -icate, -ative, etc.
  if result.endsWith("icate"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "ic"
  elif result.endsWith("ative"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem
  elif result.endsWith("alize"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "al"
  elif result.endsWith("iciti"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "ic"
  elif result.endsWith("ical"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem & "ic"
  elif result.endsWith("ful"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ness"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem

  # Step 5: Handle -ement, -ment, etc.
  if result.endsWith("ement"):
    let stem = result[0..^6]
    if measure(stem) > 0: result = stem & "e"
  elif result.endsWith("ment"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem
  elif result.endsWith("able"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ible"):
    let stem = result[0..^5]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ant"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ent"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ism"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ate"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("iti"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ous"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ive"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ize"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ion"):
    let stem = result[0..^4]
    if measure(stem) > 0 and stem.len > 0 and stem[^1] in {'s', 't'}:
      result = stem
  elif result.endsWith("er"):
    let stem = result[0..^3]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ed"):
    let stem = result[0..^3]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ing"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem
  elif result.endsWith("ly"):
    let stem = result[0..^3]
    if measure(stem) > 0: result = stem
  elif result.endsWith("al"):
    let stem = result[0..^3]
    if measure(stem) > 0: result = stem
  elif result.endsWith("y"):
    let stem = result[0..^2]
    if measure(stem) > 1: result = stem & "i"

  # Step 6: Handle final -e
  if result.endsWith("e"):
    let stem = result[0..^2]
    if measure(stem) > 1: result = stem
    elif measure(stem) == 1 and not cvc(stem): result = stem

  # Step 7: Handle final double consonant
  if measure(result) > 1 and endsDoubleConsonant(result) and result[^1] == 'l':
    result = result[0..^2]

# ============================================================
# Convenience
# ============================================================

proc smartStemWord*(word: string): string =
  ## Quick stem a single word.
  SmartStemmer().stem(word)

proc stemTokens*(tokens: seq[string]): seq[string] =
  ## Stem a token sequence. Returns empty for empty input.
  if tokens.len == 0: return
  let s = SmartStemmer()
  tokens.mapIt(s.stem(it))
