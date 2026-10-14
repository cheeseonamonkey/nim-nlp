## Smart stemmer for the nlp library.
##
## Combines pattern-based suffix stripping with irregular-form data held as
## compile-time constants in ``private/lexicon``; nothing is read from disk at
## install, compile, or run time.

import std/[tables, strutils, sequtils, sets, unicode]
import private/lexicon

let
  ## Irregular plural forms and their singular equivalents.
  irregularPlurals* = builtinIrregularPlurals.toTable
  ## Irregular verb forms and their infinitives.
  irregularVerbs* = builtinIrregularVerbs.toTable
  ## Words whose singular and plural forms are identical.
  noChangeWords* = builtinNoChangeWords.toHashSet

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
  result = count

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

type
  SuffixRule = tuple[suffix: string, repl: string, minMeasure: int, prevIn: set[char]]
    ## One suffix-stripping rule.  ``minMeasure`` is the strict lower bound on
    ## the stem's ``measure``; ``prevIn``, when non-empty, additionally requires
    ## the stem's last character to be in the given set.

proc applySuffixRules(s: string, rules: openArray[SuffixRule]): string =
  ## Apply the first matching suffix rule.  Even when the matching rule's guard
  ## rejects the stem, the scan stops and the input is returned unchanged;
  ## later rules are deliberately not tried.
  for r in rules:
    if s.endsWith(r.suffix):
      let stem = s[0 ..< s.len - r.suffix.len]
      if stem.len > 0 and measure(stem) > r.minMeasure and
         (r.prevIn == {} or stem[^1] in r.prevIn):
        return stem & r.repl
      return s
  s

proc applyEdIngStem(stem: string): string =
  ## Shared post-processing for the -ed and -ing branches of step 1.
  result = stem
  if result.endsWith("at") or result.endsWith("bl") or result.endsWith("iz"):
    result.add("e")
  elif endsDoubleConsonant(result) and result[^1] notin {'l', 's', 'z'}:
    result = result[0..^2]
  elif measure(result) == 1 and cvc(result):
    result.add("e")

const
  ## Smart step 3 rules, in the original first-match order.
  smartStep3Rules: array[21, SuffixRule] = [
    ("ational", "ate", 0, {}), ("tional", "tion", 0, {}),
    ("enci", "ence", 0, {}), ("anci", "ance", 0, {}),
    ("izer", "ize", 0, {}), ("abli", "able", 0, {}),
    ("alli", "al", 0, {}), ("entli", "ent", 0, {}),
    ("eli", "e", 0, {}), ("ousli", "ous", 0, {}),
    ("ization", "ize", 0, {}), ("ation", "ate", 0, {}),
    ("ator", "ate", 0, {}), ("alism", "al", 0, {}),
    ("iveness", "ive", 0, {}), ("fulness", "ful", 0, {}),
    ("ousness", "ous", 0, {}), ("aliti", "al", 0, {}),
    ("iviti", "ive", 0, {}), ("biliti", "ble", 0, {}),
    ("logi", "log", 0, {})]

  ## Smart step 4 rules, in the original first-match order.
  smartStep4Rules: array[7, SuffixRule] = [
    ("icate", "ic", 0, {}), ("ative", "", 0, {}), ("alize", "al", 0, {}),
    ("iciti", "ic", 0, {}), ("ical", "ic", 0, {}), ("ful", "", 0, {}),
    ("ness", "", 0, {})]

  ## Smart step 5 rules, in the original first-match order.  The final "y"
  ## rule needs a stricter measure bound and rewrites to "i".
  smartStep5Rules: array[19, SuffixRule] = [
    ("ement", "e", 0, {}), ("ment", "", 0, {}), ("able", "", 0, {}),
    ("ible", "", 0, {}), ("ant", "", 0, {}), ("ent", "", 0, {}),
    ("ism", "", 0, {}), ("ate", "", 0, {}), ("iti", "", 0, {}),
    ("ous", "", 0, {}), ("ive", "", 0, {}), ("ize", "", 0, {}),
    ("ion", "", 0, {'s', 't'}), ("er", "", 0, {}), ("ed", "", 0, {}),
    ("ing", "", 0, {}), ("ly", "", 0, {}), ("al", "", 0, {}),
    ("y", "i", 1, {})]

# ============================================================
# Smart plural detection
# ============================================================

proc detectPlural*(w: string): string =
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

# ============================================================
# Smart pipeline stages
# ============================================================

proc lookupIrregular*(w: string): tuple[found: bool, stem: string] =
  ## Irregular-form dispatch: plural table first, then verb table.
  ## Returns ``(false, w)`` when ``w`` is regular.
  if irregularPlurals.hasKey(w):
    return (true, irregularPlurals[w])
  if irregularVerbs.hasKey(w):
    return (true, irregularVerbs[w])
  (false, w)

proc smartStepEdIng*(s: string): string =
  ## Step 1: -eed/-ed/-ing handling with at/bl/iz + double-consonant cleanup.
  result = s
  if result.endsWith("eed"):
    let stem = result[0..^4]
    if measure(stem) > 0: result = stem & "ee"
  elif result.endsWith("ed"):
    let stem = result[0..^3]
    if hasVowel(stem): result = applyEdIngStem(stem)
  elif result.endsWith("ing"):
    let stem = result[0..^4]
    if hasVowel(stem): result = applyEdIngStem(stem)

proc smartStepY*(s, original: string): string =
  ## Step 2: -y -> -i.  Needs the pre-strip ``original`` to detect words
  ## already singularized by plural handling (e.g. cities -> city).
  let alreadyHandledY = original.endsWith("ies") and s.endsWith("y")
  if not alreadyHandledY and original.endsWith("y") and s.endsWith("y") and
     hasVowel(s[0..^2]):
    return s[0..^2] & "i"
  s

proc smartStep3*(s: string): string =
  ## Step 3: -ational, -tional, etc.
  applySuffixRules(s, smartStep3Rules)

proc smartStep4*(s: string): string =
  ## Step 4: -icate, -ative, etc.
  applySuffixRules(s, smartStep4Rules)

proc smartStep5*(s: string): string =
  ## Step 5: -ement, -ment, etc. (incl. the strict -y -> -i rule).
  applySuffixRules(s, smartStep5Rules)

proc smartStepFinalE*(s: string): string =
  ## Step 6: final -e strip.
  result = s
  if result.endsWith("e"):
    let stem = result[0..^2]
    if measure(stem) > 1: result = stem
    elif measure(stem) == 1 and not cvc(stem): result = stem

proc smartStepDoubleL*(s: string): string =
  ## Step 7: final double-consonant -l cleanup.
  result = s
  if measure(result) > 1 and endsDoubleConsonant(result) and result[^1] == 'l':
    result = result[0..^2]

proc stem*(s: SmartStemmer, word: string): string =
  ## Stem a single English word using smart pattern rules + Porter-style suffix stripping.
  ## Handles irregular plurals, irregular verbs, and regular suffixes.
  if word.len <= 2: return word.toLower()

  let w = word.toLower()

  # Check irregular forms first
  let irr = lookupIrregular(w)
  if irr.found:
    return irr.stem

  # Words that don't change
  if w in noChangeWords:
    return w

  # Handle plurals first
  result = detectPlural(w)

  # Porter-style suffix stripping, one named stage per step
  result = smartStepEdIng(result)
  result = smartStepY(result, w)
  result = smartStep3(result)
  result = smartStep4(result)
  result = smartStep5(result)
  result = smartStepFinalE(result)
  result = smartStepDoubleL(result)

# ============================================================
# Convenience
# ============================================================

proc smartStemWord*(word: string): string =
  ## Quick stem a single word.
  SmartStemmer().stem(word)

proc smartStemTokens*(tokens: seq[string]): seq[string] =
  ## Stem a token sequence with the smart stemmer. Returns empty for empty
  ## input. Named ``smartStemTokens`` so it does not collide with the Porter
  ## ``stemTokens`` re-exported by the ``nlp`` facade.
  if tokens.len == 0: return
  let s = SmartStemmer()
  tokens.mapIt(s.stem(it))
