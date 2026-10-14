## Stopwords for the nlp library.
##
## The built-in lists are compile-time constants in ``private/lexicon``; no
## data files are read at install, compile, or run time.

import std/[sets, strutils, sequtils, unicode]
import private/lexicon

type
  StopWords* = HashSet[string]

proc buildStopWords(builtin: openArray[string]): StopWords =
  result = initHashSet[string]()
  for w in builtin:
    result.incl(w.toLower())

let
  englishStopWords* = buildStopWords(builtinEnglishStopwords)
  spanishStopWords* = buildStopWords(builtinSpanishStopwords)
  frenchStopWords* = buildStopWords(builtinFrenchStopwords)
  germanStopWords* = buildStopWords(builtinGermanStopwords)

proc isStopWord*(word: string, language: string): bool =
  ## Check if a word is a stopword in the given language.
  ## Case-insensitive comparison. Returns false for unknown languages.
  if word.len == 0: return false
  let w = word.toLower()
  case language.toLower()
  of "english", "en": w in englishStopWords
  of "spanish", "es": w in spanishStopWords
  of "french", "fr": w in frenchStopWords
  of "german", "de": w in germanStopWords
  else: false

proc removeStopWords*(tokens: seq[string], language: string): seq[string] =
  ## Remove stopwords from a token sequence by language name.
  ## Case-insensitive comparison.
  if tokens.len == 0: return
  case language.toLower()
  of "english", "en": tokens.filterIt(it.toLower() notin englishStopWords)
  of "spanish", "es": tokens.filterIt(it.toLower() notin spanishStopWords)
  of "french", "fr": tokens.filterIt(it.toLower() notin frenchStopWords)
  of "german", "de": tokens.filterIt(it.toLower() notin germanStopWords)
  else: tokens

proc removeStopWords*(tokens: seq[string], stopwords: StopWords): seq[string] =
  ## Remove stopwords using a custom set.
  ##
  ## Tokens are lowercased before comparison, so a mixed-case token matches a
  ## lowercase entry. The caller's set is used *verbatim*: it is NOT lowercased
  ## for you, so entries must already be lowercase (e.g. `toHashSet(["the"])`,
  ## not `toHashSet(["THE"])`).
  if tokens.len == 0: return
  tokens.filterIt(it.toLower() notin stopwords)
