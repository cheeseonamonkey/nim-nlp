## Tokenizers for the nlp library.
##
## Provides the `Tokenizer` concept and built-in implementations.

import std/[unicode, strutils, sets]
import ngrams

type
  Tokenizer* = concept x
    tokenize(x, string) is seq[string]

# ============================================================
# Tokenizer types
# ============================================================

type
  WordTokenizer* = object
    ## Splits text into alphanumeric words.
    lowercase*: bool
    minLength*: int

  PatternTokenizer* = object
    ## Extracts tokens matching a small, regex-like pattern subset.
    ## Supports: \w+ (alphanumeric), \d+ (digits), \s+ (whitespace),
    ## and [chars] (character class). This is not a general regex engine.
    pattern*: string
    lowercase*: bool
    minLength*: int

  WhitespaceTokenizer* = object
    ## Splits on whitespace only.
    lowercase*: bool
    minLength*: int

  NGramTokenizer* = object
    ## Produces character n-grams.
    n*: int
    lowercase*: bool

  SentenceTokenizer* = object
    ## Splits text into sentences on `.` `!` `?`.
    discardEmpty*: bool

# ============================================================
# Implementations
# ============================================================

proc flushToken*(current: string, lowercase: bool, minLength: int,
                 result: var seq[string]) =
  ## Append one accumulated token run, honoring `lowercase`/`minLength`.
  ## Shared drain step for the character-scanning tokenizers.
  if current.len == 0: return
  let tok = if lowercase: current.toLower() else: current
  if tok.len >= minLength: result.add(tok)

template accumulateTokens(t: object, text: string, matches: untyped) =
  ## Shared character-accumulation skeleton for tokenizers that scan a
  ## string character by character. `matches` is evaluated with `ch` bound
  ## to the current character; runs of matching characters become tokens.
  if text.len != 0:
    var current = ""
    for ch {.inject.} in text:
      if matches:
        current.add(ch)
      else:
        flushToken(current, t.lowercase, t.minLength, result)
        current = ""
    flushToken(current, t.lowercase, t.minLength, result)

proc tokenize*(t: WordTokenizer, text: string): seq[string] =
  ## Split text into alphanumeric words.
  accumulateTokens(t, text, ch.isAlphaNumeric)

proc matchPattern(t: PatternTokenizer, ch: char): bool =
  ## Check if a character matches the tokenizer's simple pattern.
  if t.pattern == r"\w+":
    result = ch.isAlphaNumeric
  elif t.pattern == r"\d+":
    result = ch.isDigit
  elif t.pattern == r"\s+":
    result = ch in Whitespace
  elif t.pattern.startsWith("[") and t.pattern.endsWith("]"):
    let chars = t.pattern[1..^2]
    result = ch in chars
  else:
    # Fallback: treat as literal character match
    result = ch in t.pattern

proc tokenize*(t: PatternTokenizer, text: string): seq[string] =
  ## Extract tokens matching a simple pattern (\w+, \d+, [chars]).
  accumulateTokens(t, text, t.matchPattern(ch))

proc tokenize*(t: WhitespaceTokenizer, text: string): seq[string] =
  ## Split on whitespace.
  if text.len == 0: return
  for tok in strutils.splitWhitespace(text):
    let tk = if t.lowercase: tok.toLower() else: tok
    if tk.len >= t.minLength: result.add(tk)

proc tokenize*(t: NGramTokenizer, text: string): seq[string] =
  ## Produce Unicode-character n-grams (not UTF-8 byte slices).
  if t.n <= 0: return
  let normalized = if t.lowercase: text.toLower() else: text
  let chars = normalized.toRunes()
  if chars.len < t.n: return
  for w in windows(chars, t.n):
    var gram = newStringOfCap(t.n)
    for r in w:
      gram.add($r)
    result.add(gram)

proc isSentenceBoundary*(text: string, i: int): bool =
  ## A sentence terminator (`.`, `!`, `?`) ends a sentence only when
  ## followed by whitespace or end of input — so `"Hi."` (quote after
  ## the stop) does not split, matching long-standing splitter behavior.
  text[i] in {'.', '!', '?'} and
    (i + 1 >= text.len or text[i+1] in Whitespace)

proc pushSentence(current: string, t: SentenceTokenizer,
                  result: var seq[string]) =
  ## Append one accumulated sentence, honoring `discardEmpty`.
  let stripped = current.strip()
  if t.discardEmpty:
    if stripped.len > 0: result.add(stripped)
  else:
    result.add(stripped)

proc collectSentences(text: string, t: SentenceTokenizer,
                      result: var seq[string]) =
  ## Accumulation loop for the sentence splitter: scan, cut at each
  ## boundary, drain any trailing fragment.
  var current = newStringOfCap(text.len)
  for i in 0 ..< text.len:
    current.add(text[i])
    if isSentenceBoundary(text, i):
      pushSentence(current, t, result)
      current.setLen(0)
  if current.len > 0:
    pushSentence(current, t, result)

proc tokenize*(t: SentenceTokenizer, text: string): seq[string] =
  ## Split text into sentences.
  if text.len == 0:
    if not t.discardEmpty: result.add("")
    return
  collectSentences(text, t, result)

# ============================================================
# Convenience functions
# ============================================================

proc wordTokenize*(text: string, lowercase = true, minLength = 1): seq[string] =
  ## Quick word tokenization.
  WordTokenizer(lowercase: lowercase, minLength: minLength).tokenize(text)

proc whitespaceTokenize*(text: string, lowercase = true, minLength = 1): seq[string] =
  ## Quick whitespace tokenization.
  WhitespaceTokenizer(lowercase: lowercase, minLength: minLength).tokenize(text)

proc ngramTokenize*(text: string, n: int, lowercase = true): seq[string] =
  ## Quick character n-gram tokenization.
  NGramTokenizer(n: n, lowercase: lowercase).tokenize(text)

proc splitSentences*(text: string, discardEmpty = true): seq[string] =
  ## Quick sentence splitting.
  SentenceTokenizer(discardEmpty: discardEmpty).tokenize(text)
