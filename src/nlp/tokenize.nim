## Tokenizers for the nlp library.
##
## Provides the `Tokenizer` concept and built-in implementations.

import std/[unicode, strutils, sets]

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

proc tokenize*(t: WordTokenizer, text: string): seq[string] =
  ## Split text into alphanumeric words.
  if text.len == 0: return
  var current = ""
  for ch in text:
    if ch.isAlphaNumeric:
      current.add(ch)
    else:
      if current.len > 0:
        let tok = if t.lowercase: current.toLowerAscii() else: current
        if tok.len >= t.minLength: result.add(tok)
        current = ""
  if current.len > 0:
    let tok = if t.lowercase: current.toLowerAscii() else: current
    if tok.len >= t.minLength: result.add(tok)

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
  if text.len == 0: return
  var current = ""
  for ch in text:
    if t.matchPattern(ch):
      current.add(ch)
    else:
      if current.len > 0:
        let tok = if t.lowercase: current.toLowerAscii() else: current
        if tok.len >= t.minLength: result.add(tok)
        current = ""
  if current.len > 0:
    let tok = if t.lowercase: current.toLowerAscii() else: current
    if tok.len >= t.minLength: result.add(tok)

proc tokenize*(t: WhitespaceTokenizer, text: string): seq[string] =
  ## Split on whitespace.
  if text.len == 0: return
  for tok in strutils.splitWhitespace(text):
    let tk = if t.lowercase: tok.toLowerAscii() else: tok
    if tk.len >= t.minLength: result.add(tk)

proc tokenize*(t: NGramTokenizer, text: string): seq[string] =
  ## Produce Unicode-character n-grams (not UTF-8 byte slices).
  if t.n <= 0: return
  let normalized = if t.lowercase: text.toLowerAscii() else: text
  let chars = normalized.toRunes()
  if chars.len < t.n: return
  for i in 0..(chars.len - t.n):
    var gram = newStringOfCap(t.n)
    for j in i..<(i + t.n):
      gram.add($chars[j])
    result.add(gram)

proc tokenize*(t: SentenceTokenizer, text: string): seq[string] =
  ## Split text into sentences.
  if text.len == 0:
    if not t.discardEmpty: result.add("")
    return
  var current = newStringOfCap(text.len)
  var i = 0
  while i < text.len:
    current.add(text[i])
    if text[i] in {'.', '!', '?'}:
      if i + 1 >= text.len or text[i+1] in Whitespace:
        if t.discardEmpty:
          let stripped = current.strip()
          if stripped.len > 0: result.add(stripped)
        else:
          if current.len > 0: result.add(current.strip())
        current.setLen(0)
    inc i
  if current.len > 0:
    let stripped = current.strip()
    if t.discardEmpty:
      if stripped.len > 0: result.add(stripped)
    else:
      result.add(stripped)

# ============================================================
# N-grams from token sequences
# ============================================================

proc ngrams*(tokens: seq[string], n: int): seq[string] =
  ## Generate word n-grams from a token sequence.
  if n <= 0 or tokens.len == 0 or tokens.len < n: return
  for i in 0..(tokens.len - n):
    var gram = tokens[i]
    for j in 1..<n:
      gram.add(" ")
      gram.add(tokens[i+j])
    result.add(gram)

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
