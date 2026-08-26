## Text normalizers for the nlp library.

import std/[unicode, strutils, tables]

# ============================================================
# Lookup tables
# ============================================================

const accentTable: Table[string, string] = [
  ("à", "a"), ("á", "a"), ("â", "a"), ("ã", "a"), ("ä", "a"), ("å", "a"),
  ("ç", "c"),
  ("è", "e"), ("é", "e"), ("ê", "e"), ("ë", "e"),
  ("ì", "i"), ("í", "i"), ("î", "i"), ("ï", "i"),
  ("ñ", "n"),
  ("ò", "o"), ("ó", "o"), ("ô", "o"), ("õ", "o"), ("ö", "o"),
  ("ù", "u"), ("ú", "u"), ("û", "u"), ("ü", "u"),
  ("ý", "y"), ("ÿ", "y"),
  ("À", "a"), ("Á", "a"), ("Â", "a"), ("Ã", "a"), ("Ä", "a"), ("Å", "a"),
  ("Ç", "c"),
  ("È", "e"), ("É", "e"), ("Ê", "e"), ("Ë", "e"),
  ("Ì", "i"), ("Í", "i"), ("Î", "i"), ("Ï", "i"),
  ("Ñ", "n"),
  ("Ò", "o"), ("Ó", "o"), ("Ô", "o"), ("Õ", "o"), ("Ö", "o"),
  ("Ù", "u"), ("Ú", "u"), ("Û", "u"), ("Ü", "u"),
  ("Ý", "y"),
].toTable

# All ASCII and common Unicode punctuation
const punctuationChars: set[char] = {
  ',', '.', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}',
  '"', '\'', '`', '~', '@', '#', '$', '%', '^', '&', '*', '_', '+',
  '=', '|', '\\', '/', '<', '>'
}

# ============================================================
# Normalizer types
# ============================================================

type
  Lowercaser* = object
    ## Converts text to lowercase

  AccentStripper* = object
    ## Strips accents from Unicode characters (café → cafe)

  WhitespaceNormalizer* = object
    ## Collapses multiple whitespace into single spaces

  PunctuationStripper* = object
    ## Removes common punctuation characters

  ChainedNormalizer* = object
    ## A composition of two normalizers
    first*: proc(text: string): string
    second*: proc(text: string): string

  CustomNormalizer* = object
    ## Wraps a custom normalization function
    fn*: proc(text: string): string

  NumberNormalizer* = object
    ## Replaces digits with <num> or removes them
    replaceWith*: string  # if empty, digits are removed

# ============================================================
# Implementations
# ============================================================

proc normalize*(n: Lowercaser, text: string): string =
  ## Convert text to lowercase (returns "" for empty input).
  if text.len == 0: return ""
  result = text.toLowerAscii()

proc normalize*(n: AccentStripper, text: string): string =
  ## Strip accents using a lookup table for common accented characters.
  ## Also removes combining diacritical marks (U+0300–U+036F).
  if text.len == 0: return ""
  for r in text.runes:
    let s = $r
    if accentTable.hasKey(s):
      result.add(accentTable[s])
    elif r.int >= 0x0300 and r.int <= 0x036F:
      continue  # Skip combining diacritics
    else:
      result.add(s)

proc normalize*(n: WhitespaceNormalizer, text: string): string =
  ## Collapse multiple whitespace into single spaces, trim edges.
  if text.len == 0: return ""
  var prevSpace = false
  for ch in text:
    if ch in Whitespace:
      if not prevSpace:
        result.add(' ')
        prevSpace = true
    else:
      result.add(ch)
      prevSpace = false
  result = result.strip()

proc normalize*(n: PunctuationStripper, text: string): string =
  ## Remove common punctuation characters.
  if text.len == 0: return ""
  for ch in text:
    if ch in punctuationChars:
      continue
    result.add(ch)

proc normalize*(n: ChainedNormalizer, text: string): string =
  ## Apply two normalizers in sequence.
  n.second(n.first(text))

proc normalize*(n: CustomNormalizer, text: string): string =
  ## Apply a custom normalization function.
  if text.len == 0: return ""
  result = n.fn(text)

proc normalize*(n: NumberNormalizer, text: string): string =
  ## Replace or remove digits.
  if text.len == 0: return ""
  var i = 0
  var inNumber = false
  while i < text.len:
    if text[i].isDigit:
      if not inNumber:
        if n.replaceWith.len > 0:
          result.add(n.replaceWith)
        inNumber = true
    else:
      inNumber = false
      result.add(text[i])
    inc i

# ============================================================
# Chaining
# ============================================================

proc chain*(a: proc(text: string): string, b: proc(text: string): string): proc(text: string): string =
  ## Chain two normalizer functions.
  result = proc(text: string): string = b(a(text))

# ============================================================
# Convenience functions
# ============================================================

proc normalizeLowercase*(text: string): string =
  Lowercaser().normalize(text)

proc normalizeStripAccents*(text: string): string =
  AccentStripper().normalize(text)

proc normalizeWhitespace*(text: string): string =
  WhitespaceNormalizer().normalize(text)

proc normalizeStripPunctuation*(text: string): string =
  PunctuationStripper().normalize(text)

proc normalizeRemoveNumbers*(text: string, replacement = ""): string =
  NumberNormalizer(replaceWith: replacement).normalize(text)
