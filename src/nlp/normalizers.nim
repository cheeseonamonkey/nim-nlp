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
  ("À", "A"), ("Á", "A"), ("Â", "A"), ("Ã", "A"), ("Ä", "A"), ("Å", "A"),
  ("Ç", "C"),
  ("È", "E"), ("É", "E"), ("Ê", "E"), ("Ë", "E"),
  ("Ì", "I"), ("Í", "I"), ("Î", "I"), ("Ï", "I"),
  ("Ñ", "N"),
  ("Ò", "O"), ("Ó", "O"), ("Ô", "O"), ("Õ", "O"), ("Ö", "O"),
  ("Ù", "U"), ("Ú", "U"), ("Û", "U"), ("Ü", "U"),
  ("Ý", "Y"),
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

  HtmlStripper* = object
    ## Strips HTML/XML tags (removes "<...>" spans)

  UrlReplacer* = object
    ## Replaces http/https/www URLs with a placeholder
    replacement*: string  # defaults to "<url>" when empty at call site

  EmailReplacer* = object
    ## Replaces email-like tokens with a placeholder
    replacement*: string  # defaults to "<email>" when empty at call site

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
  ## Case is preserved: "À" → "A", "à" → "a".
  if text.len == 0: return ""
  result = newStringOfCap(text.len)
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
  result = newStringOfCap(text.len)
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
  result = newStringOfCap(text.len)
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
  result = newStringOfCap(text.len)
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

proc isEmailChar(ch: char): bool =
  ch.isAlphaNumeric or ch in {'_', '-', '.', '+', '%'}

proc emailSpanAt(text: string, pos: int): int =
  ## If an email-like span starts at `pos`, return its length, else 0.
  ## Local part: one or more isEmailChar; then '@'; then domain with a dot.
  var j = pos
  while j < text.len and isEmailChar(text[j]):
    inc j
  if j == pos or j >= text.len or text[j] != '@': return 0
  if j + 1 >= text.len or not isEmailChar(text[j + 1]): return 0
  var k = j + 1
  var hasDot = false
  while k < text.len and isEmailChar(text[k]):
    if text[k] == '.': hasDot = true
    inc k
  if not hasDot: return 0
  # Trim trailing dots (e.g. "a@b.com." → "a@b.com").
  while k > j + 1 and text[k - 1] == '.':
    dec k
  if text[k - 1] == '.': return 0
  k - pos

proc isUrlStart(text: string, i: int): bool =
  if i + 6 < text.len and text[i..i+6] == "http://": return true
  if i + 7 < text.len and text[i..i+7] == "https://": return true
  if i + 3 < text.len and text[i..i+3] == "www.": return true
  false

proc urlLenFrom(text: string, i: int): int =
  var j = i
  while j < text.len and text[j] notin Whitespace:
    inc j
  # Trim trailing punctuation unlikely to be part of the URL. ',' is NOT
  # trimmed — it is kept as a list separator (e.g. "visit www.example.com, ok?").
  while j > i and text[j - 1] in {'.', '!', '?', ';', ':', ')', ']', '}', '"', '\''}:
    dec j
  j - i

proc isHtmlTag(text: string, i: int): bool =
  ## Heuristic: "<" starts a tag only if next char begins a tag name
  ## ('/', '!', '?', letter) — not a bare comparison operator.
  if i + 1 >= text.len: return false
  let c = text[i + 1]
  c == '/' or c == '!' or c == '?' or c.isAlphaAscii

proc normalize*(n: HtmlStripper, text: string): string =
  ## Strip HTML/XML tags. Only spans that look like tags are removed
  ## ("<" followed by '/', '!', '?', or a letter, then up to next '>').
  ## Bare "a < b" / unclosed '<' is left as literal text.
  if text.len == 0: return ""
  result = newStringOfCap(text.len)
  var i = 0
  while i < text.len:
    if text[i] == '<' and isHtmlTag(text, i):
      let closeIdx = text.find('>', i + 1)
      if closeIdx == -1:
        result.add(text[i])
        inc i
      else:
        i = closeIdx + 1
    else:
      result.add(text[i])
      inc i

proc normalize*(n: UrlReplacer, text: string): string =
  ## Replace URL spans with `n.replacement` (default "<url>").
  if text.len == 0: return ""
  let repl = if n.replacement.len > 0: n.replacement else: "<url>"
  result = newStringOfCap(text.len + repl.len)
  var i = 0
  while i < text.len:
    if isUrlStart(text, i):
      let span = urlLenFrom(text, i)
      if span > 0:
        result.add(repl)
        i += span
        continue
    result.add(text[i])
    inc i

proc normalize*(n: EmailReplacer, text: string): string =
  ## Replace email-like spans with `n.replacement` (default "<email>").
  ## Scans forward from `i`; when an email starts exactly at `i`, it is
  ## replaced as a whole, so already-emitted characters never need backtracking.
  if text.len == 0: return ""
  let repl = if n.replacement.len > 0: n.replacement else: "<email>"
  result = newStringOfCap(text.len + repl.len)
  var i = 0
  while i < text.len:
    let eLen = emailSpanAt(text, i)
    if eLen > 0:
      result.add(repl)
      i += eLen
      continue
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

proc normalizeStripHtml*(text: string): string =
  HtmlStripper().normalize(text)

proc normalizeReplaceUrls*(text: string, replacement = "<url>"): string =
  UrlReplacer(replacement: replacement).normalize(text)

proc normalizeReplaceEmails*(text: string, replacement = "<email>"): string =
  EmailReplacer(replacement: replacement).normalize(text)
