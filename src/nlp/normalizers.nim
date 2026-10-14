## Text normalizers for the nlp library.

import std/[unicode, strutils, tables]
import private/lexicon

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
    ## Strips accents from common accented characters (café → cafe) using a
    ## fixed lookup table, and drops combining diacritical marks
    ## (U+0300–U+036F). It does not decompose ligatures or expand characters
    ## absent from the table: `œ`, `æ`, `ß`, and `ø` are left unchanged.

  WhitespaceNormalizer* = object
    ## Collapses multiple whitespace into single spaces

  PunctuationStripper* = object
    ## Removes a fixed ASCII punctuation set. Non-ASCII punctuation (em dash,
    ## curly quotes, ellipsis, …) is not recognized and is kept.

  ChainedNormalizer* = object
    ## A composition of two normalizers. There is no constructor for this type;
    ## use `chain <#chain>`_ to build a composed normalization proc instead.
    first*: proc(text: string): string
    second*: proc(text: string): string

  CustomNormalizer* = object
    ## Wraps a custom normalization function
    fn*: proc(text: string): string

  NumberNormalizer* = object
    ## Replaces digits with <num> or removes them
    replaceWith*: string  # if empty, digits are removed

  HtmlStripper* = object
    ## Strips HTML/XML tags with a bounded scanner (not a parser). A `>` inside
    ## a quoted attribute value is treated as text until the matching quote
    ## closes; a `<` that does not begin a tag name and an unterminated tag are
    ## left as literal text.

  UrlReplacer* = object
    ## Replaces URL spans with a placeholder. Only `http://`, `https://`, and
    ## `www.` starts are recognized. Trailing punctuation is trimmed except
    ## `,`, which is kept as a list separator.
    replacement*: string  # defaults to "<url>" when empty at call site

  EmailReplacer* = object
    ## Replaces email-like tokens with a placeholder
    replacement*: string  # defaults to "<email>" when empty at call site

  ContractionExpander* = object
    ## Expands English contractions whole-token by whole-token using the
    ## contraction table in ``private/lexicon`` ("don't" -> "do not").
    ## Only full-token table hits expand, so possessives ("the dog's bone")
    ## and "o'clock" pass through untouched. Matching is case-insensitive
    ## and accepts both the ASCII apostrophe and U+2019; when the matched
    ## token starts with an uppercase letter the expansion's first letter
    ## is capitalized ("Don't" -> "Do not"). `'s` always expands to "is"
    ## ("it's" -> "it is"; the is/has ambiguity is resolved in favour
    ## of "is") and `'d` always expands to "would".

  Normalizer* = concept n
    ## Any normalizer type: anything with `normalize(n, text) -> string`.
    ## All object normalizers above conform; so does `TextPipeline`.
    normalize(n, string) is string

  TextPipeline* = object
    ## Ordered, composable normalization pipeline. Steps run in insertion
    ## order via `addStep <#addStep>`_ and `normalize <#normalize>`_.
    ## A pipeline is itself a `Normalizer`, so pipelines nest.
    steps*: seq[proc(text: string): string]

# ============================================================
# Implementations
# ============================================================

proc normalize*(n: Lowercaser, text: string): string =
  ## Convert text to lowercase (returns "" for empty input).
  if text.len == 0: return ""
  result = text.toLower()

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
    if not text[i].isDigit:
      inNumber = false
      result.add(text[i])
      inc i
      continue
    if not inNumber:
      if n.replaceWith.len > 0:
        result.add(n.replaceWith)
      inNumber = true
    inc i

proc isEmailChar(ch: char): bool =
  ch.isAlphaNumeric or ch in {'_', '-', '.', '+', '%'}

proc scanEmailLocalEnd(text: string, pos: int): int =
  ## End index of the `isEmailChar` run starting at `pos`.
  result = pos
  while result < text.len and isEmailChar(text[result]):
    inc result

proc scanEmailDomainEnd(text: string, atSign: int): int =
  ## End index of the domain starting just after the `@` at `atSign`, with
  ## trailing dots trimmed; -1 when nothing valid follows `@`.
  if atSign + 1 >= text.len or not isEmailChar(text[atSign + 1]):
    return -1
  result = atSign + 1
  while result < text.len and isEmailChar(text[result]):
    inc result
  # Trim trailing dots (e.g. "a@b.com." → "a@b.com").
  while result > atSign + 1 and text[result - 1] == '.':
    dec result

proc emailDomainHasDot(text: string, atSign, domainEnd: int): bool =
  ## True when the trimmed domain holds a dot separating a non-empty host
  ## part from a non-empty TLD (checked *after* trailing dots are trimmed,
  ## so "a@." is not an email). A dot at index d qualifies when d > atSign+1
  ## (host part not empty) and d < domainEnd-1 (TLD part not empty).
  var d = atSign + 2
  while d < domainEnd - 1:
    if text[d] == '.':
      return true
    inc d
  false

proc emailSpanAt(text: string, pos: int): int =
  ## If an email-like span starts at `pos`, return its length, else 0.
  ## Local part: one or more isEmailChar; then '@'; then domain that must
  ## contain a dot separating a non-empty host part from a non-empty TLD
  ## (checked *after* trailing dots are trimmed, so "a@." is not an email).
  let localEnd = scanEmailLocalEnd(text, pos)
  if localEnd == pos or localEnd >= text.len or text[localEnd] != '@':
    return 0
  let domainEnd = scanEmailDomainEnd(text, localEnd)
  if domainEnd == -1:
    return 0
  if not emailDomainHasDot(text, localEnd, domainEnd):
    return 0
  domainEnd - pos

proc isUrlStart(text: string, i: int): bool =
  if i + 6 < text.len and text[i..i+6] == "http://": return true
  if i + 7 < text.len and text[i..i+7] == "https://": return true
  if i + 3 < text.len and text[i..i+3] == "www.": return true
  false

proc trimUrlTail(text: string, start, stop: int): int =
  ## Pull `stop` back over trailing punctuation unlikely to be part of the
  ## URL. ',' is NOT trimmed — it is kept as a list separator
  ## (e.g. "visit www.example.com, ok?").
  result = stop
  while result > start and text[result - 1] in {'.', '!', '?', ';', ':', ')', ']', '}', '"', '\''}:
    dec result

proc urlLenFrom(text: string, i: int): int =
  var j = i
  while j < text.len and text[j] notin Whitespace:
    inc j
  trimUrlTail(text, i, j) - i

proc urlSpanAt(text: string, i: int): int =
  ## Length of a URL starting exactly at `i`, or 0 if none starts there.
  if isUrlStart(text, i): urlLenFrom(text, i) else: 0

proc replaceSpans(
  text, repl: string, spanAt: proc(text: string, i: int): int {.nimcall.}
): string =
  ## Walk `text`; whenever `spanAt` reports a non-zero span at the current
  ## position, emit `repl` and skip the span, else emit the character.
  if text.len == 0: return ""
  result = newStringOfCap(text.len + repl.len)
  var i = 0
  while i < text.len:
    let span = spanAt(text, i)
    if span <= 0:
      result.add(text[i])
      inc i
      continue
    result.add(repl)
    i += span

proc isHtmlTag(text: string, i: int): bool =
  ## Heuristic: "<" starts a tag only if next char begins a tag name
  ## ('/', '!', '?', letter) — not a bare comparison operator.
  if i + 1 >= text.len: return false
  let c = text[i + 1]
  c == '/' or c == '!' or c == '?' or c.isAlphaAscii

proc tagEnd(text: string, openIdx: int): int =
  ## Index of the '>' that closes the tag opened at `openIdx`, or -1 if the
  ## tag never closes. A '>' inside a quoted attribute value is literal
  ## text until the matching quote closes (both " and ').
  var quote = '\0'
  var i = openIdx + 1
  while i < text.len:
    let ch = text[i]
    if quote != '\0':
      if ch == quote: quote = '\0'
    elif ch == '"' or ch == '\'':
      quote = ch
    elif ch == '>':
      return i
    inc i
  -1

proc normalize*(n: HtmlStripper, text: string): string =
  ## Strip HTML/XML tags. Only spans that look like tags are removed
  ## ("<" followed by '/', '!', '?', or a letter, then up to the closing
  ## '>', where a '>' inside a quoted attribute value is literal text).
  ## Bare "a < b" / unclosed '<' is left as literal text.
  if text.len == 0: return ""
  result = newStringOfCap(text.len)
  var i = 0
  while i < text.len:
    if text[i] != '<' or not isHtmlTag(text, i):
      result.add(text[i])
      inc i
      continue
    let closeIdx = tagEnd(text, i)
    if closeIdx == -1:
      result.add(text[i])
      inc i
      continue
    i = closeIdx + 1

proc normalize*(n: UrlReplacer, text: string): string =
  ## Replace URL spans with `n.replacement` (default "<url>").
  let repl = if n.replacement.len > 0: n.replacement else: "<url>"
  replaceSpans(text, repl, urlSpanAt)

proc normalize*(n: EmailReplacer, text: string): string =
  ## Replace email-like spans with `n.replacement` (default "<email>").
  ## Scans forward from `i`; when an email starts exactly at `i`, it is
  ## replaced as a whole, so already-emitted characters never need backtracking.
  let repl = if n.replacement.len > 0: n.replacement else: "<email>"
  replaceSpans(text, repl, emailSpanAt)

proc isCurlyApostropheAt(text: string, i: int): bool =
  ## True when a U+2019 apostrophe (UTF-8 bytes E2 80 99) starts at `i`.
  text[i] == '\xE2' and i + 2 < text.len and
    text[i + 1] == '\x80' and text[i + 2] == '\x99'

proc isTokenStartAt(text: string, i: int): bool =
  ## True when position `i` can begin a word token for contraction lookup.
  text[i].isAlphaNumeric or text[i] == '\'' or isCurlyApostropheAt(text, i)

proc scanTokenKey(text: string, start: int): tuple[key: string, next: int] =
  ## Scan the token at `start`; return the lowercase lookup key (curly
  ## apostrophes folded to ASCII) and the index one past the token end.
  var j = start
  var key = newStringOfCap(16)
  while j < text.len:
    let ch = text[j]
    if ch.isAlphaNumeric:
      key.add(ch.toLowerAscii)
      inc j
    elif ch == '\'':
      key.add('\'')
      inc j
    elif isCurlyApostropheAt(text, j):
      key.add('\'')
      j += 3
    else:
      break
  (key, j)

proc applyExpansionCaps(expansion: string, first: char): string =
  ## Capitalize the expansion's first letter when the matched token started
  ## with an uppercase letter ("Don't" -> "Do not").
  if first.isUpperAscii:
    result = expansion
    result[0] = result[0].toUpperAscii
  else:
    result = expansion

let contractionTable* = builtinContractions.toTable
  ## Lowercase contraction form (ASCII apostrophe) to lowercase expansion.

proc normalize*(n: ContractionExpander, text: string): string =
  ## Expand English contractions ("don't" -> "do not"). Text is scanned
  ## for word tokens (letters/digits plus `'` or U+2019); each token is
  ## looked up whole in the contraction table, which makes matching
  ## inherently longest-match and word-boundary delimited -- possessives and
  ## "o'clock" have no table entries and are left as-is. Returns "" for
  ## empty input.
  if text.len == 0: return ""
  result = newStringOfCap(text.len + 16)
  var i = 0
  while i < text.len:
    if not isTokenStartAt(text, i):
      result.add(text[i])
      inc i
      continue
    let (key, next) = scanTokenKey(text, i)
    if contractionTable.hasKey(key):
      result.add(applyExpansionCaps(contractionTable[key], text[i]))
    else:
      result.add(text[i ..< next])
    i = next

# ============================================================
# Chaining
# ============================================================

proc chain*(a: proc(text: string): string, b: proc(text: string): string): proc(text: string): string =
  ## Chain two normalizer functions. This is the intended composition API;
  ## the `ChainedNormalizer` type has no constructor.
  result = proc(text: string): string = b(a(text))

# ============================================================
# Composable pipelines
# ============================================================

proc addStep*(p: var TextPipeline, step: proc(text: string): string) =
  ## Append a raw proc step to the pipeline.
  p.steps.add(step)

proc normalize*(p: TextPipeline, text: string): string =
  ## Run all steps in order; an empty pipeline returns `text` unchanged.
  ## This also makes `TextPipeline` itself a `Normalizer`, so pipelines nest
  ## inside other pipelines via `addStep`.
  result = text
  for s in p.steps:
    result = s(result)

proc addStep*[N: Normalizer](p: var TextPipeline, n: N) =
  ## Append any `Normalizer` (each object normalizer type, or another
  ## `TextPipeline`) by wrapping its `normalize` as a pipeline step.
  let nn = n
  p.steps.add(proc(t: string): string = normalize(nn, t))

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

proc normalizeExpandContractions*(text: string): string =
  ContractionExpander().normalize(text)
