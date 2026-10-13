## Normalizer tests for the nlp library.

import ../src/nlp
import std/[tables, sets]

proc testNormalizers() =
  echo "\n=== Normalizers ==="

  # Lowercase
  assert "Hello".normalizeLowercase() == "hello", "lowercase"
  assert "".normalizeLowercase() == "", "lowercase empty"
  assert "123".normalizeLowercase() == "123", "lowercase numbers"
  assert "!@#".normalizeLowercase() == "!@#", "lowercase punctuation"

  # Accent strip
  let accentResult = "café naïve résumé".normalizeStripAccents()
  assert accentResult == "cafe naive resume", "accents: got " & accentResult
  assert "".normalizeStripAccents() == "", "accents empty"
  assert "abc".normalizeStripAccents() == "abc", "accents no accents"

  # Whitespace
  assert "  hello   world  ".normalizeWhitespace() == "hello world", "whitespace"
  assert "".normalizeWhitespace() == "", "whitespace empty"
  assert "   ".normalizeWhitespace() == "", "whitespace only spaces"
  assert "hello".normalizeWhitespace() == "hello", "whitespace no extra"

  # Punctuation
  assert "Hello, world!".normalizeStripPunctuation() == "Hello world", "punctuation"
  assert "".normalizeStripPunctuation() == "", "punctuation empty"
  assert "abc".normalizeStripPunctuation() == "abc", "punctuation no punct"
  assert "!@#$%^&*()".normalizeStripPunctuation() == "", "punctuation only punct"

  # Number normalizer
  assert normalizeRemoveNumbers("abc123def456") == "abcdef", "numbers remove"
  assert normalizeRemoveNumbers("abc123def456", "NUM") == "abcNUMdefNUM", "numbers replace"
  assert normalizeRemoveNumbers("") == "", "numbers empty"
  assert normalizeRemoveNumbers("abc") == "abc", "numbers no digits"
  assert normalizeRemoveNumbers("123") == "", "numbers only digits"

  # Chained normalizer
  let norm = chain(proc(text: string): string = Lowercaser().normalize(text), proc(text: string): string = PunctuationStripper().normalize(text))
  assert norm("Hello, World!") == "hello world", "chained"

  # Chained normalizer - empty
  assert norm("") == "", "chained empty"

  echo "  ALL PASS"

when isMainModule:
  testNormalizers()
