## Tokenizer tests for the nlp library.

import ../src/nlp
import std/[tables, sets]

proc testTokenizers() =
  echo "=== Tokenizers ==="

  # Word tokenizer - basic
  let t1 = wordTokenize("the quick brown fox")
  assert t1 == @["the", "quick", "brown", "fox"], "wordTokenize basic"

  # Word tokenizer - minLength
  let t2 = wordTokenize("the quick brown fox", minLength = 4)
  assert t2 == @["quick", "brown"], "wordTokenize minLength"

  # Word tokenizer - case preservation
  let t3 = wordTokenize("Hello World", lowercase = false)
  assert t3 == @["Hello", "World"], "wordTokenize case"

  # Word tokenizer - empty
  assert wordTokenize("") == @[], "wordTokenize empty"

  # Word tokenizer - only punctuation
  assert wordTokenize("!@#$%^&*()") == @[], "wordTokenize only punctuation"

  # Word tokenizer - single char
  assert wordTokenize("a") == @["a"], "wordTokenize single char"

  # Word tokenizer - unicode
  let t4 = wordTokenize("café résumé")
  assert "caf" in t4, "wordTokenize unicode handles accents"
  assert "r" in t4, "wordTokenize unicode partial"

  # Whitespace tokenizer
  let t5 = whitespaceTokenize("hello   world\tfoo")
  assert t5 == @["hello", "world", "foo"], "whitespaceTokenize"

  # Whitespace tokenizer - empty
  assert whitespaceTokenize("") == @[], "whitespaceTokenize empty"

  # Whitespace tokenizer - only whitespace
  assert whitespaceTokenize("   \t\n  ") == @[], "whitespaceTokenize only whitespace"

  # Whitespace tokenizer - minLength
  let t5b = whitespaceTokenize("a bb ccc dddd", minLength = 2)
  assert t5b == @["bb", "ccc", "dddd"], "whitespaceTokenize minLength"

  # N-gram tokenizer
  let t6 = ngramTokenize("hello", 2)
  assert t6 == @["he", "el", "ll", "lo"], "ngramTokenize"

  # N-gram tokenizer - edge cases
  assert ngramTokenize("", 2) == @[], "ngramTokenize empty"
  assert ngramTokenize("a", 2) == @[], "ngramTokenize short"
  assert ngramTokenize("hello", 0) == @[], "ngramTokenize n=0"
  assert ngramTokenize("hello", -1) == @[], "ngramTokenize n<0"
  assert ngramTokenize("hello", 10) == @[], "ngramTokenize n>len"
  assert ngramTokenize("éab", 2) == @["éa", "ab"], "ngramTokenize unicode characters"

  # Sentence tokenizer
  let t7 = splitSentences("Hello world. How are you? I'm fine!")
  assert t7 == @["Hello world.", "How are you?", "I'm fine!"], "splitSentences"

  # Sentence tokenizer - empty
  assert splitSentences("") == @[], "splitSentences empty"

  # Sentence tokenizer - no terminator
  assert splitSentences("no period here") == @["no period here"], "splitSentences no terminator"

  # Sentence tokenizer - multiple terminators
  let t7b = splitSentences("Wow! Really? Yes.")
  assert t7b == @["Wow!", "Really?", "Yes."], "splitSentences multiple"

  # Sentence tokenizer - don't discard empty
  let t7c = splitSentences("Hello. ", discardEmpty = false)
  assert t7c == @["Hello.", ""], "splitSentences keep empty trailing content"
  assert splitSentences("", discardEmpty = false) == @[""],
    "splitSentences keep empty input"
  assert splitSentences("  Hello.  Next!  ", discardEmpty = false) ==
    @["Hello.", "Next!", ""], "splitSentences strip when keeping empty content"

  # Pattern tokenizer
  let t8 = PatternTokenizer(pattern: r"\w+", lowercase: true, minLength: 1).tokenize("Hello, World!")
  assert t8 == @["hello", "world"], "patternTokenizer"

  # Pattern tokenizer - empty
  assert PatternTokenizer(pattern: r"\w+").tokenize("") == @[], "patternTokenizer empty"

  # Pattern tokenizer - no matches
  assert PatternTokenizer(pattern: r"\d+").tokenize("hello world") == @[], "patternTokenizer no match"

  # Pattern tokenizer - digits
  let t8a = PatternTokenizer(pattern: r"\d+").tokenize("abc 123 def 456")
  assert t8a == @["123", "456"], "patternTokenizer digits"

  # Pattern tokenizer - character class
  let t8b = PatternTokenizer(pattern: "[aeiou]").tokenize("hello")
  assert t8b == @["e", "o"], "patternTokenizer char class"

  # Word n-grams
  let t9 = wordTokenize("the quick brown fox").ngrams(2)
  assert t9 == @["the quick", "quick brown", "brown fox"], "word ngrams"

  # Word n-grams - edge cases
  assert wordTokenize("").ngrams(2) == @[], "word ngrams empty"
  assert wordTokenize("hello").ngrams(2) == @[], "word ngrams too short"
  assert wordTokenize("the quick").ngrams(0) == @[], "word ngrams n=0"
  assert wordTokenize("the quick").ngrams(-1) == @[], "word ngrams n<0"
  assert wordTokenize("the quick").ngrams(1) == @["the", "quick"], "word ngrams n=1"

  echo "  ALL PASS"

when isMainModule:
  testTokenizers()
