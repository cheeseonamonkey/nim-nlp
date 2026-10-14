## Stopword tests for the nlp library.

import ../src/nlp
import std/[tables, sets]

proc testStopwords() =
  echo "\n=== Stopwords ==="

  # English
  let tokens = wordTokenize("the quick brown fox jumps over the lazy dog")
  let filtered = tokens.removeStopWords("english")
  assert "the" notin filtered, "stopword removed"
  assert "quick" in filtered, "content word kept"

  # Case insensitive
  let filtered2 = wordTokenize("The THE the quick").removeStopWords("english")
  assert "quick" in filtered2, "stopword case insensitive content"
  assert "The" notin filtered2, "stopword case insensitive removed"

  # isStopWord
  assert isStopWord("the", "english") == true, "isStopWord"
  assert isStopWord("hello", "english") == false, "isStopWord negative"
  assert isStopWord("", "english") == false, "isStopWord empty"
  assert isStopWord("THE", "english") == true, "isStopWord uppercase"
  assert isStopWord("À", "french") == true, "isStopWord unicode uppercase"
  assert removeStopWords(@["À", "chat"], "french") == @["chat"],
    "removeStopWords unicode case"

  # Unknown language
  assert isStopWord("the", "klingon") == false, "isStopWord unknown lang"
  assert @["hello", "world"].removeStopWords("klingon") == @["hello", "world"], "removeStopWords unknown lang"

  # Spanish
  let spanishTokens = wordTokenize("el gato come pescado")
  let spanishFiltered = spanishTokens.removeStopWords("spanish")
  assert "el" notin spanishFiltered, "spanish stopword"
  assert "gato" in spanishFiltered, "spanish content"

  # French
  let frenchTokens = wordTokenize("le chat mange du poisson")
  let frenchFiltered = frenchTokens.removeStopWords("french")
  assert "le" notin frenchFiltered, "french stopword"
  assert "chat" in frenchFiltered, "french content"

  # German
  let germanTokens = wordTokenize("die katze isst fisch")
  let germanFiltered = germanTokens.removeStopWords("german")
  assert "die" notin germanFiltered, "german stopword"
  assert "katze" in germanFiltered, "german content"

  # Custom stopwords
  let custom = toHashSet(["fox", "dog"])
  let filtered3 = tokens.removeStopWords(custom)
  assert "fox" notin filtered3, "custom stopword"
  assert "quick" in filtered3, "custom content kept"

  # Empty input
  assert newSeq[string]().removeStopWords("english") == @[], "removeStopWords empty"
  assert newSeq[string]().removeStopWords(custom) == @[], "removeStopWords empty custom"

  echo "  ALL PASS"

proc testStopwordsCustom() =
  echo "\n=== Stopwords (custom set) ==="

  # Custom stopwords are matched case-insensitively and preserve original casing.
  let customCase = toHashSet(["fox", "dog"])
  let filteredCase = @["Fox", "Dog", "Wolf"].removeStopWords(customCase)
  assert filteredCase == @["Wolf"],
    "custom stopwords case-insensitive and preserve casing"

  echo "  ALL PASS"

when isMainModule:
  testStopwords()
  testStopwordsCustom()
