## Analysis tests for the nlp library.

import ../src/nlp
import std/[tables, sets, sequtils]

proc testAnalysis() =
  echo "\n=== Analysis ==="

  # Term frequencies
  let tokens = wordTokenize("the cat sat on the mat the cat")
  let freqs = termFrequencies(tokens)
  assert freqs.getOrDefault("the", 0) == 3, "term freq"
  assert freqs.getOrDefault("cat", 0) == 2, "term freq"

  # Document frequency
  let docs = @[
    wordTokenize("the cat sat"),
    wordTokenize("the dog sat"),
    wordTokenize("the bird flew")
  ]
  let df = documentFrequency(docs)
  assert df.getOrDefault("the", 0) == 3, "doc freq"
  assert df.getOrDefault("cat", 0) == 1, "doc freq"

  # Empty doc in corpus
  let df2 = documentFrequency(@[wordTokenize("hello"), @[], wordTokenize("hello")])
  assert df2.getOrDefault("hello", 0) == 2, "doc freq with empty doc"

  # Similarity
  let jacc = jaccardSimilarity(
    wordTokenize("the cat sat"),
    wordTokenize("the dog sat")
  )
  assert jacc > 0.0 and jacc < 1.0, "jaccard"

  # Empty similarity
  assert jaccardSimilarity(@[], @["hello"]) == 0.0, "jaccard empty"
  assert cosineSimilarityBags(@[], @["hello"]) == 0.0, "cosine empty"
  assert overlapCoefficient(@[], @["hello"]) == 0.0, "overlap empty"

  # Cosine similarity
  let cos = cosineSimilarityBags(
    wordTokenize("the cat sat"),
    wordTokenize("the dog sat")
  )
  assert cos > 0.0 and cos <= 1.0, "cosine bags"

  # Text stats
  let stats = textStats(wordTokenize("the quick brown fox"))
  assert stats.numTokens == 4, "stats numTokens"
  assert stats.numUniqueTerms == 4, "stats unique"

  # Empty stats
  let emptyStats = textStats(@[])
  assert emptyStats.numTokens == 0, "stats empty numTokens"
  assert emptyStats.numUniqueTerms == 0, "stats empty unique"
  assert emptyStats.avgTokenLength == 0.0, "stats empty avg"

  # Stats with repeated tokens
  let stats2 = wordTokenize("the cat sat on the mat").textStats()
  assert stats2.numTokens == 6, "stats2 numTokens"
  assert stats2.numUniqueTerms == 5, "stats2 unique"
  assert stats2.hapaxLegomena >= 0, "stats2 hapax"

  # Keywords
  let keywords = extractKeywords(docs, topN = 3)
  assert keywords.len > 0, "keywords"

  # Keywords - empty
  assert extractKeywords(@[], topN = 3).len == 0, "keywords empty"
  assert extractKeywords(docs, topN = 0).len == 0, "keywords topN=0"
  assert extractKeywords(docs, topN = -1).len == 0, "keywords topN<0"
  assert extractKeywords(@[@[], @["term"]], topN = 3).len == 1,
    "keywords empty document"

  # Keywords - frequency method
  let keywordsFreq = extractKeywords(docs, topN = 3, methodName = "freq")
  assert keywordsFreq.len > 0, "keywords freq method"

  # Keywords - unknown method
  assert extractKeywords(docs, topN = 3, methodName = "unknown").len == 0, "keywords unknown method"

  # Sorted terms
  let sorted = sortedTermsByFrequency(freqs)
  assert sorted.len > 0, "sorted terms"
  assert sorted[0][1] >= sorted[^1][1], "sorted order"

  # Sorted terms - empty input
  assert sortedTermsByFrequency(initTable[string, int]()).len == 0,
    "sorted terms empty"

  # Sorted terms - equal frequency falls back to lexicographic ascending
  let tied = sortedTermsByFrequency(
    {"b": 1, "a": 1, "c": 1, "d": 2}.toTable
  )
  assert tied == @[("d", 2), ("a", 1), ("b", 1), ("c", 1)],
    "sorted terms tie-break lexicographic"

  # Unique terms across documents, including duplicates and empty docs
  let uniq = uniqueTerms(@[wordTokenize("a b"), wordTokenize("b c"), @[]])
  assert uniq == toHashSet(["a", "b", "c"]), "unique terms"
  assert uniqueTerms(@[]).len == 0, "unique terms no documents"
  assert uniqueTerms(@[newSeq[string](), newSeq[string]()]).len == 0,
    "unique terms empty documents"

  # Overlap coefficient - empty on either side and unequal set sizes
  assert overlapCoefficient(@["a"], @[]) == 0.0, "overlap empty right"
  assert overlapCoefficient(@["a", "b"], @["b", "c", "d"]) == 0.5,
    "overlap unequal sets"
  assert overlapCoefficient(@["a", "a", "b"], @["a", "b"]) == 1.0,
    "overlap duplicate tokens"

  # Keywords - topN larger than the vocabulary returns every unique term
  let docs3 = @[
    wordTokenize("the cat"), wordTokenize("the dog"), wordTokenize("the bird")
  ]
  let kwAll = extractKeywords(docs3, topN = 100)
  assert kwAll.len == uniqueTerms(docs3).len, "keywords topN beyond vocabulary"
  assert kwAll.anyIt(it[0] == "the"), "keywords include corpus-wide term"

  # Keywords - a term present in every document keeps positive weight
  let kwCommon = extractKeywords(
    @[wordTokenize("apple common"), wordTokenize("banana common")], topN = 5
  )
  assert kwCommon.len > 0 and kwCommon[0][0] == "common",
    "keywords rank term present in every document"

  echo "  ALL PASS"

when isMainModule:
  testAnalysis()
