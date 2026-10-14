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

  echo "\n=== N-gram analysis ==="

  # n-gram frequencies
  let ngFreq = ngramFrequencies(@["the", "cat", "the", "cat", "sat"], 2)
  assert ngFreq.getOrDefault(@["the", "cat"], 0) == 2, "ngram freq the-cat"
  assert ngFreq.getOrDefault(@["cat", "the"], 0) == 1, "ngram freq cat-the"
  assert ngFreq.getOrDefault(@["cat", "sat"], 0) == 1, "ngram freq cat-sat"
  assert ngFreq.len == 3, "ngram freq distinct count"

  # n-gram frequencies - degenerate inputs
  assert ngramFrequencies(@[], 2).len == 0, "ngram freq empty tokens"
  assert ngramFrequencies(@["a", "b"], 0).len == 0, "ngram freq n<=0"
  assert ngramFrequencies(@["a"], 2).len == 0, "ngram freq len<n"

  # n=1 wraps each token: distinct count matches unique terms
  let ngUni = ngramFrequencies(@["a", "b", "a"], 1)
  assert ngUni.len == 2, "ngram freq n=1 distinct"
  assert ngUni.getOrDefault(@["a"], 0) == 2, "ngram freq n=1 count"

  # topNgrams - order by frequency desc
  let top2 = topNgrams(@["the", "cat", "the", "cat", "sat"], 2, topK = 2)
  assert top2.len == 2, "topNgrams len"
  assert top2[0] == (@["the", "cat"], 2), "topNgrams first"
  assert top2[1][1] == 1, "topNgrams second freq"

  # topNgrams - tie-break by joined form ascending (ba before bc)
  let tiedNgrams = topNgrams(@["a", "b", "a", "b", "c"], 2, topK = 3)
  assert tiedNgrams == @[(@["a", "b"], 2), (@["b", "a"], 1), (@["b", "c"], 1)],
    "topNgrams tie-break joined ascending"

  # topNgrams - topK bounds
  assert topNgrams(@["a", "b", "c"], 2, topK = 0).len == 0, "topNgrams topK=0"
  assert topNgrams(@["a", "b", "c"], 2, topK = -1).len == 0, "topNgrams topK<0"
  let topAll = topNgrams(@["a", "b", "c"], 2, topK = 100)
  assert topAll.len == 2, "topNgrams topK beyond vocab"
  assert topNgrams(@[], 2).len == 0, "topNgrams empty tokens"
  assert topNgrams(@["a"], 2).len == 0, "topNgrams len<n"

  # n-gram cosine - identical -> 1.0
  let ngCosSelf = ngramCosineSimilarity(
    wordTokenize("the cat sat"), wordTokenize("the cat sat"), 2)
  assert abs(ngCosSelf - 1.0) < 1e-9, "ngram cosine identical"

  # n-gram cosine - disjoint -> 0.0
  let ngCosDis = ngramCosineSimilarity(
    wordTokenize("a b c"), wordTokenize("d e f"), 2)
  assert ngCosDis == 0.0, "ngram cosine disjoint"

  # n-gram cosine - empty on either side -> 0.0
  assert ngramCosineSimilarity(@[], @["hello"], 2) == 0.0,
    "ngram cosine empty left"
  assert ngramCosineSimilarity(@["hello"], @[], 2) == 0.0,
    "ngram cosine empty right"
  assert ngramCosineSimilarity(@["a"], @["a"], 2) == 0.0,
    "ngram cosine no grams (len<n)"

  # n-gram jaccard - identical -> 1.0, disjoint -> 0.0
  let ngJacSelf = ngramJaccardSimilarity(
    wordTokenize("the cat sat"), wordTokenize("the cat sat"), 2)
  assert abs(ngJacSelf - 1.0) < 1e-9, "ngram jaccard identical"
  let ngJacDis = ngramJaccardSimilarity(
    wordTokenize("a b c"), wordTokenize("d e f"), 2)
  assert ngJacDis == 0.0, "ngram jaccard disjoint"
  assert ngramJaccardSimilarity(@[], @["hello"], 2) == 0.0,
    "ngram jaccard empty left"
  assert ngramJaccardSimilarity(@["hello"], @[], 2) == 0.0,
    "ngram jaccard empty right"

  # SHOWCASE: word order matters for n-grams but not for bags
  let orderA = wordTokenize("dog bites man")
  let orderB = wordTokenize("man bites dog")
  let bagOrder = cosineSimilarityBags(orderA, orderB)
  let gramOrder = ngramCosineSimilarity(orderA, orderB, 2)
  assert abs(bagOrder - 1.0) < 1e-9, "showcase bag cosine blind to order"
  assert gramOrder == 0.0, "showcase n-gram cosine sees order"
  assert gramOrder < bagOrder, "showcase n-grams capture word order"

  echo "\n=== Generic base primitives ==="

  # countItems - strings
  let ci = countItems(@["a", "b", "a"])
  assert ci.getOrDefault("a", 0) == 2, "countItems strings"
  assert ci.getOrDefault("b", 0) == 1, "countItems strings"
  assert countItems(newSeq[string]()).len == 0, "countItems empty"

  # countItems - seq keys (n-gram style)
  let ciSeq = countItems(@[@["a", "b"], @["a", "b"], @["b", "c"]])
  assert ciSeq.getOrDefault(@["a", "b"], 0) == 2, "countItems seq keys"
  assert ciSeq.len == 2, "countItems seq keys distinct"

  # cosineCounts - identical -> ~1.0
  let ccSelf = cosineCounts(countItems(@["a", "b", "a"]),
    countItems(@["a", "b", "a"]))
  assert abs(ccSelf - 1.0) < 1e-9, "cosineCounts identical"
  # cosineCounts - disjoint -> 0.0
  assert cosineCounts(countItems(@["a"]), countItems(@["b"])) == 0.0,
    "cosineCounts disjoint"
  # cosineCounts - empty on either side -> 0.0
  assert cosineCounts(initTable[string, int](), countItems(@["a"])) == 0.0,
    "cosineCounts empty left"
  assert cosineCounts(countItems(@["a"]), initTable[string, int]()) == 0.0,
    "cosineCounts empty right"

  # jaccardSets - partial overlap
  assert abs(jaccardSets(toHashSet(["a", "b"]), toHashSet(["b", "c"])) -
    (1.0 / 3.0)) < 1e-9, "jaccardSets partial"
  assert jaccardSets(toHashSet(["a"]), toHashSet(["a"])) == 1.0,
    "jaccardSets identical"
  assert jaccardSets(toHashSet(["a"]), toHashSet(["b"])) == 0.0,
    "jaccardSets disjoint"
  assert jaccardSets(initHashSet[string](),
    initHashSet[string]()) == 0.0, "jaccardSets both empty"

  # overlapSets - partial and containment
  assert overlapSets(toHashSet(["a", "b"]),
    toHashSet(["b", "c", "d"])) == 0.5, "overlapSets partial"
  assert overlapSets(toHashSet(["a"]), toHashSet(["a", "b"])) == 1.0,
    "overlapSets containment"
  assert overlapSets(initHashSet[string](),
    initHashSet[string]()) == 0.0, "overlapSets both empty"
  assert overlapSets(initHashSet[string](), toHashSet(["a"])) == 0.0,
    "overlapSets empty left"

  echo "  ALL PASS"

when isMainModule:
  testAnalysis()
