## LDA topic-modeling tests for the nlp library.

import ../src/nlp
import std/[tables, sets, math]

proc testTopics() =
  echo "\n=== Topics ==="

  let docs = @[
    wordTokenize("cat dog cat dog pet"),
    wordTokenize("dog cat pet animal"),
    wordTokenize("engine piston wheel car"),
    wordTokenize("car wheel piston engine"),
    wordTokenize("engine car piston wheel")
  ]

  # Fit produces the expected shape.
  let model = fitLda(docs, numTopics = 3, iterations = 30, seed = 42)
  assert model.numDocs == 5, "lda numDocs"
  assert model.numTopics == 3, "lda numTopics"
  assert model.phi.len == 3, "lda phi topics"
  assert model.theta.len == 5, "lda theta docs"

  # Same seed and iterations produce element-wise identical phi.
  let again = fitLda(docs, numTopics = 3, iterations = 30, seed = 42)
  assert again.phi.len == model.phi.len, "lda repeat phi len"
  for k in 0 ..< model.phi.len:
    assert again.phi[k].len == model.phi[k].len, "lda repeat row len"
    for v in 0 ..< model.phi[k].len:
      assert again.phi[k][v] == model.phi[k][v], "lda deterministic phi"

  # Fold-in inference returns a topic distribution summing to ~1.0.
  let dist = model.transformLda(wordTokenize("car engine wheel"), iterations = 20)
  assert dist.len == model.numTopics, "lda transform length"
  var total = 0.0
  for p in dist: total += p
  assert abs(total - 1.0) < 1e-3, "lda transform sums to 1"

  # Coherence and labeling are available and finite.
  let umass = model.coherenceUMass(docs, topN = 3)
  let cv = model.coherenceCV(docs, topN = 3)
  assert umass == umass, "lda umass coherence finite"
  assert cv == cv, "lda cv coherence finite"
  assert model.topTerms(0, n = 3).len == 3, "lda topTerms"

  # Empty corpus yields an empty model.
  let empty = fitLda(@[])
  assert empty.numDocs == 0, "lda empty numDocs"
  let emptyZero = fitLda(@[], numTopics = 0)
  assert emptyZero.numTopics == 0, "lda empty numTopics"

  echo "  ALL PASS"

proc testCoherenceHelpers() =
  echo "\n=== Coherence helpers ==="

  # sortedPairKey canonicalizes unordered pairs for every lookup table.
  assert sortedPairKey("b", "a") == ("a", "b"), "pair key sorts"
  assert sortedPairKey("a", "b") == ("a", "b"), "pair key keeps order"
  assert sortedPairKey("x", "x") == ("x", "x"), "pair key self-pair"

  # normalizeUnit maps to [0, 1]; a constant input stays neutral at 0.5.
  assert normalizeUnit(@[2.0, 4.0, 6.0]) == @[0.0, 0.5, 1.0],
    "normalizeUnit min-max"
  assert normalizeUnit(@[3.0, 3.0]) == @[0.5, 0.5],
    "normalizeUnit constant input"
  assert normalizeUnit(newSeq[float](0)) == newSeq[float](0),
    "normalizeUnit empty"

  # meanPairwisePmi agrees with hand-scored PMI on a tiny corpus.
  # Corpus: [a b] [a b c] [c] -> df = {a:2,b:2,c:2},
  # pairs = {(a,b):2,(a,c):1,(b,c):1}, D = 3.
  var df = initTable[string, int]()
  df["a"] = 2
  df["b"] = 2
  df["c"] = 2
  var pr = initTable[(string, string), int]()
  pr[("a", "b")] = 2
  pr[("a", "c")] = 1
  pr[("b", "c")] = 1
  const D = 3
  let mean = meanPairwisePmi(@["a", "b", "c"], df, pr, D)
  let expectMean = (ln(1.5) + 2.0 * ln(0.75)) / 3.0
  assert abs(mean - expectMean) < 1e-12, "meanPairwisePmi averages pairs"
  assert meanPairwisePmi(@["solo"], df, pr, D) == 0.0,
    "meanPairwisePmi needs a pair"

  # meanPmiScores matches direct pairwise scoring per candidate.
  let per = meanPmiScores(@["a", "b", "c"], df, pr, D)
  assert per.len == 3, "meanPmiScores length"
  let expectA = (ln(1.5) + ln(0.75)) / 2.0
  assert abs(per[0] - expectA) < 1e-12, "meanPmiScores candidate average"

  # umassPairScore is the log co-occurrence ratio: co=2, dj=2 -> ln(1).
  assert abs(umassPairScore("a", "b", df, pr)) < 1e-12,
    "umassPairScore value"
  assert abs(umassPairScore("b", "a", df, pr) -
    umassPairScore("a", "b", df, pr)) < 1e-12,
    "umassPairScore is order-insensitive"

  echo "  ALL PASS"

when isMainModule:
  testTopics()
  testCoherenceHelpers()
