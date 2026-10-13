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

when isMainModule:
  testTopics()
