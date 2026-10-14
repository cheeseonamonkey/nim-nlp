## Topic-coherence metrics for the nlp library.
##
## Document-level co-occurrence statistics used to score and label LDA
## topics: PMI-reranked topic terms, UMass coherence, and c_v-style PMI
## coherence. Kept separate from the Gibbs sampler in `topics`.

import std/[tables, math, sets, sequtils]
import topics
import private/model_utils

proc sortedPairKey*(a, b: string): (string, string) =
  ## Canonical unordered-pair key (`a <= b` first) for co-occurrence
  ## tables — every df/pair lookup goes through this ordering.
  if a <= b: (a, b) else: (b, a)

proc accumulateDocCooccurrence(
  doc: seq[string],
  df: var Table[string, int],
  pair: var Table[(string, string), int]
) =
  ## Per-document step of co-occurrence building: count each unique term
  ## once toward `df`, each unordered term pair once toward `pair`.
  let uniq = toHashSet(doc)
  for t in uniq:
    df[t] = df.getOrDefault(t, 0) + 1
  let terms = toSeq(uniq)
  for i in 0 ..< terms.len:
    for j in (i + 1) ..< terms.len:
      let key = sortedPairKey(terms[i], terms[j])
      pair[key] = pair.getOrDefault(key, 0) + 1

proc buildCooccurrence(
  docs: seq[seq[string]]
): (Table[string, int], Table[(string, string), int]) =
  ## Document-level co-occurrence for PMI. Pair key is sorted (a <= b).
  var df = initTable[string, int]()
  var pair = initTable[(string, string), int]()
  for doc in docs:
    accumulateDocCooccurrence(doc, df, pair)
  (df, pair)

proc pmiForPair(
  a, b: string,
  df: Table[string, int],
  pair: Table[(string, string), int],
  D: int,
  eps: float = 1e-12
): float =
  let key = sortedPairKey(a, b)
  let co = pair.getOrDefault(key, 0).float
  let pa = df.getOrDefault(a, 0).float / max(1.0, D.float)
  let pb = df.getOrDefault(b, 0).float / max(1.0, D.float)
  let pab = co / max(1.0, D.float)
  if pab <= eps or pa <= eps or pb <= eps: return -10.0
  ln(pab / (pa * pb))

proc meanPmiScores*(
  terms: seq[string],
  df: Table[string, int],
  pair: Table[(string, string), int],
  D: int
): seq[float] =
  ## Per-term mean PMI to every *other* term — the per-candidate score
  ## behind PMI-reranked labeling.
  result = newSeq[float](terms.len)
  for i, term in terms:
    var s = 0.0
    var c = 0
    for j, other in terms:
      if i == j: continue
      s += pmiForPair(term, other, df, pair, D)
      inc c
    result[i] = if c == 0: 0.0 else: s / c.float

proc meanPairwisePmi*(
  terms: seq[string],
  df: Table[string, int],
  pair: Table[(string, string), int],
  D: int
): float =
  ## Mean PMI over all unordered term pairs — the per-topic score behind
  ## c_v coherence. Returns 0.0 when there is no pair to score.
  var s = 0.0
  var c = 0
  for i in 1 ..< terms.len:
    for j in 0 ..< i:
      s += pmiForPair(terms[i], terms[j], df, pair, D)
      inc c
  if c == 0: 0.0 else: s / c.float

proc normalizeUnit*(scores: seq[float]): seq[float] =
  ## Min-max normalization to [0, 1]. A degenerate (constant) input maps
  ## to 0.5 throughout so blending stays neutral instead of dividing by ~0.
  if scores.len == 0: return @[]
  var lo = scores[0]
  var hi = scores[0]
  for v in scores:
    if v < lo: lo = v
    if v > hi: hi = v
  let span = hi - lo
  result = newSeq[float](scores.len)
  for i, v in scores:
    result[i] =
      if span <= 1e-12: 0.5
      else: (v - lo) / span

proc umassPairScore*(
  wi, wj: string,
  df: Table[string, int],
  pair: Table[(string, string), int],
  eps: float = 1e-12
): float =
  ## One UMass pair term: log((D(w_i,w_j)+eps)/D(w_j)).
  let key = sortedPairKey(wi, wj)
  let co = pair.getOrDefault(key, 0).float
  let dj = df.getOrDefault(wj, 0).float
  ln((co + eps) / max(eps, dj))

proc topTermsByPMI*(
  m: LdaModel,
  documents: seq[seq[string]],
  topic: int,
  n: int = 10,
  candidatePool: int = 20,
): seq[(string, float)] =
  ## PMI-reranked labeling: score each candidate term by its mean PMI
  ## to the other candidates (document co-occurrence PMI), then rank by
  ## 0.5*phi + 0.5*normalized-MI. Falls back to `topTerms` when
  ## co-occurrence is unavailable.
  if topic < 0 or topic >= m.phi.len or n <= 0: return @[]
  let pool = m.topTerms(topic, max(n, candidatePool))
  if pool.len == 0: return @[]
  if documents.len == 0:
    return pool[0 ..< min(n, pool.len)]
  let D = documents.len
  let (df, pair) = buildCooccurrence(documents)
  var poolTerms = newSeqOfCap[string](pool.len)
  for (term, _) in pool: poolTerms.add(term)
  let meanPmi = meanPmiScores(poolTerms, df, pair, D)
  let normPmi = normalizeUnit(meanPmi)
  # Blend 0.5*phi + 0.5*normalized-PMI, then report raw phi weights.
  var blended = newSeqOfCap[(string, float)](pool.len)
  var phiByTerm = initTable[string, float]()
  for i, (term, phiW) in pool:
    blended.add((term, 0.5 * phiW + 0.5 * normPmi[i]))
    phiByTerm[term] = phiW
  let ranked = topScored(blended, n)
  result = newSeqOfCap[(string, float)](ranked.len)
  for (term, _) in ranked:
    result.add((term, phiByTerm[term]))

proc coherenceUMass*(
  m: LdaModel,
  documents: seq[seq[string]],
  topN: int = 10,
  eps: float = 1e-12
): float =
  ## UMass coherence averaged over topics: mean over consecutive pairs
  ## in `topTerms` of log((D(w_i,w_j)+eps)/D(w_j)). Document-level.
  if m.numTopics == 0 or documents.len == 0 or topN <= 0: return 0.0
  let (df, pair) = buildCooccurrence(documents)
  var total = 0.0
  var count = 0
  for k in 0 ..< m.numTopics:
    let top = m.topTerms(k, topN)
    if top.len < 2: continue
    for i in 1 ..< top.len:
      for j in 0 ..< i:
        total += umassPairScore(top[i][0], top[j][0], df, pair, eps)
        inc count
  if count == 0: 0.0 else: total / count.float

proc coherenceCV*(
  m: LdaModel,
  documents: seq[seq[string]],
  topN: int = 10,
): float =
  ## c_v-style coherence (PMI-based): mean over topics of mean pairwise
  ## document-level PMI among top terms. Higher is more coherent.
  if m.numTopics == 0 or documents.len == 0 or topN <= 0: return 0.0
  let D = documents.len
  let (df, pair) = buildCooccurrence(documents)
  var total = 0.0
  var topicsUsed = 0
  for k in 0 ..< m.numTopics:
    let top = m.topTerms(k, topN)
    if top.len < 2: continue
    var terms = newSeqOfCap[string](top.len)
    for (term, _) in top: terms.add(term)
    total += meanPairwisePmi(terms, df, pair, D)
    inc topicsUsed
  if topicsUsed == 0: 0.0 else: total / topicsUsed.float
