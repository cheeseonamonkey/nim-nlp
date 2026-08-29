## LDA (Latent Dirichlet Allocation) for the nlp library.
##
## Collapsed Gibbs sampling — dependency-free (std/random + std/math only).
## Mirrors the `fitTfidf`/`fitBm25`/`fitCounts` call shape for familiarity.
##
## Advanced aspects included:
## - Perplexity (training, via phi/theta)
## - Coherence: UMass (document-level log-conditional) and c_v-style PMI
##   (document co-occurrence) averaged over topics
## - Topic labeling: `topTerms` (phi) and `topTermsByPMI` (PMI-reranked)
## - Fold-in inference for unseen docs: `transformLda`
##
## No new data files, no external dependencies. Deterministic via `seed`.

import std/[tables, math, random, algorithm, sets, sequtils]
import types

type
  LdaModel* = object
    ## Fitted LDA model.
    vocabulary*: Vocabulary
    numDocs*: int
    numTopics*: int
    alpha*: float
    beta*: float
    iterations*: int
    seed*: int64
    # Sufficient statistics
    docTopicCounts*: seq[seq[int]]  # D x K
    topicWordCounts*: seq[seq[int]] # K x V
    topicSums*: seq[int]            # K
    docLengths*: seq[int]           # D
    # Token-level assignments: parallel to `tokenizedDocs` but as term ids
    docTokens*: seq[seq[int]]       # D x Nd  (term ids, -1 for OOV — filtered out)
    assignments*: seq[seq[int]]     # D x Nd  (topic id per token)
    # Derived distributions (computed after sampling)
    phi*: seq[seq[float]]   # K x V  p(w|z)
    theta*: seq[seq[float]] # D x K  p(z|d)
    tokenizedDocs*: seq[seq[string]]

# ============================================================
# Helpers
# ============================================================

proc buildVocab(docs: seq[seq[string]]): Vocabulary =
  result = newVocabulary()
  for doc in docs:
    for term in doc:
      discard result.add(term)

proc rebuildPhiTheta(m: var LdaModel) =
  let V = m.vocabulary.len
  let K = m.numTopics
  let D = m.numDocs
  m.phi = newSeq[seq[float]](K)
  for k in 0 ..< K:
    m.phi[k] = newSeq[float](V)
    let denom = m.topicSums[k].float + V.float * m.beta
    for v in 0 ..< V:
      m.phi[k][v] =
        if denom == 0.0: 0.0
        else: (m.topicWordCounts[k][v].float + m.beta) / denom
  m.theta = newSeq[seq[float]](D)
  for d in 0 ..< D:
    m.theta[d] = newSeq[float](K)
    let denom = m.docLengths[d].float + K.float * m.alpha
    for k in 0 ..< K:
      m.theta[d][k] =
        if denom == 0.0: 0.0
        else: (m.docTopicCounts[d][k].float + m.alpha) / denom

proc sampleFromProbs(rng: var Rand, probs: seq[float]): int =
  ## Categorical sample — probs need not sum to 1.
  var total = 0.0
  for p in probs: total += p
  if total <= 0.0:
    return rng.rand(probs.len - 1)
  var r = rng.rand(total)
  var cum = 0.0
  for i, p in probs:
    cum += p
    if r < cum: return i
  probs.len - 1

# ============================================================
# Fit — collapsed Gibbs
# ============================================================

proc fitLda*(
  documents: seq[seq[string]],
  numTopics: int = 5,
  iterations: int = 500,
  alpha: float = 0.1,
  beta: float = 0.01,
  seed: int64 = 42,
): LdaModel =
  ## Fit LDA via collapsed Gibbs sampling.
  ##
  ## - `numTopics` (K): number of latent topics.
  ## - `iterations`: Gibbs sweeps over the corpus.
  ## - `alpha`, `beta`: Dirichlet priors for doc-topic and topic-word.
  ## - `seed`: RNG seed for determinism.
  ## Returns an empty model when `documents` is empty or `numTopics <= 0`.
  if documents.len == 0 or numTopics <= 0:
    result.vocabulary = newVocabulary()
    result.numDocs = 0
    result.numTopics = max(0, numTopics)
    result.alpha = alpha
    result.beta = beta
    result.iterations = iterations
    result.seed = seed
    return
  if alpha <= 0.0 or beta <= 0.0:
    raise newException(ValueError, "LDA requires alpha > 0 and beta > 0")
  if iterations <= 0:
    raise newException(ValueError, "LDA requires iterations > 0")

  let vocab = buildVocab(documents)
  let V = vocab.len
  let K = numTopics
  let D = documents.len

  result.vocabulary = vocab
  result.numDocs = D
  result.numTopics = K
  result.alpha = alpha
  result.beta = beta
  result.iterations = iterations
  result.seed = seed
  result.tokenizedDocs = documents

  # Map docs to term ids, filtering OOV (none, since vocab is from docs).
  result.docTokens = newSeq[seq[int]](D)
  result.docLengths = newSeq[int](D)
  for d, doc in documents:
    var ids: seq[int]
    ids = newSeqOfCap[int](doc.len)
    for term in doc:
      let id = vocab.id(term)
      if id >= 0:
        ids.add(id)
    result.docTokens[d] = ids
    result.docLengths[d] = ids.len

  result.docTopicCounts = newSeq[seq[int]](D)
  for d in 0 ..< D:
    result.docTopicCounts[d] = newSeq[int](K)
  result.topicWordCounts = newSeq[seq[int]](K)
  for k in 0 ..< K:
    result.topicWordCounts[k] = newSeq[int](V)
  result.topicSums = newSeq[int](K)
  result.assignments = newSeq[seq[int]](D)

  var rng = initRand(seed)

  # Random initial assignment.
  for d in 0 ..< D:
    let Nd = result.docTokens[d].len
    result.assignments[d] = newSeq[int](Nd)
    for n in 0 ..< Nd:
      let w = result.docTokens[d][n]
      let k = rng.rand(K - 1)
      result.assignments[d][n] = k
      inc result.docTopicCounts[d][k]
      inc result.topicWordCounts[k][w]
      inc result.topicSums[k]

  # Gibbs sweeps.
  for _ in 0 ..< iterations:
    for d in 0 ..< D:
      for n in 0 ..< result.docTokens[d].len:
        let w = result.docTokens[d][n]
        var kOld = result.assignments[d][n]
        # Decrement
        dec result.docTopicCounts[d][kOld]
        dec result.topicWordCounts[kOld][w]
        dec result.topicSums[kOld]
        # Conditional p(z=k | rest) ∝ (n_{d,k}+α)(n_{k,w}+β)/(n_{k,·}+Vβ)
        var probs = newSeq[float](K)
        for k in 0 ..< K:
          let a = result.docTopicCounts[d][k].float + alpha
          let b = result.topicWordCounts[k][w].float + beta
          let c = result.topicSums[k].float + V.float * beta
          probs[k] = a * b / c
        let kNew = sampleFromProbs(rng, probs)
        result.assignments[d][n] = kNew
        inc result.docTopicCounts[d][kNew]
        inc result.topicWordCounts[kNew][w]
        inc result.topicSums[kNew]

  rebuildPhiTheta(result)

# ============================================================
# Accessors & topic labeling
# ============================================================

proc docTopicDistribution*(m: LdaModel, docId: int): seq[float] =
  ## p(z | d) for a training doc. Empty seq if out of range or empty model.
  if docId < 0 or docId >= m.theta.len: return @[]
  result = m.theta[docId]

proc topicWordDistribution*(m: LdaModel, topic: int): seq[float] =
  ## p(w | z) for a topic. Empty seq if out of range.
  if topic < 0 or topic >= m.phi.len: return @[]
  result = m.phi[topic]

proc topTerms*(m: LdaModel, topic: int, n: int = 10): seq[(string, float)] =
  ## Top `n` terms for `topic` by phi = p(w|z), tie-broken lexicographically.
  if topic < 0 or topic >= m.phi.len or n <= 0: return @[]
  let V = m.vocabulary.len
  var scored: seq[(string, float)] = newSeqOfCap[(string, float)](V)
  for v in 0 ..< V:
    scored.add((m.vocabulary[v], m.phi[topic][v]))
  scored.sort(proc(a, b: (string, float)): int =
    if a[1] == b[1]: cmp(a[0], b[0])
    else: cmp(b[1], a[1]))
  if scored.len > n: scored[0 ..< n] else: scored

proc buildDocFreq(docs: seq[seq[string]]): Table[string, int] =
  result = initTable[string, int]()
  for doc in docs:
    var seen = initHashSet[string]()
    for t in doc:
      if t notin seen:
        seen.incl(t)
        result[t] = result.getOrDefault(t, 0) + 1

proc buildCooccurrence(
  docs: seq[seq[string]]
): (Table[string, int], Table[(string, string), int]) =
  ## Document-level co-occurrence for PMI. Pair key is sorted (a <= b).
  var df = initTable[string, int]()
  var pair = initTable[(string, string), int]()
  for doc in docs:
    var uniq = toHashSet(doc)
    for t in uniq:
      df[t] = df.getOrDefault(t, 0) + 1
    let terms = toSeq(uniq)
    for i in 0 ..< terms.len:
      for j in (i + 1) ..< terms.len:
        let a = terms[i]
        let b = terms[j]
        let key = if a <= b: (a, b) else: (b, a)
        pair[key] = pair.getOrDefault(key, 0) + 1
  (df, pair)

proc pmiForPair(
  a, b: string,
  df: Table[string, int],
  pair: Table[(string, string), int],
  D: int,
  eps: float = 1e-12
): float =
  let key = if a <= b: (a, b) else: (b, a)
  let co = pair.getOrDefault(key, 0).float
  let pa = df.getOrDefault(a, 0).float / max(1.0, D.float)
  let pb = df.getOrDefault(b, 0).float / max(1.0, D.float)
  let pab = co / max(1.0, D.float)
  if pab <= eps or pa <= eps or pb <= eps: return -10.0
  ln(pab / (pa * pb))

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
  # Mean PMI per candidate
  var meanPmi = newSeq[float](pool.len)
  for i, (term, _) in pool:
    var s = 0.0
    var c = 0
    for j, (other, _) in pool:
      if i == j: continue
      s += pmiForPair(term, other, df, pair, D)
      inc c
    meanPmi[i] = if c == 0: 0.0 else: s / c.float
  # Normalize mean PMI to [0,1] for blending
  var lo = meanPmi[0]
  var hi = meanPmi[0]
  for v in meanPmi:
    if v < lo: lo = v
    if v > hi: hi = v
  let span = hi - lo
  var scored: seq[(string, float, float)] # (term, blended, phi)
  for i, (term, phiW) in pool:
    let normPmi =
      if span <= 1e-12: 0.5
      else: (meanPmi[i] - lo) / span
    let blended = 0.5 * phiW + 0.5 * normPmi
    scored.add((term, blended, phiW))
  scored.sort(proc(a, b: (string, float, float)): int =
    if a[1] == b[1]: cmp(a[0], b[0])
    else: cmp(b[1], a[1]))
  result = newSeqOfCap[(string, float)](min(n, scored.len))
  for i in 0 ..< min(n, scored.len):
    result.add((scored[i][0], scored[i][2]))

# ============================================================
# Perplexity & coherence
# ============================================================

proc perplexity*(m: LdaModel): float =
  ## Training perplexity: exp(- log p(w) / N).
  ## Uses current phi/theta; returns 0 for empty model.
  if m.numDocs == 0 or m.vocabulary.len == 0: return 0.0
  var logLik = 0.0
  var N = 0
  for d in 0 ..< m.numDocs:
    for n in 0 ..< m.docTokens[d].len:
      let w = m.docTokens[d][n]
      var p = 0.0
      for k in 0 ..< m.numTopics:
        p += m.theta[d][k] * m.phi[k][w]
      if p <= 1e-12: p = 1e-12
      logLik += ln(p)
      inc N
  if N == 0: return 0.0
  exp(-logLik / N.float)

proc coherenceUMass*(
  m: LdaModel,
  documents: seq[seq[string]],
  topN: int = 10,
  eps: float = 1e-12
): float =
  ## UMass coherence averaged over topics: mean over consecutive pairs
  ## in `topTerms` of log((D(w_i,w_j)+eps)/D(w_j)). Document-level.
  if m.numTopics == 0 or documents.len == 0 or topN <= 0: return 0.0
  let df = buildDocFreq(documents)
  var pair = initTable[(string, string), int]()
  for doc in documents:
    let uniq = toHashSet(doc)
    let terms = toSeq(uniq)
    for i in 0 ..< terms.len:
      for j in (i + 1) ..< terms.len:
        let a = terms[i]; let b = terms[j]
        let key = if a <= b: (a, b) else: (b, a)
        pair[key] = pair.getOrDefault(key, 0) + 1
  var total = 0.0
  var count = 0
  for k in 0 ..< m.numTopics:
    let top = m.topTerms(k, topN)
    if top.len < 2: continue
    for i in 1 ..< top.len:
      for j in 0 ..< i:
        let wi = top[i][0]; let wj = top[j][0]
        let key = if wi <= wj: (wi, wj) else: (wj, wi)
        let co = pair.getOrDefault(key, 0).float
        let dj = df.getOrDefault(wj, 0).float
        total += ln((co + eps) / max(eps, dj))
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
    var s = 0.0
    var c = 0
    for i in 1 ..< top.len:
      for j in 0 ..< i:
        s += pmiForPair(top[i][0], top[j][0], df, pair, D)
        inc c
    if c > 0:
      total += s / c.float
      inc topicsUsed
  if topicsUsed == 0: 0.0 else: total / topicsUsed.float

# ============================================================
# Fold-in: infer topic distribution for an unseen document
# ============================================================

proc transformLda*(
  m: LdaModel,
  doc: seq[string],
  iterations: int = 100,
  seed: int64 = 0,
): seq[float] =
  ## Fold-in inference for an unseen tokenized document.
  ## Keeps phi fixed and Gibbs-samples topic assignments for the new doc.
  ## `seed==0` derives a seed from model seed + doc hash for determinism.
  if m.numTopics == 0 or m.vocabulary.len == 0 or doc.len == 0:
    return newSeq[float](max(0, m.numTopics))
  if iterations <= 0:
    raise newException(ValueError, "transformLda requires iterations > 0")
  var ids: seq[int]
  ids = newSeqOfCap[int](doc.len)
  for term in doc:
    let id = m.vocabulary.id(term)
    if id >= 0: ids.add(id)
  if ids.len == 0:
    # No in-vocab terms → prior only
    result = newSeq[float](m.numTopics)
    for k in 0 ..< m.numTopics: result[k] = 1.0 / m.numTopics.float
    return
  let effSeed =
    if seed != 0: seed
    else:
      var h = m.seed
      for id in ids: h = h * 31 + id.int64
      h
  var rng = initRand(effSeed)
  var docCounts = newSeq[int](m.numTopics)
  var assigns = newSeq[int](ids.len)
  for n, _ in ids:
    let k = rng.rand(m.numTopics - 1)
    assigns[n] = k
    inc docCounts[k]
  for _ in 0 ..< iterations:
    for n, w in ids:
      let kOld = assigns[n]
      dec docCounts[kOld]
      var probs = newSeq[float](m.numTopics)
      for k in 0 ..< m.numTopics:
        let a = docCounts[k].float + m.alpha
        let b = m.phi[k][w]
        # Collapsed conditional ∝ (n_{d,k}+α) * φ_{k,w}
        probs[k] = a * b
      let kNew = sampleFromProbs(rng, probs)
      assigns[n] = kNew
      inc docCounts[kNew]
  result = newSeq[float](m.numTopics)
  let denom = ids.len.float + m.numTopics.float * m.alpha
  for k in 0 ..< m.numTopics:
    result[k] = (docCounts[k].float + m.alpha) / denom
