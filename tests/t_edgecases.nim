## Edge-case and defect-regression tests for the nlp library.
##
## Two kinds of assertions live here:
##
## 1. Edge cases whose current behavior is correct (or at least defensible)
##    and worth freezing, so a refactor cannot silently change them.
## 2. Known defects, marked `BUG(PINNED)`. These deliberately assert today's
##    WRONG output. Fixing a defect will therefore break exactly one visible
##    assertion instead of silently changing behavior. When a defect is fixed,
##    update the pinned assertion to the corrected value.
##
## Known defects recorded here:
##
##   D1  (fixed in 0.2.1) BM25's probabilistic IDF collapsed to zero for
##       terms present in about half the corpus or more, and the `score > 0`
##       filter then silently dropped every hit — searching for a common word
##       returned nothing at all. The weight is now shifted by +1 inside the
##       logarithm so it stays strictly positive. The retrieval section below
##       asserts the corrected behavior.
##   D2  `normalizeReplaceEmails("a@.")` yields "<email>." — "a@" is not an
##       email. The trailing-dot trim removes the very dot that satisfied the
##       dot-in-domain requirement, and the span is reported anyway.
##   D3  `normalizeStripHtml` stops at the first '>' even inside a quoted
##       attribute, so `<a href="x>y">z</a>` becomes `y">z`.
##   D4  The custom-stopword overload is not case-insensitive despite its
##       doc comment: the caller's set is used verbatim while only the tokens
##       are lowercased, so `toHashSet(["THE"]).removeStopWords` matches
##       nothing.
##   D5  `transformLda` returns an all-zero vector for an empty document (a
##       "distribution" summing to 0) but a uniform vector for an
##       out-of-vocabulary document. The two empty-ish inputs disagree.
##   D6  `stemTokens` is exported by both `porter_stemmer` and
##       `smart_stemmer` with the same signature, and `nlp.nim` re-exports
##       both, so `import nlp; stemTokens(tokens)` is an ambiguous call that
##       fails to compile. Callers must disambiguate by importing one module
##       directly. This file does exactly that.
##
## The stemmer pins at the bottom are not defects: the two stemmers
## intentionally disagree, and the pins protect that disagreement from being
## "cleaned up" by a refactor.

import ../src/nlp
import ../src/nlp/porter_stemmer as porter
import ../src/nlp/smart_stemmer as smart
import std/[assertions, math, sets]

# ============================================================
# Tokenizer edge cases
# ============================================================

proc testTokenizerEdges() =
  echo "=== Edge: tokenizers ==="

  # Character n-grams guard non-positive n (they must not emit empty strings).
  assert ngramTokenize("abc", 0) == @[], "ngram n=0"
  assert ngramTokenize("abc", -1) == @[], "ngram n<0"
  assert ngramTokenize("abc", 4) == @[], "ngram n>len"
  assert ngramTokenize("abc", 3) == @["abc"], "ngram n==len"

  # n-grams operate on runes, not bytes: an astral-plane character is one unit.
  assert ngramTokenize("a😀b", 2) == @["a😀", "😀b"], "ngram over runes"

  # The word tokenizer is ASCII-alphanumeric only, so accented letters split.
  assert wordTokenize("café") == @["caf"], "word tokenizer splits on accent"
  assert wordTokenize("naïve") == @["na", "ve"], "word tokenizer splits on diaeresis"

  # NBSP (U+00A0) is not ASCII whitespace, so it does not separate tokens.
  assert whitespaceTokenize("a\xC2\xA0b") == @["a\xC2\xA0b"], "NBSP is not a separator"

  # Sentence splitting is punctuation-driven and unaware of abbreviations or
  # quotes: "Dr." splits, and a full stop inside quotes does not.
  assert splitSentences("Dr. Smith went home. He slept.") ==
    @["Dr.", "Smith went home.", "He slept."], "sentence splitter and abbreviations"
  assert splitSentences("He said \"Hi.\" Then left.") ==
    @["He said \"Hi.\" Then left."], "sentence splitter and quoted stop"
  assert splitSentences("Wait... what?!") == @["Wait...", "what?!"],
    "sentence splitter runs of terminators"

  echo "  ALL PASS"

# ============================================================
# Normalizer edge cases
# ============================================================

proc testNormalizerEdges() =
  echo "=== Edge: normalizers ==="

  # Combining diacritics are removed even when the base letter is not in the
  # table; characters that need a multi-letter expansion are left alone.
  assert normalizeStripAccents("cafe\xCC\x81") == "cafe", "combining acute removed"
  assert normalizeStripAccents("cœur") == "cœur", "oe-ligature left alone"
  assert normalizeStripAccents("straße") == "straße", "sharp-s left alone"

  # Punctuation stripping is ASCII-only.
  assert normalizeStripPunctuation("a—b—“c”") == "a—b—“c”", "unicode punctuation kept"

  # Number normalization replaces runs of digits, not numeric literals.
  assert normalizeRemoveNumbers("v1.2.3") == "v..", "digit runs replaced"

  # HTML stripping is a tag scanner, not a parser: nested tags work, a bare
  # comparison and a heart are left alone, and an unterminated tag is kept.
  assert normalizeStripHtml("<i><b>x</b></i>") == "x", "nested tags"
  assert normalizeStripHtml("a < b") == "a < b", "bare comparison kept"
  assert normalizeStripHtml("<3") == "<3", "heart kept"
  assert normalizeStripHtml("unclosed <tag") == "unclosed <tag", "unterminated tag kept"

  # BUG(PINNED) D3: a '>' inside a quoted attribute value ends the tag early.
  assert normalizeStripHtml("<a href=\"x>y\">z</a>") == "y\">z",
    "BUG(PINNED) D3: quoted '>' truncates the tag"

  # Email spans; the simple heuristic needs a dot in the domain.
  assert normalizeReplaceEmails("a@b") == "a@b", "email needs a dotted domain"
  assert normalizeReplaceEmails("a@") == "a@", "email needs a domain"
  assert normalizeReplaceEmails("a@b.c") == "<email>", "short dotted domain matches"
  # BUG(PINNED) D2: the dot is trimmed away and the span is still reported.
  assert normalizeReplaceEmails("a@.") == "<email>.",
    "BUG(PINNED) D2: 'a@.' is not an email"

  # URL spans; only http/https/www are recognized, and trailing punctuation is
  # trimmed from the span rather than consumed.
  assert normalizeReplaceUrls("ftp://x.y/z") == "ftp://x.y/z", "only http/https/www"
  assert normalizeReplaceUrls("visit www.example.com, ok?") == "visit <url> ok?",
    "comma is a list separator, not URL punctuation"
  assert normalizeReplaceUrls("see (http://x.y/z?a=b&c=d)") == "see (<url>)",
    "closing paren trimmed from span"
  assert normalizeReplaceUrls("go to http://x.y/z.") == "go to <url>.",
    "trailing dot trimmed from span"
  assert normalizeReplaceUrls("www.") == "<url>.", "bare www still yields a span"

  echo "  ALL PASS"

# ============================================================
# Retrieval edge cases and defects
# ============================================================

proc testRetrievalEdges() =
  echo "=== Edge: retrieval ==="

  let docs = @[wordTokenize("a b"), wordTokenize("b c")]
  let bm = fitBm25(docs)

  # D1 (fixed in 0.2.1): common terms are found again. "b" is in every
  # document, and the equal scores tie-break by ascending docId.
  let commonHits = bm.searchBm25(wordTokenize("b"))
  assert commonHits.len == 2, "BM25 finds a term present in every document"
  assert commonHits[0][0] == 0 and commonHits[1][0] == 1, "BM25 tie-break by docId"
  assert commonHits[0][1] > 0.0 and commonHits[1][1] > 0.0,
    "BM25 common-term scores stay positive"
  assert abs(commonHits[0][1] - 0.1823215567939546) < 1e-9,
    "BM25 common-term score " & $commonHits[0][1]

  # A term present in half the documents is also found, and only the matching
  # document is returned.
  let halfHits = bm.searchBm25(wordTokenize("a"))
  assert halfHits.len == 1 and halfHits[0][0] == 0,
    "BM25 finds a term present in half the corpus"
  assert abs(halfHits[0][1] - 0.6931471805599453) < 1e-9,
    "BM25 half-corpus score " & $halfHits[0][1]

  # Repeating a query term still increases the score.
  let doubled = bm.searchBm25(wordTokenize("b b"))
  assert doubled[0][1] > commonHits[0][1], "BM25 query frequency increases score"

  # TF-IDF agrees that the term is present; both models now find it.
  let tfHits = fitTfidf(docs).searchTfidf(wordTokenize("b"))
  assert tfHits.len == 2, "tfidf finds the common term in both documents"
  assert tfHits[0][1] > 0.0 and tfHits[1][1] > 0.0, "tfidf scores stay positive"

  # A rare term in a larger corpus keeps a larger weight than a common one.
  let bm5 = fitBm25(@[
    wordTokenize("zzz a"), wordTokenize("b c"), wordTokenize("d e"),
    wordTokenize("f g"), wordTokenize("h i")])
  let rare = bm5.searchBm25(wordTokenize("zzz"))
  assert rare.len == 1 and rare[0][0] == 0, "rare term found"
  assert abs(rare[0][1] - 1.3862943611198906) < 1e-9, "rare term idf " & $rare[0][1]
  assert rare[0][1] > commonHits[0][1], "rare terms outweigh common ones"

  # Empty corpus: average length is zero and search is a no-op, not a crash
  # or a division by zero.
  let empty = fitBm25(@[newSeq[string](), newSeq[string]()])
  assert empty.avgDocLength == 0.0, "empty corpus avgDocLength"
  assert empty.searchBm25(wordTokenize("x")).len == 0, "empty corpus search"
  assert empty.searchBm25(wordTokenize("x"), topK = 0).len == 0, "topK=0 search"

  # Parameter validation.
  doAssertRaises(ValueError):
    discard bm.searchBm25(wordTokenize("a"), k1 = -1.0)
  doAssertRaises(ValueError):
    discard bm.searchBm25(wordTokenize("a"), b = 1.5)
  doAssertRaises(ValueError):
    discard bm.searchBm25(wordTokenize("a"), b = NaN)
  doAssertRaises(ValueError):
    discard bm.searchBm25(wordTokenize("a"), k1 = NaN)

  # Vocabulary construction agrees between the two models.
  assert fitCounts(docs).vocabulary.len == fitTfidf(docs).vocabulary.len,
    "count and tfidf vocabularies agree"

  echo "  ALL PASS"

# ============================================================
# Topic-model edge cases and defects
# ============================================================

proc testTopicEdges() =
  echo "=== Edge: topics ==="

  let docs = @[
    wordTokenize("cat sat mat"),
    wordTokenize("dog sat log"),
    wordTokenize("cat dog mat log")]

  let m = fitLda(docs, numTopics = 2, iterations = 40, seed = 7)

  # Same seed and iteration count must reproduce the model exactly.
  let again = fitLda(docs, numTopics = 2, iterations = 40, seed = 7)
  assert m.phi == again.phi, "lda is deterministic for a fixed seed"

  # A document made only of out-of-vocabulary terms gets a uniform
  # distribution, which is a valid distribution.
  let oov = transformLda(m, wordTokenize("zzz qqq"))
  assert oov.len == 2, "oov fold-in length"
  assert abs(oov[0] - 0.5) < 1e-9 and abs(oov[1] - 0.5) < 1e-9, "oov fold-in uniform"

  # BUG(PINNED) D5: an empty document returns zeros, which does not sum to 1
  # and disagrees with the uniform answer given to an all-OOV document.
  let emptyDoc = transformLda(m, newSeq[string]())
  assert emptyDoc == @[0.0, 0.0], "BUG(PINNED) D5: empty doc yields an all-zero 'distribution'"

  # Out-of-range accessors return empty rather than raising.
  assert m.topTerms(0, 0).len == 0, "topTerms n=0"
  assert m.topTerms(0, -1).len == 0, "topTerms n<0"
  assert m.topTerms(99, 3).len == 0, "topTerms bad topic"
  assert m.topicWordDistribution(-1).len == 0, "topicWordDistribution bad topic"
  assert m.docTopicDistribution(99).len == 0, "docTopicDistribution bad doc"

  assert m.perplexity() > 0.0, "perplexity is positive"

  # Degenerate fits are documented to return an empty model. Note that the
  # requested topic count is retained: only numDocs is zeroed, so a default
  # fit over an empty corpus has numTopics == 5.
  let none = fitLda(@[])
  assert none.numDocs == 0, "empty corpus model has no documents"
  assert none.numTopics == 5, "empty corpus model keeps the requested K"
  let noTopics = fitLda(docs, numTopics = 0)
  assert noTopics.numDocs == 0 and noTopics.numTopics == 0,
    "zero requested topics yields an empty model"

  # Parameter validation, including NaN which must not slip through the
  # comparison-based guards.
  doAssertRaises(ValueError):
    discard fitLda(docs, numTopics = 2, iterations = 0)
  doAssertRaises(ValueError):
    discard fitLda(docs, numTopics = 2, alpha = 0.0)
  doAssertRaises(ValueError):
    discard fitLda(docs, numTopics = 2, alpha = NaN)
  doAssertRaises(ValueError):
    discard fitLda(docs, numTopics = 2, beta = -1.0)
  doAssertRaises(ValueError):
    discard transformLda(m, wordTokenize("cat"), iterations = 0)

  echo "  ALL PASS"

# ============================================================
# Stopword edge cases and defects
# ============================================================

proc testStopwordEdges() =
  echo "=== Edge: stopwords ==="

  # The language overload is case-insensitive in both directions.
  assert isStopWord("THE", "English"), "stopword language lookup is case-insensitive"
  assert wordTokenize("the and cat").removeStopWords("ENGLISH") == @["cat"],
    "language overload accepts uppercase language"

  # BUG(PINNED) D4: the custom-set overload lowercases the token but uses the
  # caller's set verbatim, so a set of uppercase words never matches despite
  # the "case-insensitive" doc comment.
  assert wordTokenize("the and cat").removeStopWords(toHashSet(["THE", "And"])) ==
    @["the", "and", "cat"], "BUG(PINNED) D4: uppercase custom set matches nothing"

  # A pre-lowercased custom set behaves as documented.
  assert wordTokenize("the and cat").removeStopWords(toHashSet(["the", "and"])) ==
    @["cat"], "lowercase custom set works"

  # Unknown language passes tokens through untouched.
  assert wordTokenize("the cat").removeStopWords("klingon") == @["the", "cat"],
    "unknown language passthrough"

  echo "  ALL PASS"

# ============================================================
# Stemmer divergence pins (not defects)
# ============================================================

proc testStemmerDivergence() =
  echo "=== Edge: stemmer divergence ==="

  # The two stemmers disagree on purpose: the smart stemmer resolves plural
  # forms lexically (-ies -> -y) while Porter strips suffixes mechanically.
  # These pins must survive any refactor of either stemmer.
  assert stemWord("flies") == "fli", "porter flies"
  assert smartStemWord("flies") == "fly", "smart flies"
  assert stemWord("cities") == "citi", "porter cities"
  assert smartStemWord("cities") == "city", "smart cities"

  # Where they agree, they must stay agreed.
  assert stemWord("running") == "run" and smartStemWord("running") == "run",
    "both stem running"
  assert stemWord("be") == "be" and smartStemWord("be") == "be", "both keep short words"
  assert stemWord("") == "" and smartStemWord("") == "", "both keep empty"

  # Irregular-form lookup takes precedence over the no-change list.
  assert smartStemWord("means") == "mean", "irregular verb beats no-change word"
  assert smartStemWord("SHEEP") == "sheep", "no-change word is lowercased"

  # Batch helpers agree with their single-word counterparts. They must be
  # called through their own modules: see defect D6.
  assert porter.stemTokens(wordTokenize("running cities")) == @["run", "citi"],
    "porter batch"
  assert smart.stemTokens(wordTokenize("running")) == @["run"],
    "smart batch"

  echo "  ALL PASS"

# ============================================================
# Type edge cases
# ============================================================

proc testTypesEdges() =
  echo "=== Edge: types ==="

  # A non-finite component makes the norm infinite, so the cosine guard
  # collapses the result to 0.0 rather than propagating Inf or NaN.
  var a: SparseVector
  a.set(0, Inf)
  a.set(1, 1.0)
  var b: SparseVector
  b.set(0, 1.0)
  b.set(1, 1.0)
  assert cosineSimilarity(a, b) == 0.0, "infinite component yields 0.0"

  # set() ignores invalid indices and NaN, and removes explicit zeros.
  var v: SparseVector
  v.set(-1, 1.0)
  assert v.len == 0, "negative index ignored"
  v.set(0, NaN)
  assert v.len == 0, "NaN value ignored"
  v.set(0, 3.0)
  assert v.len == 1 and v.get(0) == 3.0, "value stored"
  v.set(0, 0.0)
  assert v.len == 0, "explicit zero removes the entry"

  # pairs() and values() stay aligned, and get() agrees with them.
  var s: SparseVector
  s.set(0, Inf)
  s.set(1, 1.0)
  let ps = s.pairs
  let vs = s.values
  assert ps.len == 2 and vs.len == 2, "pairs and values lengths"
  for i, (idx, value) in ps:
    assert vs[i] == value, "pairs and values stay aligned"
    assert s.get(idx) == value, "get agrees with pairs"

  # Vocabulary ids restart after clear(), and misses are sentinels.
  var vocab = newVocabulary()
  discard vocab.add("x")
  assert vocab.id("x") == 0, "first id"
  vocab.clear()
  assert vocab.len == 0, "clear empties"
  assert vocab.add("y") == 0, "ids restart after clear"
  assert vocab.id("missing") == -1, "missing id sentinel"
  assert vocab[999] == "", "out-of-range term sentinel"
  assert vocab[-1] == "", "negative out-of-range term sentinel"

  echo "  ALL PASS"

# ============================================================
# Main
# ============================================================

testTokenizerEdges()
testNormalizerEdges()
testRetrievalEdges()
testTopicEdges()
testStopwordEdges()
testStemmerDivergence()
testTypesEdges()
echo "\n=== ALL EDGE TESTS PASSED ==="
