## Edge-case and defect-regression tests for the nlp library.
##
## Two kinds of assertions live here:
##
## 1. Edge cases whose current behavior is correct (or at least defensible)
##    and worth freezing, so a refactor cannot silently change them.
##
## All recorded defects D1-D6 are now fixed; every assertion here tests
## corrected behavior, so any regression resurfaces immediately.
## Historically: entries marked `BUG(PINNED)` asserted the WRONG output on
## purpose so a fix would break exactly one visible assertion. Fixed entries
## keep their `BUG(PINNED)` prose in the sections below for context.
##
## Known defects recorded here:
##
##   D1  (fixed in 0.2.1) BM25's probabilistic IDF collapsed to zero for
##       terms present in about half the corpus or more, and the `score > 0`
##       filter then silently dropped every hit — searching for a common word
##       returned nothing at all. The weight is now shifted by +1 inside the
##       logarithm so it stays strictly positive. The retrieval section below
##       asserts the corrected behavior.
##   D2  (fixed in 0.2.2) `normalizeReplaceEmails("a@.")` used to yield
##       "<email>." although "a@" is not an email: the trailing-dot trim
##       removed the very dot that satisfied the dot-in-domain requirement,
##       and the span was reported anyway. The domain must now still contain
##       a dot separating a non-empty host from a non-empty TLD *after* the
##       trim. The normalizer section below asserts the corrected behavior.
##   D3  (fixed in 0.2.2) `normalizeStripHtml` used to stop at the first '>'
##       even inside a quoted attribute, so `<a href="x>y">z</a>` became
##       `y">z`. A '>' inside a quoted attribute value is now literal text
##       until the matching quote closes.
##   D4  (resolved in 0.2.2) The custom-stopword overload's doc comment
##       claimed case-insensitive matching, but the caller's set is used
##       verbatim while only the tokens are lowercased. Decision: keep the
##       behavior (lowercasing the set per call would allocate on the hot
##       path) and fix the docs instead — the set must contain lowercase
##       entries, tokens are lowercased before comparison.
##   D5  (fixed in 0.2.2) `transformLda` used to return an all-zero vector
##       for an empty document (a "distribution" summing to 0) but a uniform
##       vector for an out-of-vocabulary document. Both no-evidence inputs
##       now return the uniform prior (sums to 1).
##   D6  (fixed in 0.2.1) `stemTokens` was exported by both stemmers with the
##       same signature and `nlp.nim` re-exported both, so
##       `import nlp; stemTokens(tokens)` was an ambiguous call that failed to
##       compile. The smart variant is now `smartStemTokens` (the name the
##       README already documented), leaving `stemTokens` unambiguously
##       Porter's.
##
## The stemmer pins at the bottom are not defects: the two stemmers
## intentionally disagree, and the pins protect that disagreement from being
## "cleaned up" by a refactor.

import ../src/nlp
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

  # D3 (fixed in 0.2.2): a '>' inside a quoted attribute value is literal
  # text until the matching quote closes — double or single quotes.
  assert normalizeStripHtml("<a href=\"x>y\">z</a>") == "z",
    "D3 fixed: quoted '>' no longer truncates the tag"
  assert normalizeStripHtml("<a href='x>y'>z</a>") == "z",
    "D3 fixed: single-quoted '>' no longer truncates the tag"
  assert normalizeStripHtml("<a title=\"a>b\" href=\"c>d\">ok</a>") == "ok",
    "multiple quoted '>' attributes survive"
  assert normalizeStripHtml("unclosed <a href=\"x>y") == "unclosed <a href=\"x>y",
    "quote opened but tag never closed"

  # Email spans; the simple heuristic needs a dot in the domain.
  assert normalizeReplaceEmails("a@b") == "a@b", "email needs a dotted domain"
  assert normalizeReplaceEmails("a@") == "a@", "email needs a domain"
  assert normalizeReplaceEmails("a@b.c") == "<email>", "short dotted domain matches"
  # D2 (fixed in 0.2.2): the domain must still be host.tld *after* the
  # trailing-dot trim, so "a@." is literal text now.
  assert normalizeReplaceEmails("a@.") == "a@.",
    "D2 fixed: 'a@.' is not an email"
  assert normalizeReplaceEmails("user@x.org.") == "<email>.",
    "D2 fixed: trailing sentence dot stays outside the span"
  assert normalizeReplaceEmails("write to a@b.org.") == "write to <email>.",
    "trailing sentence dot after a real email"
  assert normalizeReplaceEmails("mail me (a@b.c) now") == "mail me (<email>) now",
    "span works through surrounding punctuation"
  assert normalizeReplaceEmails("a@b.") == "a@b.",
    "dot-only domain with empty host is not an email"

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

  # D5 (fixed in 0.2.2): both no-evidence inputs now return the uniform
  # prior, a valid distribution summing to 1.
  let emptyDoc = transformLda(m, newSeq[string]())
  assert emptyDoc.len == 2, "empty fold-in length"
  assert abs(emptyDoc[0] - 0.5) < 1e-9 and abs(emptyDoc[1] - 0.5) < 1e-9 and
    abs(sum(emptyDoc) - 1.0) < 1e-9,
    "D5 fixed: empty doc folds in as a no-evidence uniform distribution"

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

  # D4 (resolved in 0.2.2): the doc now states the set must be lowercase.
  # An uppercase set never matches — this is documented behavior, not a bug.
  assert wordTokenize("the and cat").removeStopWords(toHashSet(["THE", "And"])) ==
    @["the", "and", "cat"],
    "D4: uppercase custom set matches nothing (set must be lowercase)"
  # Tokens themselves are lowercased before comparison, so mixed-case tokens
  # still match a lowercase set, and casing is preserved in the output.
  assert @["The", "AND", "Cat"].removeStopWords(toHashSet(["the", "and"])) ==
    @["Cat"], "mixed-case tokens match a lowercase set"

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

  # Batch helpers are reachable from the facade without qualification, and
  # the two stemmers keep separate names (defect D6).
  assert stemTokens(wordTokenize("running cities")) == @["run", "citi"],
    "porter batch"
  assert smartStemTokens(wordTokenize("running")) == @["run"],
    "smart batch"
  assert smartStemTokens(wordTokenize("cities")) == @["city"],
    "smart batch resolves plurals like the single-word helper"

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
