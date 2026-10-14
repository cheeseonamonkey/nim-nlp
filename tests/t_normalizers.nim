## Normalizer tests for the nlp library.

import ../src/nlp
import std/[tables, sets]

proc checkNorm[N: Normalizer](n: N): string =
  ## Direct concept-conformance probe: any Normalizer typechecks here.
  normalize(n, "Hi!")

proc testNormalizers() =
  echo "\n=== Normalizers ==="

  # Lowercase
  assert "Hello".normalizeLowercase() == "hello", "lowercase"
  assert "".normalizeLowercase() == "", "lowercase empty"
  assert "123".normalizeLowercase() == "123", "lowercase numbers"
  assert "!@#".normalizeLowercase() == "!@#", "lowercase punctuation"
  assert "CAFÉ".normalizeLowercase() == "café", "lowercase unicode"

  # Accent strip
  let accentResult = "café naïve résumé".normalizeStripAccents()
  assert accentResult == "cafe naive resume", "accents: got " & accentResult
  assert "".normalizeStripAccents() == "", "accents empty"
  assert "abc".normalizeStripAccents() == "abc", "accents no accents"

  # Whitespace
  assert "  hello   world  ".normalizeWhitespace() == "hello world", "whitespace"
  assert "".normalizeWhitespace() == "", "whitespace empty"
  assert "   ".normalizeWhitespace() == "", "whitespace only spaces"
  assert "hello".normalizeWhitespace() == "hello", "whitespace no extra"

  # Punctuation
  assert "Hello, world!".normalizeStripPunctuation() == "Hello world", "punctuation"
  assert "".normalizeStripPunctuation() == "", "punctuation empty"
  assert "abc".normalizeStripPunctuation() == "abc", "punctuation no punct"
  assert "!@#$%^&*()".normalizeStripPunctuation() == "", "punctuation only punct"

  # Number normalizer
  assert normalizeRemoveNumbers("abc123def456") == "abcdef", "numbers remove"
  assert normalizeRemoveNumbers("abc123def456", "NUM") == "abcNUMdefNUM", "numbers replace"
  assert normalizeRemoveNumbers("") == "", "numbers empty"
  assert normalizeRemoveNumbers("abc") == "abc", "numbers no digits"
  assert normalizeRemoveNumbers("123") == "", "numbers only digits"

  # Chained normalizer
  let norm = chain(proc(text: string): string = Lowercaser().normalize(text), proc(text: string): string = PunctuationStripper().normalize(text))
  assert norm("Hello, World!") == "hello world", "chained"

  # Chained normalizer - empty
  assert norm("") == "", "chained empty"

  # Contraction expansion
  assert normalizeExpandContractions("I don't like it, can't you see?") == "I do not like it, cannot you see?", "contractions sentence"
  assert normalizeExpandContractions("the dog's bone") == "the dog's bone", "contractions possessive untouched"
  assert normalizeExpandContractions("meet at 5 o'clock") == "meet at 5 o'clock", "contractions o'clock untouched"
  assert normalizeExpandContractions("Don\xE2\x80\x99t stop believin'") == "Do not stop believin'", "contractions curly apostrophe"
  assert normalizeExpandContractions("Don't stop") == "Do not stop", "contractions leading caps"
  assert normalizeExpandContractions("DON'T PANIC") == "Do not PANIC", "contractions all caps"
  assert normalizeExpandContractions("it's raining") == "it is raining", "contractions its-is"
  assert normalizeExpandContractions("(can't) won't.") == "(cannot) will not.", "contractions punctuation boundaries"
  assert normalizeExpandContractions("") == "", "contractions empty"
  assert ContractionExpander().normalize("y'all let's go") == "you all let us go", "contractions object api"
  let cchain = chain(proc(text: string): string = normalizeExpandContractions(text), proc(text: string): string = Lowercaser().normalize(text))
  assert cchain("DON'T Stop") == "do not stop", "contractions chained"

  # Leading-apostrophe forms expand; possessives and clock words still pass through.
  assert normalizeExpandContractions("'tis the season") == "it is the season", "contractions leading 'tis"
  assert normalizeExpandContractions("'twas brillig") == "it was brillig", "contractions leading 'twas"
  assert normalizeExpandContractions("take 'em all") == "take them all", "contractions leading 'em"
  assert normalizeExpandContractions("'til morning") == "until morning", "contractions leading 'til"
  assert normalizeExpandContractions("'cause I said so") == "because I said so", "contractions leading 'cause"
  assert normalizeExpandContractions("'round the corner") == "around the corner", "contractions leading 'round"
  assert normalizeExpandContractions("dogs' bones") == "dogs' bones", "contractions trailing possessive untouched"
  assert normalizeExpandContractions("'hi' there") == "'hi' there", "contractions quoted word untouched"

  # Normalizer concept conformance (each distinct type satisfies the concept)
  assert checkNorm(Lowercaser()) == "hi!", "concept lowercaser"
  assert checkNorm(PunctuationStripper()) == "Hi", "concept punct stripper"
  assert checkNorm(WhitespaceNormalizer()) == "Hi!", "concept whitespace"
  assert checkNorm(ContractionExpander()) == "Hi!", "concept contractions"
  assert checkNorm(NumberNormalizer(replaceWith: "")) == "Hi!", "concept numbers"
  assert checkNorm(HtmlStripper()) == "Hi!", "concept html"
  assert checkNorm(UrlReplacer()) == "Hi!", "concept url"
  assert checkNorm(EmailReplacer()) == "Hi!", "concept email"
  assert checkNorm(TextPipeline()) == "Hi!", "concept pipeline"

  # TextPipeline composition
  var pEmpty = TextPipeline()
  assert pEmpty.normalize("Hello, DON'T.") == "Hello, DON'T.", "pipeline empty returns input"

  # Order matters: expand-then-strip vs strip-then-expand
  var orderA = TextPipeline()
  orderA.addStep(ContractionExpander())
  orderA.addStep(PunctuationStripper())
  var orderB = TextPipeline()
  orderB.addStep(PunctuationStripper())
  orderB.addStep(ContractionExpander())
  assert orderA.normalize("can't.") == "cannot", "pipeline expand-then-strip"
  assert orderB.normalize("can't.") == "cant", "pipeline strip-then-expand"
  assert orderA.normalize("can't.") != orderB.normalize("can't."), "pipeline order matters"

  # Nesting: pipeline inside pipeline
  var inner = TextPipeline()
  inner.addStep(Lowercaser())
  inner.addStep(WhitespaceNormalizer())
  var outer = TextPipeline()
  outer.addStep(inner)
  outer.addStep(PunctuationStripper())
  assert outer.normalize("  HELLO,   WORLD!  ") == "hello world", "pipeline nesting"

  # Mixed object-type + raw-proc steps
  var mixed = TextPipeline()
  mixed.addStep(Lowercaser())
  mixed.addStep(proc(t: string): string = "<" & t & ">")
  assert mixed.normalize("Hi") == "<hi>", "pipeline mixed proc step"

  # Several distinct normalizer types through the concept overload
  var multi = TextPipeline()
  multi.addStep(Lowercaser())
  multi.addStep(ContractionExpander())
  multi.addStep(NumberNormalizer(replaceWith: ""))
  multi.addStep(WhitespaceNormalizer())
  multi.addStep(PunctuationStripper())
  assert multi.normalize("  I DON'T have 2 dogs, can't you?  ") == "i do not have dogs cannot you", "pipeline multi-type"

  # Extracted-helper behavior pins (email/url/html scanners)
  assert EmailReplacer().normalize("mail me at a@b.com.") == "mail me at <email>.", "email trailing dot trimmed"
  assert EmailReplacer().normalize("a@b no mail") == "a@b no mail", "email dotless untouched"
  assert UrlReplacer().normalize("see https://x.io!") == "see <url>!", "url bang trimmed"
  assert UrlReplacer().normalize("go to www.a.com.") == "go to <url>.", "url dot trimmed"
  assert HtmlStripper().normalize("<a title=\"a>b\">x</a>") == "x", "html quoted gt kept in tag"
  assert HtmlStripper().normalize("a < b") == "a < b", "html bare lt kept"

  echo "  ALL PASS"

when isMainModule:
  testNormalizers()
