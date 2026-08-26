## Comprehensive test suite for the nlp library.
##
## Tests cover:
## - Tokenizers (word, whitespace, ngram, sentence, regex)
## - Normalizers (lowercase, accent, whitespace, punctuation, number)
## - Stopwords (English, Spanish, French, German + custom)
## - Smart Stemmer (irregulars, plurals, verbs, suffixes)
## - Porter Stemmer
## - Analysis (frequency, similarity, stats, keywords)
## - Embeddings (TF-IDF, BM25)

import ../src/nlp
import std/[tables, sets]

# ============================================================
# Tokenizer edge cases
# ============================================================

proc testTokenizers() =
  echo "=== Tokenizers ==="

  # Word tokenizer - basic
  let t1 = wordTokenize("the quick brown fox")
  assert t1 == @["the", "quick", "brown", "fox"], "wordTokenize basic"

  # Word tokenizer - minLength
  let t2 = wordTokenize("the quick brown fox", minLength = 4)
  assert t2 == @["quick", "brown"], "wordTokenize minLength"

  # Word tokenizer - case preservation
  let t3 = wordTokenize("Hello World", lowercase = false)
  assert t3 == @["Hello", "World"], "wordTokenize case"

  # Word tokenizer - empty
  assert wordTokenize("") == @[], "wordTokenize empty"

  # Word tokenizer - only punctuation
  assert wordTokenize("!@#$%^&*()") == @[], "wordTokenize only punctuation"

  # Word tokenizer - single char
  assert wordTokenize("a") == @["a"], "wordTokenize single char"

  # Word tokenizer - unicode
  let t4 = wordTokenize("café résumé")
  assert "caf" in t4, "wordTokenize unicode handles accents"
  assert "r" in t4, "wordTokenize unicode partial"

  # Whitespace tokenizer
  let t5 = whitespaceTokenize("hello   world\tfoo")
  assert t5 == @["hello", "world", "foo"], "whitespaceTokenize"

  # Whitespace tokenizer - empty
  assert whitespaceTokenize("") == @[], "whitespaceTokenize empty"

  # Whitespace tokenizer - only whitespace
  assert whitespaceTokenize("   \t\n  ") == @[], "whitespaceTokenize only whitespace"

  # Whitespace tokenizer - minLength
  let t5b = whitespaceTokenize("a bb ccc dddd", minLength = 2)
  assert t5b == @["bb", "ccc", "dddd"], "whitespaceTokenize minLength"

  # N-gram tokenizer
  let t6 = ngramTokenize("hello", 2)
  assert t6 == @["he", "el", "ll", "lo"], "ngramTokenize"

  # N-gram tokenizer - edge cases
  assert ngramTokenize("", 2) == @[], "ngramTokenize empty"
  assert ngramTokenize("a", 2) == @[], "ngramTokenize short"
  assert ngramTokenize("hello", 0) == @[], "ngramTokenize n=0"
  assert ngramTokenize("hello", -1) == @[], "ngramTokenize n<0"
  assert ngramTokenize("hello", 10) == @[], "ngramTokenize n>len"
  assert ngramTokenize("éab", 2) == @["éa", "ab"], "ngramTokenize unicode characters"

  # Sentence tokenizer
  let t7 = splitSentences("Hello world. How are you? I'm fine!")
  assert t7 == @["Hello world.", "How are you?", "I'm fine!"], "splitSentences"

  # Sentence tokenizer - empty
  assert splitSentences("") == @[], "splitSentences empty"

  # Sentence tokenizer - no terminator
  assert splitSentences("no period here") == @["no period here"], "splitSentences no terminator"

  # Sentence tokenizer - multiple terminators
  let t7b = splitSentences("Wow! Really? Yes.")
  assert t7b == @["Wow!", "Really?", "Yes."], "splitSentences multiple"

  # Sentence tokenizer - don't discard empty
  let t7c = splitSentences("Hello. ", discardEmpty = false)
  assert "Hello." in t7c, "splitSentences keep content"

  # Pattern tokenizer
  let t8 = PatternTokenizer(pattern: r"\w+", lowercase: true, minLength: 1).tokenize("Hello, World!")
  assert t8 == @["hello", "world"], "patternTokenizer"

  # Pattern tokenizer - empty
  assert PatternTokenizer(pattern: r"\w+").tokenize("") == @[], "patternTokenizer empty"

  # Pattern tokenizer - no matches
  assert PatternTokenizer(pattern: r"\d+").tokenize("hello world") == @[], "patternTokenizer no match"

  # Pattern tokenizer - digits
  let t8a = PatternTokenizer(pattern: r"\d+").tokenize("abc 123 def 456")
  assert t8a == @["123", "456"], "patternTokenizer digits"

  # Pattern tokenizer - character class
  let t8b = PatternTokenizer(pattern: "[aeiou]").tokenize("hello")
  assert t8b == @["e", "o"], "patternTokenizer char class"

  # Word n-grams
  let t9 = wordTokenize("the quick brown fox").ngrams(2)
  assert t9 == @["the quick", "quick brown", "brown fox"], "word ngrams"

  # Word n-grams - edge cases
  assert wordTokenize("").ngrams(2) == @[], "word ngrams empty"
  assert wordTokenize("hello").ngrams(2) == @[], "word ngrams too short"
  assert wordTokenize("the quick").ngrams(0) == @[], "word ngrams n=0"
  assert wordTokenize("the quick").ngrams(-1) == @[], "word ngrams n<0"
  assert wordTokenize("the quick").ngrams(1) == @["the", "quick"], "word ngrams n=1"

  echo "  ALL PASS"

# ============================================================
# Normalizer edge cases
# ============================================================

proc testNormalizers() =
  echo "\n=== Normalizers ==="

  # Lowercase
  assert "Hello".normalizeLowercase() == "hello", "lowercase"
  assert "".normalizeLowercase() == "", "lowercase empty"
  assert "123".normalizeLowercase() == "123", "lowercase numbers"
  assert "!@#".normalizeLowercase() == "!@#", "lowercase punctuation"

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

  echo "  ALL PASS"

# ============================================================
# Stopwords edge cases
# ============================================================

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

# ============================================================
# Smart stemmer edge cases
# ============================================================

proc testSmartStemmer() =
  echo "\n=== Smart Stemmer ==="

  # --- Basic plurals ---
  assert smartStemWord("cats") == "cat", "cats -> cat"
  assert smartStemWord("dogs") == "dog", "dogs -> dog"
  assert smartStemWord("cars") == "car", "cars -> car"
  assert smartStemWord("houses") == "house", "houses -> house"

  # --- -ies pattern ---
  assert smartStemWord("cities") == "city", "cities -> city"
  assert smartStemWord("puppies") == "puppy", "puppies -> puppy"
  assert smartStemWord("stories") == "story", "stories -> story"
  assert smartStemWord("parties") == "party", "parties -> party"
  assert smartStemWord("ladies") == "lady", "ladies -> lady"

  # --- -ves pattern ---
  assert smartStemWord("wolves") == "wolf", "wolves -> wolf"
  assert smartStemWord("knives") == "knife", "knives -> knife"
  assert smartStemWord("leaves") == "leaf", "leaves -> leaf"
  assert smartStemWord("shelves") == "shelf", "shelves -> shelf"
  assert smartStemWord("thieves") == "thief", "thieves -> thief"
  assert smartStemWord("lives") == "life", "lives -> life"

  # --- -oes pattern ---
  assert smartStemWord("heroes") == "hero", "heroes -> hero"
  assert smartStemWord("potatoes") == "potato", "potatoes -> potato"
  assert smartStemWord("tomatoes") == "tomato", "tomatoes -> tomato"
  assert smartStemWord("echoes") == "echo", "echoes -> echo"
  assert smartStemWord("vetoes") == "veto", "vetoes -> veto"
  assert smartStemWord("torpedoes") == "torpedo", "torpedoes -> torpedo"

  # --- -ches/-shes/-xes/-zes/-sses ---
  assert smartStemWord("churches") == "church", "churches -> church"
  assert smartStemWord("matches") == "match", "matches -> match"
  assert smartStemWord("boxes") == "box", "boxes -> box"
  assert smartStemWord("foxes") == "fox", "foxes -> fox"
  assert smartStemWord("quizzes") == "quiz", "quizzes -> quiz"
  assert smartStemWord("buzzes") == "buzz", "buzzes -> buzz"
  assert smartStemWord("watches") == "watch", "watches -> watch"
  assert smartStemWord("dishes") == "dish", "dishes -> dish"
  assert smartStemWord("classes") == "class", "classes -> class"
  assert smartStemWord("kisses") == "kiss", "kisses -> kiss"

  # --- Latin/Greek irregulars ---
  assert smartStemWord("children") == "child", "children -> child"
  assert smartStemWord("feet") == "foot", "feet -> foot"
  assert smartStemWord("mice") == "mouse", "mice -> mouse"
  assert smartStemWord("geese") == "goose", "geese -> goose"
  assert smartStemWord("oxen") == "ox", "oxen -> ox"
  assert smartStemWord("people") == "person", "people -> person"
  assert smartStemWord("men") == "man", "men -> man"
  assert smartStemWord("women") == "woman", "women -> woman"
  assert smartStemWord("teeth") == "tooth", "teeth -> tooth"
  assert smartStemWord("analyses") == "analysis", "analyses -> analysis"
  assert smartStemWord("diagnoses") == "diagnosis", "diagnoses -> diagnosis"
  assert smartStemWord("hypotheses") == "hypothesis", "hypotheses -> hypothesis"
  assert smartStemWord("phenomena") == "phenomenon", "phenomena -> phenomenon"
  assert smartStemWord("criteria") == "criterion", "criteria -> criterion"
  assert smartStemWord("matrices") == "matrix", "matrices -> matrix"
  assert smartStemWord("vertices") == "vertex", "vertices -> vertex"
  assert smartStemWord("appendices") == "appendix", "appendices -> appendix"
  assert smartStemWord("indices") == "index", "indices -> index"
  assert smartStemWord("fungi") == "fungus", "fungi -> fungus"
  assert smartStemWord("nuclei") == "nucleus", "nuclei -> nucleus"
  assert smartStemWord("stimuli") == "stimulus", "stimuli -> stimulus"
  assert smartStemWord("data") == "datum", "data -> datum"
  assert smartStemWord("cacti") == "cactus", "cacti -> cactus"
  assert smartStemWord("alumni") == "alumnus", "alumni -> alumnus"
  assert smartStemWord("radii") == "radius", "radii -> radius"
  assert smartStemWord("crises") == "crisis", "crises -> crisis"
  assert smartStemWord("theses") == "thesis", "theses -> thesis"
  assert smartStemWord("symposia") == "symposium", "symposia -> symposium"
  assert smartStemWord("oases") == "oasis", "oases -> oasis"
  assert smartStemWord("corpora") == "corpus", "corpora -> corpus"
  assert smartStemWord("genera") == "genus", "genera -> genus"
  assert smartStemWord("spectra") == "spectrum", "spectra -> spectrum"
  assert smartStemWord("strata") == "stratum", "strata -> stratum"
  assert smartStemWord("larvae") == "larva", "larvae -> larva"
  assert smartStemWord("nebulae") == "nebula", "nebulae -> nebula"
  assert smartStemWord("vertebrae") == "vertebra", "vertebrae -> vertebra"
  assert smartStemWord("vitae") == "vita", "vitae -> vita"
  assert smartStemWord("ellipses") == "ellipsis", "ellipses -> ellipsis"
  assert smartStemWord("paralyses") == "paralysis", "paralyses -> paralysis"
  assert smartStemWord("syntheses") == "synthesis", "syntheses -> synthesis"
  assert smartStemWord("synopses") == "synopsis", "synopses -> synopsis"
  assert smartStemWord("nemeses") == "nemesis", "nemeses -> nemesis"
  assert smartStemWord("neuroses") == "neurosis", "neuroses -> neurosis"
  assert smartStemWord("metamorphoses") == "metamorphosis", "metamorphoses -> metamorphosis"
  assert smartStemWord("memoranda") == "memorandum", "memoranda -> memorandum"
  assert smartStemWord("quanta") == "quantum", "quanta -> quantum"
  assert smartStemWord("referenda") == "referendum", "referenda -> referendum"
  assert smartStemWord("errata") == "erratum", "errata -> erratum"
  assert smartStemWord("stadia") == "stadium", "stadia -> stadium"
  assert smartStemWord("media") == "medium", "media -> medium"
  assert smartStemWord("millennia") == "millennium", "millennia -> millennium"
  assert smartStemWord("curricula") == "curriculum", "curricula -> curriculum"
  assert smartStemWord("symposia") == "symposium", "symposia -> symposium"
  assert smartStemWord("octopi") == "octopus", "octopi -> octopus"

  # --- Irregular verbs ---
  assert smartStemWord("was") == "be", "was -> be"
  assert smartStemWord("were") == "be", "were -> be"
  assert smartStemWord("been") == "be", "been -> be"
  assert smartStemWord("being") == "be", "being -> be"
  assert smartStemWord("began") == "begin", "began -> begin"
  assert smartStemWord("begun") == "begin", "begun -> begin"
  assert smartStemWord("broke") == "break", "broke -> break"
  assert smartStemWord("broken") == "break", "broken -> break"
  assert smartStemWord("chose") == "choose", "chose -> choose"
  assert smartStemWord("chosen") == "choose", "chosen -> choose"
  assert smartStemWord("came") == "come", "came -> come"
  assert smartStemWord("did") == "do", "did -> do"
  assert smartStemWord("done") == "do", "done -> do"
  assert smartStemWord("doing") == "do", "doing -> do"
  assert smartStemWord("drove") == "drive", "drove -> drive"
  assert smartStemWord("driven") == "drive", "driven -> drive"
  assert smartStemWord("driving") == "drive", "driving -> drive"
  assert smartStemWord("ate") == "eat", "ate -> eat"
  assert smartStemWord("eaten") == "eat", "eaten -> eat"
  assert smartStemWord("eating") == "eat", "eating -> eat"
  assert smartStemWord("fell") == "fall", "fell -> fall"
  assert smartStemWord("fallen") == "fall", "fallen -> fall"
  assert smartStemWord("found") == "find", "found -> find"
  assert smartStemWord("gave") == "give", "gave -> give"
  assert smartStemWord("given") == "give", "given -> give"
  assert smartStemWord("giving") == "give", "giving -> give"
  assert smartStemWord("went") == "go", "went -> go"
  assert smartStemWord("gone") == "go", "gone -> go"
  assert smartStemWord("goes") == "go", "goes -> go"
  assert smartStemWord("going") == "go", "going -> go"
  assert smartStemWord("grew") == "grow", "grew -> grow"
  assert smartStemWord("grown") == "grow", "grown -> grow"
  assert smartStemWord("had") == "have", "had -> have"
  assert smartStemWord("has") == "have", "has -> have"
  assert smartStemWord("having") == "have", "having -> have"
  assert smartStemWord("knew") == "know", "knew -> know"
  assert smartStemWord("known") == "know", "known -> know"
  assert smartStemWord("made") == "make", "made -> make"
  assert smartStemWord("makes") == "make", "makes -> make"
  assert smartStemWord("making") == "make", "making -> make"
  assert smartStemWord("ran") == "run", "ran -> run"
  assert smartStemWord("runs") == "run", "runs -> run"
  assert smartStemWord("running") == "run", "running -> run"
  assert smartStemWord("said") == "say", "said -> say"
  assert smartStemWord("says") == "say", "says -> say"
  assert smartStemWord("saying") == "say", "saying -> say"
  assert smartStemWord("saw") == "see", "saw -> see"
  assert smartStemWord("seen") == "see", "seen -> see"
  assert smartStemWord("spoke") == "speak", "spoke -> speak"
  assert smartStemWord("spoken") == "speak", "spoken -> speak"
  assert smartStemWord("took") == "take", "took -> take"
  assert smartStemWord("taken") == "take", "taken -> take"
  assert smartStemWord("takes") == "take", "takes -> take"
  assert smartStemWord("taking") == "take", "taking -> take"
  assert smartStemWord("thought") == "think", "thought -> think"
  assert smartStemWord("thinks") == "think", "thinks -> think"
  assert smartStemWord("thinking") == "think", "thinking -> think"
  assert smartStemWord("threw") == "throw", "threw -> throw"
  assert smartStemWord("thrown") == "throw", "thrown -> throw"
  assert smartStemWord("wrote") == "write", "wrote -> write"
  assert smartStemWord("written") == "write", "written -> write"
  assert smartStemWord("writes") == "write", "writes -> write"
  assert smartStemWord("writing") == "write", "writing -> write"
  assert smartStemWord("swam") == "swim", "swam -> swim"
  assert smartStemWord("swum") == "swim", "swum -> swim"
  assert smartStemWord("swims") == "swim", "swims -> swim"
  assert smartStemWord("swimming") == "swim", "swimming -> swim"
  assert smartStemWord("bought") == "buy", "bought -> buy"
  assert smartStemWord("buys") == "buy", "buys -> buy"
  assert smartStemWord("buying") == "buy", "buying -> buy"
  assert smartStemWord("brought") == "bring", "brought -> bring"
  assert smartStemWord("brings") == "bring", "brings -> bring"
  assert smartStemWord("bringing") == "bring", "bringing -> bring"
  assert smartStemWord("caught") == "catch", "caught -> catch"
  assert smartStemWord("catches") == "catch", "catches -> catch"
  assert smartStemWord("catching") == "catch", "catching -> catch"
  assert smartStemWord("fought") == "fight", "fought -> fight"
  assert smartStemWord("fights") == "fight", "fights -> fight"
  assert smartStemWord("fighting") == "fight", "fighting -> fight"
  assert smartStemWord("taught") == "teach", "taught -> teach"
  assert smartStemWord("teaches") == "teach", "teaches -> teach"
  assert smartStemWord("teaching") == "teach", "teaching -> teach"
  assert smartStemWord("sang") == "sing", "sang -> sing"
  assert smartStemWord("sung") == "sing", "sung -> sing"
  assert smartStemWord("sings") == "sing", "sings -> sing"
  assert smartStemWord("singing") == "sing", "singing -> sing"
  assert smartStemWord("drank") == "drink", "drank -> drink"
  assert smartStemWord("drunk") == "drink", "drunk -> drink"
  assert smartStemWord("drinks") == "drink", "drinks -> drink"
  assert smartStemWord("drinking") == "drink", "drinking -> drink"
  assert smartStemWord("stole") == "steal", "stole -> steal"
  assert smartStemWord("stolen") == "steal", "stolen -> steal"
  assert smartStemWord("steals") == "steal", "steals -> steal"
  assert smartStemWord("stealing") == "steal", "stealing -> steal"
  assert smartStemWord("froze") == "freeze", "froze -> freeze"
  assert smartStemWord("frozen") == "freeze", "frozen -> freeze"
  assert smartStemWord("freezes") == "freeze", "freezes -> freeze"
  assert smartStemWord("freezing") == "freeze", "freezing -> freeze"
  assert smartStemWord("held") == "hold", "held -> hold"
  assert smartStemWord("holds") == "hold", "holds -> hold"
  assert smartStemWord("holding") == "hold", "holding -> hold"
  assert smartStemWord("kept") == "keep", "kept -> keep"
  assert smartStemWord("keeps") == "keep", "keeps -> keep"
  assert smartStemWord("keeping") == "keep", "keeping -> keep"
  assert smartStemWord("left") == "leave", "left -> leave"
  assert smartStemWord("leaving") == "leave", "leaving -> leave"
  assert smartStemWord("lost") == "lose", "lost -> lose"
  assert smartStemWord("loses") == "lose", "loses -> lose"
  assert smartStemWord("losing") == "lose", "losing -> lose"
  assert smartStemWord("meant") == "mean", "meant -> mean"
  assert smartStemWord("means") == "mean", "means -> mean"
  assert smartStemWord("meaning") == "mean", "meaning -> mean"
  assert smartStemWord("met") == "meet", "met -> meet"
  assert smartStemWord("meets") == "meet", "meets -> meet"
  assert smartStemWord("meeting") == "meet", "meeting -> meet"
  assert smartStemWord("sat") == "sit", "sat -> sit"
  assert smartStemWord("sits") == "sit", "sits -> sit"
  assert smartStemWord("sitting") == "sit", "sitting -> sit"
  assert smartStemWord("slept") == "sleep", "slept -> sleep"
  assert smartStemWord("sleeps") == "sleep", "sleeps -> sleep"
  assert smartStemWord("sleeping") == "sleep", "sleeping -> sleep"
  assert smartStemWord("spent") == "spend", "spent -> spend"
  assert smartStemWord("spends") == "spend", "spends -> spend"
  assert smartStemWord("spending") == "spend", "spending -> spend"
  assert smartStemWord("stood") == "stand", "stood -> stand"
  assert smartStemWord("stands") == "stand", "stands -> stand"
  assert smartStemWord("standing") == "stand", "standing -> stand"
  assert smartStemWord("swept") == "sweep", "swept -> sweep"
  assert smartStemWord("sweeps") == "sweep", "sweeps -> sweep"
  assert smartStemWord("sweeping") == "sweep", "sweeping -> sweep"
  assert smartStemWord("swung") == "swing", "swung -> swing"
  assert smartStemWord("swings") == "swing", "swings -> swing"
  assert smartStemWord("swinging") == "swing", "swinging -> swing"
  assert smartStemWord("won") == "win", "won -> win"
  assert smartStemWord("wins") == "win", "wins -> win"
  assert smartStemWord("winning") == "win", "winning -> win"
  assert smartStemWord("understood") == "understand", "understood -> understand"
  assert smartStemWord("understands") == "understand", "understands -> understand"
  assert smartStemWord("understanding") == "understand", "understanding -> understand"

  # --- Words that don't change ---
  assert smartStemWord("sheep") == "sheep", "sheep -> sheep"
  assert smartStemWord("deer") == "deer", "deer -> deer"
  assert smartStemWord("fish") == "fish", "fish -> fish"
  assert smartStemWord("moose") == "moose", "moose -> moose"
  assert smartStemWord("trout") == "trout", "trout -> trout"
  assert smartStemWord("salmon") == "salmon", "salmon -> salmon"
  assert smartStemWord("series") == "series", "series -> series"
  assert smartStemWord("species") == "species", "species -> species"
  assert smartStemWord("means") == "mean", "means -> mean (verb takes precedence)"

  # --- Edge cases ---
  assert smartStemWord("") == "", "empty"
  assert smartStemWord("a") == "a", "single char"
  assert smartStemWord("I") == "i", "single char uppercase"
  assert smartStemWord("running") == "run", "running -> run"
  assert smartStemWord("jumps") == "jump", "jumps -> jump"
  assert smartStemWord("talking") == "talk", "talking -> talk"
  assert smartStemWord("talked") == "talk", "talked -> talk"
  assert smartStemWord("played") == "play", "played -> play"

  echo "  ALL PASS"

# ============================================================
# Porter stemmer tests
# ============================================================

proc testPorterStemmer() =
  echo "\n=== Porter Stemmer ==="

  # Basic
  assert stemWord("running") == "run", "porter running"
  assert stemWord("flies") == "fli", "porter flies"
  assert stemWord("denied") == "deni", "porter denied"

  # Edge cases
  assert stemWord("") == "", "porter empty"
  assert stemWord("a") == "a", "porter single char"
  assert stemWord("ab") == "ab", "porter two chars"
  assert stemWord("ABC") == "abc", "porter uppercase"

  # Common words
  assert stemWord("relational") == "rel", "porter relational"
  assert stemWord("conditional") == "condit", "porter conditional"
  assert stemWord("national") == "nat", "porter national"
  assert stemWord("rational") == "rat", "porter rational"
  assert stemWord("nationality") == "nation", "porter nationality"
  assert stemWord("rationality") == "ration", "porter rationality"

  echo "  ALL PASS"

# ============================================================
# Analysis tests
# ============================================================

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

  echo "  ALL PASS"

# ============================================================
# Embeddings tests
# ============================================================

proc testEmbeddings() =
  echo "\n=== Embeddings ==="

  let docs = @[
    wordTokenize("the cat sat on the mat"),
    wordTokenize("the dog sat on the log"),
    wordTokenize("cats and dogs are great pets"),
    wordTokenize("the quick brown fox jumps over the lazy dog"),
    wordTokenize("a journey of a thousand miles begins with a single step")
  ]

  # TF-IDF
  var tfidf = fitTfidf(docs)
  assert tfidf.numDocs == 5, "tfidf numDocs"
  assert tfidf.vocabulary.len > 0, "tfidf vocab"

  # TF-IDF transform
  let vec = tfidf.transformTfidf(wordTokenize("the cat is on the mat"))
  assert vec.len > 0, "tfidf transform"

  # TF-IDF empty transform
  let emptyVec = tfidf.transformTfidf(@[])
  assert emptyVec.len == 0, "tfidf transform empty"

  # TF-IDF search
  let results = tfidf.searchTfidf(wordTokenize("cat mat"), topK = 3)
  assert results.len > 0, "tfidf search"
  assert results[0][0] == 0, "tfidf search ranking"

  # TF-IDF empty search
  assert tfidf.searchTfidf(@[]).len == 0, "tfidf search empty"
  assert tfidf.searchTfidf(wordTokenize("cat"), topK = 0).len == 0, "tfidf search topK=0"

  # BM25
  var bm25 = fitBm25(docs)
  assert bm25.numDocs == 5, "bm25 numDocs"

  let bm25Results = bm25.searchBm25(wordTokenize("cats pets"), topK = 3)
  assert bm25Results.len > 0, "bm25 search"
  let once = bm25.searchBm25(@["cat"])
  let twice = bm25.searchBm25(@["cat", "cat"])
  assert twice[0][1] > once[0][1], "bm25 query frequency"

  # BM25 empty search
  assert bm25.searchBm25(@[]).len == 0, "bm25 search empty"
  assert bm25.searchBm25(wordTokenize("cat"), topK = 0).len == 0, "bm25 search topK=0"
  var invalidBm25Raised = false
  try:
    discard bm25.searchBm25(@["cat"], b = 1.5)
  except ValueError:
    invalidBm25Raised = true
  assert invalidBm25Raised, "bm25 parameter validation"

  # Empty model
  var emptyTfidf = fitTfidf(@[])
  assert emptyTfidf.numDocs == 0, "tfidf empty model"
  assert emptyTfidf.searchTfidf(wordTokenize("cat")).len == 0, "tfidf empty model search"

  var emptyBm25 = fitBm25(@[])
  assert emptyBm25.numDocs == 0, "bm25 empty model"
  assert emptyBm25.searchBm25(wordTokenize("cat")).len == 0, "bm25 empty model search"

  # Model with empty docs
  var sparseTfidf = fitTfidf(@[@[], @["hello"], @[]])
  assert sparseTfidf.numDocs == 3, "tfidf sparse docs"
  assert sparseTfidf.vocabulary.len == 1, "tfidf sparse vocab"

  echo "  ALL PASS"

# ============================================================
# Types / Vocabulary tests
# ============================================================

proc testTypes() =
  echo "\n=== Types ==="

  # SparseVector
  var v = newSparseVector()
  v.set(0, 1.5)
  v.set(1, 2.5)
  assert v.get(0) == 1.5, "sparse set/get"
  assert v.get(1) == 2.5, "sparse set/get 2"
  assert v.get(999) == 0.0, "sparse get missing"
  assert v.len == 2, "sparse len"

  # SparseVector - clear
  v.clear()
  assert v.len == 0, "sparse clear"

  # SparseVector - containsIdx
  v.set(5, 1.0)
  assert v.containsIdx(5), "sparse containsIdx true"
  assert not v.containsIdx(3), "sparse containsIdx false"

  # SparseVector - cosine with empty
  let a = newSparseVector()
  let b = newSparseVector()
  assert cosineSimilarity(a, b) == 0.0, "cosine empty vectors"

  # SparseVector - cosine one empty
  var c = newSparseVector()
  c.set(0, 1.0)
  assert cosineSimilarity(c, b) == 0.0, "cosine one empty"

  # Vocabulary
  var vocab = newVocabulary()
  let id1 = vocab.add("hello")
  let id2 = vocab.add("world")
  assert id2 != id1, "vocab different ids"
  let id3 = vocab.add("hello")
  assert id1 == id3, "vocab dedup"
  assert vocab.len == 2, "vocab len"
  assert vocab.id("hello") == id1, "vocab lookup"
  assert vocab[id1] == "hello", "vocab reverse lookup"

  # Vocabulary - bounds checking
  assert vocab[-1] == "", "vocab out of bounds negative"
  assert vocab[999] == "", "vocab out of bounds positive"

  # Vocabulary - contains
  assert vocab.contains("hello"), "vocab contains true"
  assert not vocab.contains("nonexistent"), "vocab contains false"

  # Vocabulary - containsIdx
  assert vocab.containsIdx(0), "vocab containsIdx true"
  assert vocab.containsIdx(1), "vocab containsIdx true 2"
  assert not vocab.containsIdx(-1), "vocab containsIdx negative"
  assert not vocab.containsIdx(999), "vocab containsIdx positive"

  # Vocabulary - clear
  vocab.clear()
  assert vocab.len == 0, "vocab clear"

  echo "  ALL PASS"

# ============================================================
# Main
# ============================================================

testTokenizers()
testNormalizers()
testStopwords()
testPorterStemmer()
testSmartStemmer()
testAnalysis()
testEmbeddings()
testTypes()
echo "\n=== ALL TESTS PASSED ==="
