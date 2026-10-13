#!/usr/bin/env bash
# Install the local package into a throwaway NIMBLE_DIR and exercise it from a
# standalone consumer, with the repository kept off the import path.
#
# nimble 0.22.2 does not accept a bare path argument for `nimble install`
# (it treats /path/to/repo as a package name), so the install is run from
# inside the package directory instead:
#     ( cd "$REPO" && NIMBLE_DIR="$TMP" nimble install --accept )

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Installing nlp into isolated NIMBLE_DIR: $TMP"
# Seed the package list from the machine's cache so the install needs no
# network access; a fresh NIMBLE_DIR would otherwise prompt and download it.
if [ -f "$HOME/.nimble/packages_official.json" ]; then
  cp "$HOME/.nimble/packages_official.json" "$TMP/packages_official.json"
fi
( cd "$REPO" && NIMBLE_DIR="$TMP" nimble install --accept )

# Regression guard: bundled data must not leak to a top-level pkgs2/data.
if [ -e "$TMP/pkgs2/data" ]; then
  echo "FAIL: leaked data directory at $TMP/pkgs2/data" >&2
  exit 1
fi

PKG="$(NIMBLE_DIR="$TMP" nimble path nlp | tail -n 1)"
echo "==> Installed package: $PKG"

cat > "$TMP/consumer.nim" <<'NIM'
import nlp

doAssert isStopWord("the", "english"), "isStopWord"

let filtered = @["the", "wolves"].removeStopWords("english")
doAssert "the" notin filtered, "removeStopWords drops stopword"
doAssert "wolves" in filtered, "removeStopWords keeps content word"

doAssert smartStemWord("running") == "run", "smartStemWord"
doAssert stemWord("running") == "run", "stemWord"

let docs = @[
  @["the", "cat", "sat", "on", "the", "mat"],
  @["the", "dog", "sat", "on", "the", "log"]
]

doAssert fitTfidf(docs).searchTfidf(@["cat"]).len > 0, "searchTfidf"
doAssert fitBm25(docs).searchBm25(@["cat"]).len > 0, "searchBm25"

let lda = fitLda(docs, numTopics = 2, iterations = 20, seed = 7)
let theta = lda.transformLda(@["cat", "sat"], iterations = 10)
var total = 0.0
for p in theta: total += p
doAssert abs(total - 1.0) < 1e-3, "transformLda sums to 1"

echo "consumer ok"
NIM

echo "==> Compiling and running consumer against the installed package only"
# `--noNimblePath` keeps any globally installed nlp out of the import path so
# this really exercises the package produced in $TMP.
( cd "$TMP" && nim c --hints:off --noNimblePath --path:"$PKG" -r consumer.nim )
