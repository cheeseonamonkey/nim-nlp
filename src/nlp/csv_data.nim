## Small CSV adapter used for the embedded library data.
##
## Nim's standard library parser is used instead of a runtime file dependency.
## Callers only provide CSV text, so the data can be embedded with staticRead.

import std/[parsecsv, streams]

proc csvRows*(data, filename: string): seq[seq[string]] =
  var stream = newStringStream(data)
  var parser: CsvParser
  parser.open(stream, filename)
  while parser.readRow():
    result.add(parser.row)
  parser.close()
