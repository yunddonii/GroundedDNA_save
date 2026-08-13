"""Read one field out of a run's args.txt (Phase 2 helper).

`args.txt` is `name----value` with dashes padding to a fixed width, so a plain
grep for the name also matches every field whose name contains it.
"""
import re
import sys

path, field = sys.argv[1:3]
pattern = re.compile(rf"^{re.escape(field)}-+(.*)$")
for line in open(path, encoding="utf-8"):
    match = pattern.match(line.rstrip("\n"))
    if match:
        print(match.group(1).strip())
        break
else:
    raise SystemExit(f"{path} has no field {field!r}")
