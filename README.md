# valve.py
VALVE bindings for Python

[lib]
name = "valve"
crate-type = ["cdylib", "lib"]

[dependencies.pyo3]
version = "0.16.5"
# "extension-module" tells pyo3 we want to build an extension module (skips linking against libpython.so)
features = ["extension-module"]

NEED TO ALSO ADD THIS TO [dependencies]:

futures = "0.3"

Makefile:
---------

test: clean target/debug/valve | build test/output
	test/main.py --load test/src/table.tsv build > /dev/null
	test/round_trip.sh
	scripts/export.py messages build/valve.db test/output/ column datatype prefix rule table foobar foreign_table import
	diff -q test/expected/messages.tsv test/output/messages.tsv
	test/main.py --insert_update test/src/table.tsv build > /dev/null
	test/insert_update.sh
