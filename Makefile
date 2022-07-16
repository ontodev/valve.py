MAKEFLAGS += --warn-undefined-variables
.DELETE_ON_ERROR:
.SUFFIXES:

.installed:
	cargo install cargo-download
	cargo download ontodev_valve=0.1.0 -x -o valve.rs
	cd valve.rs && ln -s ../../valve_py.rs src/
	echo -e "\nmod valve_py;" >> valve.rs/src/lib.rs
	echo -e "\n\
	# [lib] and [dependencies.pyo3] are needed for valve.py bindings:\n\
	[lib]\n\
	name = \"ontodev_valve\"\n\
	crate-type = [\"cdylib\", \"lib\"]\n\
	\n\
	[dependencies.pyo3]\n\
	version = \"0.16.5\"\n\
	# \"extension-module\" tells pyo3 we want to build an extension module (skips linking against libpython.so)\n\
	features = [\"extension-module\"]" >> valve.rs/Cargo.toml
	cd valve.rs && python3 -m venv .venv
	cd valve.rs && ln -s ../requirements.txt
	cd valve.rs && source .venv/bin/activate && pip install -U -r requirements.txt
	cd valve.rs && source .venv/bin/activate && maturin develop --release
	cp test/expected/* valve.rs/test/expected/
	cp test/main.py test/insert_update.sh valve.rs/test
	touch $@

valve.rs/build/:
	mkdir -p $@

valve.rs/test/output:
	mkdir -p $@

.PHONY: clean cleanrs install test

clean:
	rm -Rf .installed valve.rs
	git checkout valve_py.rs

cleanrs:
	rm -Rf valve.rs/build valve.rs/test/output

install: .installed

test: cleanrs | valve.rs/build/ valve.rs/test/output
	cd valve.rs && source .venv/bin/activate && test/main.py --load test/src/table.tsv build/valve.db > /dev/null
	cd valve.rs && test/round_trip.sh
	cd valve.rs && scripts/export.py messages build/valve.db test/output/ column datatype prefix rule table foobar foreign_table import
	diff -q valve.rs/test/expected/messages.tsv valve.rs/test/output/messages.tsv
	cd valve.rs && source .venv/bin/activate && test/main.py --insert_update test/src/table.tsv build/valve.db > /dev/null
	cd valve.rs && test/insert_update.sh
