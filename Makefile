MAKEFLAGS += --warn-undefined-variables
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
.SUFFIXES:

.PHONY: install test clean cleanrs

install: .installed

test: cleanrs install | valve.rs/build/ valve.rs/test/output
	cd valve.rs && source .venv/bin/activate && test/main.py --load test/src/table.tsv build/valve.db > /dev/null
	cd valve.rs && test/round_trip.sh
	cd valve.rs && scripts/export.py messages build/valve.db test/output/ column datatype prefix rule table foobar foreign_table import
	cd valve.rs && diff -q test/expected/messages.tsv test/output/messages.tsv
	cd valve.rs && source .venv/bin/activate && test/main.py --insert_update test/src/table.tsv build/valve.db > /dev/null
	cd valve.rs && test/insert_update.sh

clean:
	rm -Rf .installed valve.rs dist .venv
	git checkout valve_py.rs

cleanrs:
	rm -Rf valve.rs/build valve.rs/test/output

valve.rs/Cargo.toml:
	cargo install cargo-quickinstall
	cargo quickinstall cargo-download
	cargo download ontodev_valve==`cat ontodev_valve_version` -x -o valve.rs
	cd valve.rs && ln -s ../../valve_py.rs src/
	cd valve.rs && echo -e "\nmod valve_py;" >> src/lib.rs
	cd valve.rs && cat ../extra_cargo_entries.toml >> Cargo.toml

.installed: valve.rs/Cargo.toml
	cd valve.rs && python3 -m venv .venv
	cd valve.rs && ln -s ../requirements.txt
	cd valve.rs && source .venv/bin/activate && pip install -U -r requirements.txt
	source valve.rs/.venv/bin/activate && maturin develop --release -m valve.rs/Cargo.toml
	cp test/expected/* valve.rs/test/expected/
	cp test/main.py test/insert_update.sh valve.rs/test
	touch $@

valve.rs/build/:
	mkdir -p $@

valve.rs/test/output:
	mkdir -p $@
