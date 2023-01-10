MAKEFLAGS += --warn-undefined-variables
.DEFAULT_GOAL := valve.rs/target/release/ontodev_valve
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
.SUFFIXES:

.PHONY: test pg_test sqlite_test clean cleandb cleantestout

clean:
	rm -Rf valve.rs dist .venv

cleandb:
	rm -Rf valve.rs/build

cleantestout:
	rm -Rf valve.rs/test/output

valve.rs/test/main.py: test/main.py | valve.rs
	cp -pf $< $@

valve.rs/build/: | valve.rs
	mkdir -p $@

valve.rs/test/output: | valve.rs
	mkdir -p $@

test: pg_test sqlite_test

tables_to_test = column datatype rule table table1 table2 table3 table4 table5 table6 table7

pg_test: valve.rs/target/release/ontodev_valve cleantestout valve.rs/test/main.py valve.rs/test/insert_update.sh | valve.rs/test/output
	@echo "Testing valve on postgresql ..."
	# This target assumes that we have a postgresql server, accessible by the current user via the
	# UNIX socket /var/run/postgresql, in which a database called `valve_postgres` has been created.
	# It also requires that `psycopg2` has been installed.
	cd valve.rs && source .venv/bin/activate && test/main.py --load test/src/table.tsv postgresql:///valve_postgres > /dev/null
	cd valve.rs && source .venv/bin/activate && test/round_trip.sh postgresql:///valve_postgres test/src/table.tsv
	cd valve.rs && source .venv/bin/activate && scripts/export.py messages postgresql:///valve_postgres test/output/ $(tables_to_test)
	cd valve.rs && diff -q test/expected/messages.tsv test/output/messages.tsv
	cd valve.rs && source .venv/bin/activate && test/main.py --insert_update test/src/table.tsv postgresql:///valve_postgres > /dev/null
	cd valve.rs && source .venv/bin/activate && test/insert_update.sh postgresql:///valve_postgres
	cd valve.rs && source .venv/bin/activate && scripts/export.py messages postgresql:///valve_postgres test/output/ $(tables_to_test)
	cd valve.rs && diff -q test/expected/messages_after_api_test.tsv test/output/messages.tsv
	@echo "Test succeeded!"

sqlite_test: valve.rs/target/release/ontodev_valve cleandb cleantestout valve.rs/test/main.py valve.rs/test/insert_update.sh | valve.rs/build/ valve.rs/test/output
	@echo "Testing valve on sqlite ..."
	cd valve.rs && source .venv/bin/activate && test/main.py --load test/src/table.tsv build/valve.db > /dev/null
	cd valve.rs && source .venv/bin/activate && test/round_trip.sh build/valve.db test/src/table.tsv
	cd valve.rs && source .venv/bin/activate && scripts/export.py messages build/valve.db test/output/ $(tables_to_test)
	cd valve.rs && diff -q test/expected/messages.tsv test/output/messages.tsv
	cd valve.rs && source .venv/bin/activate && test/main.py --insert_update test/src/table.tsv build/valve.db > /dev/null
	cd valve.rs && source .venv/bin/activate && test/insert_update.sh build/valve.db
	cd valve.rs && source .venv/bin/activate && scripts/export.py messages build/valve.db test/output/ $(tables_to_test)
	cd valve.rs && diff -q test/expected/messages_after_api_test.tsv test/output/messages.tsv
	@echo "Test succeeded!"

rs-version := $(shell grep valve\.rs VALVE.VERSION |awk '{print $$2}')
py-version := $(shell grep valve\.py VALVE.VERSION |awk '{print $$2}')

valve.rs:
	curl -L -o valve.tar https://crates.io/api/v1/crates/ontodev_valve/${rs-version}/download
	tar xvf valve.tar
	rm -f valve.tar
	mv ontodev_valve-${rs-version} valve.rs
	cd valve.rs && python3 -m venv .venv
	cd valve.rs && cat ../requirements.txt >> requirements.txt
	cd valve.rs && source .venv/bin/activate && pip install -r requirements.txt

valve.rs/Cargo.toml: | valve.rs
	python3 override_valve_version.py ${py-version} $@ > $@.new
	cd valve.rs && /bin/mv -f $(@F).new $(@F)
	cd valve.rs && ln -s ../../valve_py.rs src/
	cd valve.rs && echo -e "\nmod valve_py;" >> src/lib.rs
	cd valve.rs && cat ../extra_cargo_entries.toml >> $(@F)

valve.rs/target/release/ontodev_valve: valve.rs/Cargo.toml $(wildcard valve.rs/src/*)
	source valve.rs/.venv/bin/activate && maturin develop --release -m $<
