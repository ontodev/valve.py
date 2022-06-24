# valve.py
VALVE bindings for Python

## Setup

1. `git clone git@github.com:ontodev/valve.py.git -b valve_rs_python_bindings`
2. `cd valve.py`
3. `git clone git@github.com:ontodev/valve.rs.git -b python_bindings`
4. `cd valve.rs/`
5. `ln -s ../../valve_py.rs src/`
6. Edit `src/lib.rs` and add the line `mod valve_py;` at the end of the file.
7. Edit `Cargo.toml` and make the following changes:
   - Add the line `futures = "0.3"` to the the `[dependencies]` block.
   - Add the following lines:
       ```
       [lib]
       name = "valve"
       crate-type = ["cdylib", "lib"]

       [dependencies.pyo3]
       version = "0.16.5"
       # "extension-module" tells pyo3 we want to build an extension module (skips linking against libpython.so)
       features = ["extension-module"]
       ```
8. Install maturin

        python3 -m venv .venv
        source .venv/bin/activate
        pip install -U pip maturin

9. Build

       `maturin develop` for local installation
       `maturin build` for creating a wheel

10. Copy the test files over

        cp ../test/expected/* test/expected/
        cp ../test/main.py ../test/insert_update.sh test/

11. Replace the recipe for the `test` target in the `Makefile` with the following:
    ```
    test: clean target/debug/valve | build test/output
    	test/main.py --load test/src/table.tsv build > /dev/null
    	test/round_trip.sh
    	scripts/export.py messages build/valve.db test/output/ column datatype prefix rule table foobar foreign_table import
    	diff -q test/expected/messages.tsv test/output/messages.tsv
    	test/main.py --insert_update test/src/table.tsv build > /dev/null
    	test/insert_update.sh
    ```
    **Note that the indented lines should be indended using TABS, not spaces.**

12. Test

        make test
