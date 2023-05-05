# valve.py
VALVE bindings for Python

## Install/setup from source

1. Retrieve valve.py from GitHub:

	    git clone git@github.com:ontodev/valve.py.git
	    cd valve.py
	    make test

2. Activate the virtual environment:

        source valve.rs/.venv/bin/activate

3. Add the statement

        import ontodev_valve

    to the top of your python script.

## Usage examples

See the file `test/main.py` for usage examples.

## API reference

### `valve(table_table, db_path, command, verbose, config_table="table")`

Given a path to a configuration table (either a table.tsv file or a database containing a
table named "table"), and a directory in which to find/create a database: configure the
database using the configuration which can be looked up using the table table, and
optionally create and/or load it according to the value of `command`
(see [Valve Command](#ValveCommand)). If the `verbose` flag is set to true, output status
messages while loading. If `config_table` (which defaults to "table") is given and
`table_table` indicates a database, query the table called `config_table` for the
table table information. Returns the configuration map as a String.

### `get_matching_values(config, db_path, table_name, column_name, matching_string)`

Given a config map represented as a JSON string, a directory containing the database, the table name and column name from which to retrieve matching values, return a JSON array (represented as a string) of possible valid values for the given column which contain the matching string (optional) as a substring (or all of them if no matching string is given). The JSON array returned is formatted for Typeahead, i.e., it takes the form: `[{"id": id, "label": label, "order": order}, ...]`.

### `validate_row(config, db_path, table_name, row, existing_row, row_number)`

Given a config map represented as a JSON string, a directory in which to find the database, a table name, a row, and if the row already exists in the database, its associated row number (optional), perform both intra- and inter-row validation and return the validated row as a JSON string.

### `update_row(config, db_path, table_name, row, row_number)`

Given a config map represented as a JSON string, a directory in which the database is located, a table name, a row represented as a JSON string, and its associated row number, update the row in the database.

### `insert_new_row(config, db_path, table_name, row)`

Given a config map represented as a JSON string, a directory in which the database is located, a table name, and a row represented as a JSON string, insert the new row to the database.

### ValveCommand

The following commands may be used with the valve() function.

#### ValveCommand.Config

Configure but do not create or load.

#### ValveCommand.Create

Configure and create but do not load.

#### ValveCommand.Load

Configure, create, and load.

## Before creating a new release

Edit the file `VALVE.VERSION` and adjust the version of valve.py (and, if necessary, valve.rs). After pushing your commit, create a new release in GitHub with the new version number as the release name and tag.
