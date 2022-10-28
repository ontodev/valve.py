# valve.py
VALVE bindings for Python

## Setup

	git clone git@github.com:ontodev/valve.py.git
	cd valve.py
	make

## Testing

    cd valve.py
	make test

## Usage

1. Activate the virtual environment:

        source valve.rs/.venv/bin/activate

2. Add the statement

        import ontodev_valve

    to the top of your python script.

### Examples

See the file `test/main.py` for usage examples.

## API reference

### `configure_and_or_load(table_table, db_path, load)`

Given a path to a table table file (table.tsv), a directory in which to find/create a database: configure the database using the configuration which can be looked up using the table table, and optionally load it if the `load` flag is set to true.

Returns the configuration map back as a JSON string.

### `get_matching_values(config, db_path, table_name, column_name, matching_string)`

Given a config map represented as a JSON string, a directory containing the database, the table name and column name from which to retrieve matching values, return a JSON array (represented as a string) of possible valid values for the given column which contain the matching string (optional) as a substring (or all of them if no matching string is given). The JSON array returned is formatted for Typeahead, i.e., it takes the form: `[{"id": id, "label": label, "order": order}, ...]`.

### `validate_row(config, db_path, table_name, row, existing_row, row_number)`

Given a config map represented as a JSON string, a directory in which to find the database, a table name, a row, and if the row already exists in the database, its associated row number (optional), perform both intra- and inter-row validation and return the validated row as a JSON string.

### `update_row(config, db_path, table_name, row, row_number)`

Given a config map represented as a JSON string, a directory in which the database is located, a table name, a row represented as a JSON string, and its associated row number, update the row in the database.

### `insert_new_row(config, db_path, table_name, row)`

Given a config map represented as a JSON string, a directory in which the database is located, a table name, and a row represented as a JSON string, insert the new row to the database.

## Before creating a new release

Edit the file `VALVE.VERSION` and adjust the version of valve.py (and, if necessary, valve.rs). After pushing your commit, create a new release in GitHub with the new version number as the release name and tag.
