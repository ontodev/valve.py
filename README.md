# valve.py
VALVE in Python

* [Command Line Usage](#command-line-usage)
	* [Configuration Files](#configuration-files)
	* [Functions](#functions)
	* [Other Options](#other-options)
* [API](#api)

## Setup

The latest release of VALVE can be downloaded from PyPI using `pip install`:
```
python3 -m pip install ontodev-valve
```

Alternatively, for the developer version, you may clone this repository, navigate to it, and install locally:
```
python3 -m pip install .
```

To confirm installation, run `valve -h`

## Command Line Usage

```
valve path [path ...] [-d DISTINCT] [-r ROW_START] -o OUTPUT
```

Each `path` may be a file or a directory. If a directory is passed, VALVE will search for all TSVs and CSVs within that directory and add them to the list of input files. It will not search nested directories.

At this time, only TSV and CSV tables are accepted.

The output `-o`/`--output` must be a path to a TSV or CSV file to write validation messages to. The output is formatted based on [COGS message tables](https://github.com/ontodev/cogs#message-tables). An example table can be found [here](https://github.com/ontodev/valve.py/blob/main/tests/resources/errors.tsv).

---

### Configuration Files

Two VALVE configuration files (as TSV or CSV) are required:
* `datatype`
* `field`

You may also include an optional `rule` table.

These can be passed as individual files to the input, or you can pass a directory containing these files. [More details on these files can be found here](https://github.com/ontodev/valve/blob/main/README.md#configuration-files).

---

### Functions

VALVE functions are provided as values to the `type` column in the field table or the `* condition` fields in the rule table.

[More details on functions can be found here](https://github.com/ontodev/valve/blob/main/README.md#functions).

---

### Other Options

#### Distinct Messages

Often, the same validation problem is found duplicated on multiple rows. It may be beneficial to just see only the *first* instance of any unique message. The `-d`/`--distinct` option collects distinct messages and writes *only* the input rows that correspond to these messages to a new `*_distinct` file in the provided directory:
```
valve input/ -d distinct/ -o problems_distinct.tsv
```

For example, if multiple problems are found in `input/table.tsv`, the first row with the message will be written to `distinct/table_distinct.tsv`. The cell locations in the output (`problems_distinct.tsv`) correspond to the cells in `distinct/table_distinct.tsv`, not the original input.

#### Row Start

By default, VALVE begins validation on row 2 of all input files. The first row must always be the headers, but if you wish to skip N number of rows, you can do so with `-r`/`--row-start`:
```
valve input/ -r 3 -o problems.tsv
```

This tells VALVE to begin validation on row 3 of all input files, excluding the VALVE configuration files.

## API

You can import the VALVE module into your projects:
```
import valve
```

<!-- TODO: add link to auto-generated docs -->
The main method is [`valve.validate`](https://github.com/ontodev/valve.py/blob/main/valve/valve.py#L1470), which accepts either a list of input paths (files or directories) or a config dictionary like the one output by [`valve.get_config_from_tables`](https://github.com/ontodev/valve.py/blob/main/valve/valve.py#L1392). `valve.validate` returns a list of messages. Each message is a dictionary with fields for [COGS message tables](https://github.com/ontodev/cogs#message-tables).

### Custom Functions

You may call `valve.validate` with an optional `functions={...}` argument. The dictionary value should be in the format of function name (for use in rule and field tables) -> function object (which may or may not have the same name). The function name should not collide with any [builtin functions](https://github.com/ontodev/valve/blob/main/README.md#functions). The function must be defined in your file with the following required parameters in this order, even if they are not all used:

1. `config`: VALVE configuration dictionary
2. `args`: parsed (via `valve.parse`) arguments from the function
3. `table`: table name containing value
4. `column`: column name containing value
5. `row_idx`: row index containing value
6. `value`: value to run the function on

The function should return a list of messages (empty on success). The messages are dictionaries with the following keys:
* `table`: table name (no parent directories or extension)
* `cell`: A1 format of cell location (you can use `idx_to_a1` to get this\*)
* `message`: detailed error message

\* When getting the A1 format of the location, note that the `row_idx` always starts at zero, without headers (or any skipped rows) included in the list of rows. You must add `row_start` to this to get the correct row number.

You may also include a `suggestion` key if you want to provide a suggested replacement value.

For example:
```python
def foo_bar(config, args, table, column, row_idx, value):
    required_in_value = args[0]["value"]
    if required_in_value not in value:
        row_start = config["row_start"]
        col_idx = config["table_details"][table]["fields"].index(column)
        cell_loc = valve.idx_to_a1(row_idx + row_start, col_idx + 1)
        return [
            {
                "table": table,
                "cell": cell_loc,
                "message": f"'{value}' must contain '{required_in_value}'",
            }
        ]
    return []

valve.validate("inputs/", functions={"foo": foo_bar})
```
