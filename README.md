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

You may call `valve.validate` with an optional `functions={...}` argument. The dictionary value should be in the format of function name (for use in rule and field tables) -> details dict. The details dict includes the following items:
* `usage`: usage text (optional)
* `validate`: the function to run for VALVE validation
* `check`: the [expected structure](#checking-with-a-list) of the arguments OR a custom [check function](#checking-with-a-function)

The function name should not collide with any [builtin functions](https://github.com/ontodev/valve/blob/main/README.md#functions). The function must be defined in your file with the following required parameters in this order, even if they are not all used:

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
def validate_foo(config, args, table, column, row_idx, value):
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

valve.validate(
    "inputs/",
    functions={
        "foo": {
            "usage": "foo(string)",
            "check": ["string"],
            "validate": validate_foo
        }
    }
)
```

#### Checking with a list

The `check` list outlines what the arguments passed in should look like. The example above uses a list to validate that exactly one string is passed to `foo`. Each element in the list is an argument type:
* `column`: a column in the target table (the `table` column of the rule or field table)
* `expression`: function or datatype
* `field`: a table-column pair where the table is in the inputs and the column is in the table
* `named:...`: named argument followed by the argument key (e.g., if your named arg looks like `distinct=true`, then this value will be `named:distinct`)
* `regex_match`: a regex pattern
* `regex_sub`: a regex substitution
* `string`: any other string
* `tree`: a defined treename (table-column pair)

If an argument can be of multiple types, you can join them with ` or `. For example, for an argument that can be either a string or a field: `string or field`.

Optional and multi-arity arguments can be specified with special modifiers attached to the end:
* `*`: zero or more
* `?`: zero or one
* `+`: one or more

For example, if you expect one or more string arguments: `string*`. Named arguments are almost always optional, so these would look like: `named:distinct?`. Optional or multi-arity arguments should always be the last parameters.

#### Checking with a function

Lists do not allow you to check dependencies between arguments, so it may be beneficial to define your own `check` function. This function must have four parameters (but not all need to be used):
* `config`: VALVE configuration dictionary
* `table`: the target table that the function will be run in
* `column`: the target column that the function will be run in
* `args`: a list of parsed args passed to the function

The function should return a string error message if any error was found, otherwise, it should return `None`. The custom functions are useful for when you want to validate more than just the structure, for example, if you expect two values that are tables other than the target table:
```python
def validate_foo(config, args, table, column, row_idx, value):
    ...

def check_foo(config, table, column, args):
    i = 1
    for a in args:
        if i == 2:
            return f"foo expects 2 arguments, but {len(args)} were given"
        if a["type"] != "string":
            return f"foo argument {i} must be a string representing a table"
        if a["value"] == table:
            return f"foo argument {i} must not be '{table}'"
        if a["value"] not in config["table_details"]:
            return f"foo argument {i} must be a table in inputs other than '{table}'"
        i += 1

valve.validate(
    "inputs/",
    functions={
        "foo": {
            "usage": "foo(string, string)",
            "check": check_foo,
            "validate": validate_foo
        }
    }
)
```
