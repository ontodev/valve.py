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

Three VALVE configuration files (as TSV or CSV) are required:
* `datatype`
* `field`
* `rule`

These can be passed as individual files to the input, or you can pass a directory containing these files.

#### Datatype Table

Datatypes allow you to define regex patterns for cell values. The datatypes are a hierarchy of types, and when a datatype is provided as a `type` or `condition`, all parent values are also checked.

The datatype table requires the following fields:
* `datatype`: name of datatype
* `parent`: parent datatype
* `match`: regex match
* `level`: validation fail level when a value does not meet the regex match (info, warn, or error)
* `description`: brief description of datatype
* `instructions`: how to fix problems
* `replace`: regex automatic replacement

The regex patterns should be enclosed with forward slashes (e.g., `/^$/` matches blanks). Replacements should be formatted like `sed` replacements (e.g., `s/\n/ /g` replaces newlines with spaces).

[Example datatype table](https://github.com/ontodev/valve.py/blob/main/tests/resources/inputs/datatype.tsv)

#### Field Table

The field table allows you to define checks for the contents of columns.

The field table requires the following fields:
* `table`: table name within inputs
* `column`: column name within table
* `type`: function or datatype to validate
* `note`: developer note

All contents of the `table.column` are validated against the `type`.

[Example field table](https://github.com/ontodev/valve.py/blob/main/tests/resources/inputs/field.tsv)

#### Rule Table

The rule table allows you to define more complex "when" rules.

The rule table requires the following fields:
* `when table`: table name within inputs
* `when column`: column name within "when table"
* `when condition`: condition to check contents of "when table"."when column" against
* `then table`: table name within inputs
* `then column`: column name within "then table"
* `then condition`: datatype or function to validate when "when condition" returns true
* `level`: validation fail level when the "then condition" fails (info, warn, or error)
* `description`: description of failure, included in message
* `note`: developer note

If the contents of the `"when table"."when column"` do not pass the `when condition`, then the `then condition` is never run. Failing the `when condition` is not considered a validation failure.

[Example rule table](https://github.com/ontodev/valve.py/blob/main/tests/resources/inputs/rule.tsv)

---

### Functions

VALVE functions are provided as values to the `type` column in the field table or the `* condition` fields in the rule table.

When referencing the "target column", that is either the `column` from the field table, or the `then column` from the rule table.

#### CURIE

Usage: `CURIE(str-or-column, [str-or-column, ...])`

This function validates that the contents of the target column are all CURIEs and the prefix of each CURIE is present in the argument list. The `str-or-column` may be a double-quoted string (e.g., `CURIE("foo")`) or a `table.column` pair in which prefixes are defined (e.g., `CURIE(prefix.prefix)`). You may provide one or more arguments.

#### distinct

Usage: `distinct(expr, [table.column, ...])`

This function validates that the contents of the target column are all distinct. If other `table.column` pairs (one or more) are provided after the `expr`, the values of the target column must also be distinct with all those values. The `expr` is either a datatype or another function to perform on the contents of the column.

#### in

Usage: `in(str-or-column, [str-or-column, ...])`

This function validates that the contents of the target column are values present in the argument list. The `str-or-column` may be a double-quoted string (e.g., `in("a", "b", "c")`) or a `table.column` pair in which allowed values are defined (e.g., `in(external.Label)`). You may provide one or more arguments.

#### list

Usage: `list("char", expr)`

This function splits the contents of the target column on the `char` (e.g, `|`) and then checks `expr` on each sub-value. The `expr` is either a datatype or another function to perform. If one sub-value fails the `expr` check, this function fails.

#### lookup

Usage: `lookup(table.column, table.column2)`

This function should be used only in the `then condition` field of the rule table. This function takes the contents of the `when column` and searches for that value in `table.column`. If that value is found, then the `then column` value must be the corresponding value from `table.column2`. Both `table` names passed to `lookup` must be the same.

Given the contents of the rule table:

| when table | when column | when condition | then table | then column | then condition | 
| ---------- | ----------- | -------------- | ---------- | ----------- | -------------- |
| exposure   | Material    | not blank      | exposure   | Material ID | lookup(external.Label, external.ID) |

... validates that when `exposure.Material` is not blank, the `exposure."Material ID"` in that same row is the `external.ID` in the same row as the `exposure.Material` value in `external.Label`:

**external**

| ID      | Label |
| ------- | ----- |
| FOO:123 | bar   |

**exposure**

| Material | Material ID |
| -------- | ----------- |
| bar      | FOO:123     |

#### split

Usage: `split("char", count, expr1, expr2, [expr3, ...])`

This function splits the contents of the target column on the `char`. The number of sub-values must be equal to the `count` and the number of `exprs` provided after must also be equal to the `count`. Each `expr` is a datatype or function that is checked against the corresponding sub-value.

Given the contents of the field table:

| table | column | type |
| ----- | ------ | ---- |
| foo   | bar    | split("&", 2, CURIE(prefix.prefix), in("a", "b", "c")) |

And given the value to check:

> FOO:123 & a

"FOO:123" will be validated against `CURIE(prefix.prefix)` and "a" will be validated against `in("a", "b", "c")`.

#### tree

Usage: `tree(table.column, [table2.column2])`

This function creates a tree structure using the contents of the target column as "parent" values and the contents of `table.column` and "child" values. The `table` portion of the first argument must be the same as the `table` field in the field table. An optional `table2.column2` can be passed as long as `table2.column2` has already been defined as a tree. This means that the current tree will extend the `table2.column2` tree. All "parent" values are required to be in the "child" values, or in the extended tree (if provided).

The `tree` function may only be used as a `type` in the field table. The tree name which can be referenced later in other `tree` functions and the `under` function is the `table` and `column` pair from the field table, e.g. this creates the tree `foo.bar` with child values form `foo.baz`:

| table | column | type          |
| ----- | ------ | ------------- |
| foo   | bar    | tree(foo.baz) |

#### under

Usage: `under(table.column, "top level", [direct=true])`

This function looks for all descendants of `"top level"` in a tree built from `table.column`. Please note that you must first define a `table.column` (corresponding to the `table` and `column` from the field table) tree using the `tree` function. If `direct=true` is included, only *direct* children of `"top level"` are considered allowed values.

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
