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
