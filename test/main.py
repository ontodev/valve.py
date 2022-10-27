#!/usr/bin/env python3

import json
import re
import sys
import time
from ontodev_valve import (
    configure_and_or_load,
    get_matching_values,
    validate_row,
    update_row,
    insert_new_row,
)

from argparse import ArgumentParser


def log(message, suppress_time=True):
    if not suppress_time:
        print(f"{time.asctime()} {message}", file=sys.stderr)
    else:
        print(f"{message}", file=sys.stderr)


def warn(message, suppress_time=True):
    log(f"WARNING: {message}", suppress_time)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "table",
        help="A TSV file containing high-level information about the data in the database",
    )
    parser.add_argument(
        "db",
        help="""Either a database connection URL or a path to a SQLite database file. In the
        case of a URL, you must use one of the following schemes: potgresql://<URL>
        (for postgreSQL), sqlite://<relative path> or file:<relative path> (for SQLite).
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--load", action="store_true")
    group.add_argument("--insert_update", action="store_true")
    args = parser.parse_args()

    db = args.db
    if not db.startswith("postgresql://"):
        m = re.search(r"(^(file:|sqlite://))?(.+?)(\?.+)?$", db)
        if m:
            path = m[3]
            params = m[4] or ""
            db = f"{path}{params}"
        else:
            print(f"Could not parse database specification: {db}", file=sys.stderr)
            sys.exit(1)

    if args.load:
        config = configure_and_or_load(args.table, args.db, True)
    elif args.insert_update:
        config = configure_and_or_load(args.table, args.db, False)
        matching_values = get_matching_values(config, args.db, "foobar", "child")
        matching_values = json.loads(matching_values)
        assert matching_values == [
            {"id": "a", "label": "a", "order": 1},
            {"id": "b", "label": "b", "order": 2},
            {"id": "c", "label": "c", "order": 3},
            {"id": "d", "label": "d", "order": 4},
            {"id": "e", "label": "e", "order": 5},
            {"id": "f", "label": "f", "order": 6},
            {"id": "g", "label": "g", "order": 7},
            {"id": "h", "label": "h", "order": 8},
        ]

        # NOTE: No validation of the validate/insert/update functions is done below. You must use an
        # external script to fetch the data from the database and run a diff against a known good
        # sample.

        row = {
            "child": {"messages": [], "valid": True, "value": "b"},
            "parent": {"messages": [], "valid": True, "value": "f"},
            "xyzzy": {"messages": [], "valid": True, "value": "w"},
            "foo": {"messages": [], "valid": True, "value": "A"},
            "bar": {
                "messages": [
                    {"level": "error", "message": "An unrelated error", "rule": "custom:unrelated"}
                ],
                "valid": False,
                "value": "B",
            },
        }

        result_row = validate_row(config, args.db, "foobar", json.dumps(row), True, 1)
        update_row(config, args.db, "foobar", result_row, 1)

        row = {
            "id": {"messages": [], "valid": True, "value": "BFO:0000027"},
            "label": {"messages": [], "valid": True, "value": "bazaar"},
            "parent": {
                "messages": [
                    {"level": "error", "message": "An unrelated error", "rule": "custom:unrelated"}
                ],
                "valid": False,
                "value": "barrie",
            },
            "source": {"messages": [], "valid": True, "value": "BFOBBER"},
            "type": {"messages": [], "valid": True, "value": "owl:Class"},
        }

        result_row = validate_row(config, args.db, "import", json.dumps(row), False)
        new_row_num = insert_new_row(config, args.db, "import", result_row)
