import csv
import itertools
import logging
import os
import re
import sys

from argparse import ArgumentParser
from lark import Lark, Tree
from lark.exceptions import UnexpectedInput

# TODO
#  - handle numeric datatypes (later)
#  - look into building docs from doc strings
#  - eventually be able to pass in an excel file


# Required headers for 'datatype' table
datatype_headers = [
    "datatype",
    "parent",
    "match",
    "level",
]
# Other allowed values: description, instructions, replace

# Required headers for 'field' table
field_headers = ["table", "column", "type", "note"]

# Required headers for 'rule' table
rule_headers = [
    "when table",
    "when column",
    "when condition",
    "then table",
    "then column",
    "then condition",
]
# Other allowed values: level, description, note

# Supported function names
funct_names = ["CURIE", "from", "in", "list", "lookup", "split", "tree", "under"]


# ---- MISC HELPERS ----


def build_datatype_ancestors(datatypes, datatype, ancestors=None):
    """Recursively build a list of ancestor datatypes for a given datatype.

    :param datatypes: map of datatype name -> details
    :param datatype: datatype to get ancestors of
    :param ancestors: current list of ancestors to build on, or None to start a new list
    :return: list of ancestor datatypes
    """
    if not ancestors:
        ancestors = []
    parent = datatypes[datatype].get("parent")
    if parent:
        ancestors.append(parent)
        ancestors.extend(build_datatype_ancestors(datatypes, parent, ancestors=ancestors))
    return ancestors


def build_table_descendants(tree, node, descendants=None):
    """Recursively build a list of descendants for a given node from a tree structure.

    :param tree: map of parent node -> list of children nodes
    :param node: node to get all descendants of
    :param descendants: current list of descendants to build on, or None to start a new list
    :return: list of descendant nodes
    """
    if not descendants:
        descendants = []
    children = tree.get(node, [])
    for c in children:
        if c not in descendants:
            descendants.append(c)
            add_descendants = build_table_descendants(tree, c, descendants=descendants)
            if add_descendants:
                descendants.extend(add_descendants)
    return descendants


def idx_to_a1(row, col):
    """Convert a row & column to A1 notation. Adapted from gspread.utils.

    :param row: row index
    :param col: column index
    :return: A1 notation of row:column
    """
    div = col
    column_label = ""

    while div:
        (div, mod) = divmod(div, 26)
        if mod == 0:
            mod = 26
            div -= 1
        column_label = chr(mod + 64) + column_label

    return f"{column_label}{row}"


def tree2list(tree, indent_str, level=0):
    """Adaption of Lark.Tree pretty to return a list. Levels are represented by the indent_str as an
    element that preceeds the item. For example, an item on level 3 will be preceded by a list
    element that is 3 indent_str characters.

    :param Lark.Tree tree: parsed tree to convert to list
    :param indent_str: character to represent indents in the tree
    :param level: current level, or 0
    :return: a list of the tree with indent elements separating different levels
    """
    if len(tree.children) == 1 and not isinstance(tree.children[0], Tree):
        itm = tree.children[0].rstrip('"').lstrip('""')
        return [indent_str * level, tree._pretty_label(), "\t", itm, "\n"]

    tree_list = [indent_str * level, tree._pretty_label(), "\n"]
    for n in tree.children:
        if isinstance(n, Tree):
            tree_list += tree2list(n, indent_str, level=level + 1)
        else:
            tree_list += [indent_str * (level + 1), "%s" % (n,), "\n"]

    return tree_list


# ---- INPUT TABLES ----


def get_table_details(tables):
    """Build a dictionary of table details.

    :param tables: list of table paths
    :return: dict of table name -> {fields, rows}
    """
    table_details = {}
    for table in tables:
        sep = "\t"
        if table.endswith("csv"):
            sep = ","
        with open(table, "r") as f:
            reader = csv.DictReader(f, delimiter=sep)
            table_name = os.path.splitext(os.path.basename(table))[0]
            rows = []
            for row in reader:
                rows.append(row)
            table_details[table_name] = {"fields": reader.fieldnames, "rows": rows}
    return table_details


def read_datatype_table(datatype_table):
    """Build a dictionary of datatypes.

    :param datatype_table: path to datatype table
    :return: dict of datatype -> details (parent, match, level, description, instructions, replace)
    """
    errors = []
    sep = "\t"
    if datatype_table.endswith("csv"):
        sep = ","

    # Read the datatypes from the sheet
    datatypes = {}
    with open(datatype_table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames
        missing = list(set(datatype_headers) - set(headers))
        if missing:
            raise Exception(f"Missing required headers for 'datatype: " + ", ".join(missing))
        idx = 2
        for row in reader:
            dt = row["datatype"]
            del row["datatype"]
            row["idx"] = idx
            datatypes[dt] = row
            idx += 1

    # Validate the datatypes
    dt_names = datatypes.keys()
    for dt, details in datatypes.items():
        idx = details["idx"]

        parent = details["parent"]
        if parent != "" and parent not in dt_names:
            errors.append(
                {
                    "table": datatype_table,
                    "cell": idx_to_a1(idx, headers.index("parent") + 1),
                    "rule": "unknown parent datatype",
                    "message": "the parent datatype must be defined in the 'datatype' sheet",
                    "kill": True,
                }
            )

        level = details.get("level", "")
        if not validate_level(level):
            errors.append(
                {
                    "table": datatype_table,
                    "cell": idx_to_a1(idx, headers.index("level") + 1),
                    "rule": "unknown level",
                    "message": "the 'level' must be one of: ERROR, WARN, INFO",
                    "kill": True,
                }
            )
    return datatypes, errors


def read_field_table(config, field_table):
    """Build a dictionary of fields.

    :param config: valve config dictionary containing table_details
    :param field_table: path to the 'field' table
    :return: dictionary of table-name -> field-name -> types
    """
    errors = []
    sep = "\t"
    if field_table.endswith("csv"):
        sep = ","

    table_details = config["table_details"]

    # Dict of table name -> field types in that table
    table_fields = {}
    trees = {}
    with open(field_table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)

        # Validate headers, quit on error
        headers = reader.fieldnames
        if headers != field_headers:
            raise Exception(f"Headers for 'field' at {field_table} do not match required fields")

        # Validate field table contents
        idx = 1
        for row in reader:
            idx += 1
            table = row["table"]
            column = row["column"]

            # Dict of field name -> its type (parsed)
            if table in table_fields:
                field_types = table_fields[table]
            else:
                field_types = {}

            if column in field_types:
                # This column already has an entry in field
                errors.append(
                    {
                        "table": field_table,
                        "cell": idx_to_a1(idx, headers.index("column") + 1),
                        "rule": "duplicate column",
                        "message": "this column value is already defined in 'field'",
                        "kill": True,
                    }
                )
            else:
                # Parse the field type
                parsed_type, add_errors = validate_condition(
                    config, field_table, idx_to_a1(idx, headers.index("type") + 1), row["type"]
                )
                errors.extend(add_errors)
                if not parsed_type:
                    # Type could not be parsed - grammar issue
                    continue
                if "function" in parsed_type and "tree" in parsed_type["function"]:
                    # Special processing for `tree` function
                    # This does not get added to field_types,
                    # but a tree is built and added to global trees
                    tree, add_errors = validate_tree_type(
                        table_details,
                        field_table,
                        idx,
                        headers.index("type") + 1,
                        table,
                        column,
                        parsed_type,
                    )
                    errors.extend(add_errors)
                    if tree:
                        trees[f"{table}.{column}"] = tree
                        config["trees"] = trees
                    continue
                field_types[column] = {
                    "parsed": parsed_type,
                    "unparsed": row["type"],
                    "field ID": idx,
                }
                table_fields[table] = field_types

    return table_fields, trees, errors


def read_rule_table(config, rule_table):
    """Build a dictionary of rules.

    :param config: valve config dictionary containing table_details
    :param rule_table: path to rule table
    :return: dictionary of table-name -> column-name -> condition
        (when_condition, table, column, then_condition, unparsed_then, level, message, rule ID)
    """
    errors = []
    sep = "\t"
    if rule_table.endswith("csv"):
        sep = ","

    table_columns = {
        table_name: details["fields"] for table_name, details in config["table_details"].items()
    }

    table_rules = {}
    with open(rule_table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames
        missing = list(set(rule_headers) - set(headers))
        if missing:
            raise Exception(f"Missing required headers for 'field': " + ", ".join(missing))

        idx = 1
        for row in reader:
            idx += 1
            # Validate the when table.column (check that these exist)
            when_table = row["when table"]
            when_table_loc = idx_to_a1(idx, headers.index("when table") + 1)
            when_column = row["when column"]
            when_column_loc = idx_to_a1(idx, headers.index("when column") + 1)

            if when_table not in table_columns.keys():
                errors.append(
                    {
                        "table": "rule",
                        "cell": when_table_loc,
                        "rule": "unknown table",
                        "message": "the table must exist in the input",
                        "kill": True,
                    }
                )
            else:
                if when_column not in table_columns[when_table]:
                    errors.append(
                        {
                            "table": "rule",
                            "cell": when_column_loc,
                            "rule": "unknown column",
                            "message": f"the provided column must exist in '{when_table}'",
                            "kill": True,
                        }
                    )

            # Validate the when condition
            when_condition = row["when condition"]
            parsed_when_condition, add_errors = validate_condition(
                config,
                rule_table,
                idx_to_a1(idx, headers.index("when condition") + 1),
                when_condition,
            )
            if not parsed_when_condition:
                # when-cond could not be parsed
                errors.extend(add_errors)
                continue

            # Get the existing columns -> rules for given table
            if when_table in table_rules:
                column_rules = table_rules[when_table]
            else:
                column_rules = {}

            # Get the existing rules for given column
            if when_column in column_rules:
                rules = column_rules[when_column]
            else:
                rules = []

            # Validate the then condition
            then_condition = row["then condition"]
            parsed_then_condition, add_errors = validate_condition(
                config,
                rule_table,
                idx_to_a1(idx, headers.index("then condition") + 1),
                then_condition,
            )
            if not parsed_then_condition:
                # then-cond could not be parsed
                errors.extend(add_errors)
                continue

            # Validate the message level
            level = row.get("level", "")
            if not validate_level(level):
                errors.append(
                    {
                        "table": rule_table,
                        "cell": idx_to_a1(idx, headers.index("level") + 1),
                        "rule": "unknown level",
                        "message": "the 'level' must be one of: ERROR, WARN, INFO",
                        "kill": True,
                    }
                )

            # Validate the when table.column (check that these exist)
            then_table = row["then table"]
            then_table_loc = idx_to_a1(idx, headers.index("then table") + 1)
            then_column = row["then column"]
            then_column_loc = idx_to_a1(idx, headers.index("then column") + 1)
            if then_table not in table_columns.keys():
                errors.append(
                    {
                        "table": "rule",
                        "cell": then_table_loc,
                        "rule": "unknown table",
                        "message": "the table must exist in the input",
                        "kill": True,
                    }
                )
            else:
                if then_column not in table_columns[then_table]:
                    errors.append(
                        {
                            "table": "rule",
                            "cell": then_column_loc,
                            "rule": "unknown column",
                            "message": f"the provided column must exist in '{then_table}'",
                            "kill": True,
                        }
                    )

            # Add this condition to the dicts
            rules.append(
                {
                    "when_condition": parsed_when_condition,
                    "table": then_table,
                    "column": then_column,
                    "then_condition": parsed_then_condition,
                    "unparsed_then": then_condition,
                    "level": level,
                    "message": row.get("description", None),
                    "rule ID": idx,
                }
            )
            column_rules[when_column] = rules
            table_rules[when_table] = column_rules

    return table_rules, errors


# ---- INPUT VALIDATION ----


def build_tree(table, rows, col_idx, parent_column, child_column):
    """Build a hierarchy for the `tree` function while validating the values.

    :param table: table name
    :param rows: table rows
    :param col_idx: column index that the tree function appears in
    :param parent_column: name of column that 'Parent' values are in
    :param child_column: name of column that 'Child' values are in
    :return: map of parent -> children, list of errors (if any)
    """
    errors = []
    tree = {}
    allowed_values = [row[child_column] for row in rows]
    row_idx = 0
    for row in rows:
        row_idx += 1
        parent = row[parent_column]
        child = row[child_column]
        if not parent or parent.strip() == "":
            continue
        if parent not in allowed_values:
            # show an error on the parent value, but the parent still appears in the tree
            errors.append(
                {
                    "table": table,
                    "cell": idx_to_a1(row_idx, col_idx + 1),
                    "rule ID": "field:" + str(row_idx),
                    "rule": "value not in tree",
                    "level": "error",
                    "message": f"{parent} from {table}.{parent_column} does not exist in {table}."
                    + child_column,
                }
            )

        if parent in tree:
            children = tree[parent]
        else:
            children = []

        if child not in children:
            children.append(child)
        tree[parent] = children

    # Add to tree set
    return tree, errors


def is_function(config, table_name, arg_pos, arg):
    """Check if a provided arg is a valid valve function.

    :param config: valve config dictionary
    :param table_name: name of table that function is defined in
    :param arg_pos: position of arg in parent function
    :param arg: parsed arg
    :return: True on success or False on fail, error message on fail
    """
    if not isinstance(arg, dict) or "function" not in arg:
        return False, f"argument {arg_pos} must be a function"
    success, err = validate_function(config, table_name, "", arg["function"])
    if not success:
        return False, err["message"]
    return True, None


def is_table_column(table_details, arg_pos, arg):
    """Check if a provided arg is a valid table.column pair ({"table": str, "column": str}). The
    table should exist in table_details, and the column should exist in the details for that table.

    :param table_details: map of table name -> details (rows, fields)
    :param arg_pos: position of arg in parent function
    :param arg: parsed arg
    :return: True on success or False on fail, error message on fail
    """
    if not isinstance(arg, dict) or "table" not in arg or "column" not in arg:
        # must be a table.column dict
        return False, f"argument {arg_pos} must be a table.column pair"
    # the table and column must exist
    table_name = arg["table"]
    column_name = arg["column"]
    if table_name not in table_details:
        return False, f"argument {arg_pos} references a table ({table_name}) that does not exist"
    if column_name not in table_details[table_name]["fields"]:
        return (
            False,
            f"argument {arg_pos} references a column ({column_name}) in '{table_name}' "
            "that does not exist",
        )
    return True, None


def validate_function(config, table_name, loc, function):
    """Validate a function.

    :param config: valve config dictionary
    :param table_name: name of table that the function is defined in
    :param loc: A1 cell location of function
    :param function: parsed function as dictionary
    :return: parsed function or None on error, error table entry on error
    """
    errors = []
    table_details = config["table_details"]
    funct_name = list(function["function"].keys())[0]
    if funct_name not in funct_names:
        return (
            None,
            {
                "table": table_name,
                "cell": loc,
                "rule": "unknown function",
                "level": "error",
                "message": f"function name ({funct_name}) must be one of: " + ",".join(funct_names),
                "kill": True,
            },
        )

    # Special validation for each function
    args = function["function"][funct_name]
    if funct_name == "CURIE":
        # CURIE(table.column)
        if len(args) != 1:
            # must have exactly one value
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": "CURIE must have exactly one argument",
                    "kill": True,
                },
            )
        success, err = is_table_column(table_details, 1, args[0])
        if not success:
            # value must be a table.column dict
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": err,
                    "kill": True,
                },
            )

    elif funct_name == "from":
        # from(table.column)
        if len(args) != 1:
            # must have exactly one value
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": "from must have exactly one argument",
                    "kill": True,
                },
            )
        success, err = is_table_column(table_details, 1, args[0])
        if not success:
            # value must be a table.column dict
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": err,
                    "kill": True,
                },
            )

    elif funct_name == "in":
        # in(x, y, z, ...)
        x = 1
        for arg in args:
            # all args must be strings, not dicts
            if not isinstance(arg, str):
                return (
                    None,
                    {
                        "table": table_name,
                        "cell": loc,
                        "rule": f"{funct_name} function error",
                        "level": "error",
                        "message": f"argument {x} must be a string",
                        "kill": True,
                    },
                )
        x += 1

    elif funct_name == "list":
        # list(split, funct)
        if len(args) != 2:
            # must have exactly two values
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": "list must have exactly two arguments",
                    "kill": True,
                },
            )
        if not isinstance(args[0], str):
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": f"argument 1 must be a string",
                    "kill": True,
                },
            )
        # second value must be a valid function
        success, err = is_function(config, table_name, 2, args[1])
        if not success:
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": err,
                    "kill": True,
                },
            )

    elif funct_name == "lookup":
        # lookup(table.column, table.column)
        # Validate that the arguments are table-columns and the tables are the same
        if len(args) != 2:
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": "lookup must have exactly two arguments",
                    "kill": True,
                },
            )
        x = 0
        while x < 2:
            success, err = is_table_column(table_details, x + 1, args[x])
            if not success:
                return (
                    None,
                    {
                        "table": table_name,
                        "cell": loc,
                        "rule": f"{funct_name} function error",
                        "level": "error",
                        "message": err,
                        "kill": True,
                    },
                )
            x += 1

        table_name = args[0]["table"]
        table_name_2 = args[1]["table"]
        if table_name != table_name_2:
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": f"the first table name ({table_name}) must be the same as "
                    f"the second table name ({table_name_2})",
                    "kill": True,
                },
            )

    elif funct_name == "split":
        # split(split, int, funct, funct,)
        if len(args) < 4:
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": "split must have at least four arguments",
                    "kill": True,
                },
            )
        if not isinstance(args.pop(0), str):
            # first value must be a string
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": f"argument 1 must be a string",
                    "kill": True,
                },
            )
        try:
            funct_count = int(args.pop(0))
        except ValueError:
            # second value must be a number (passed as str)
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": f"argument 2 must be a whole number",
                    "kill": True,
                },
            )
        if len(args) != funct_count:
            # rem args must be equal to the last value
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": f"split must include {funct_count} functions",
                    "kill": True,
                },
            )
        x = 3
        for arg in args:
            # rem args must be valid functions
            success, err = is_function(config, table_name, x, arg)
            if not success:
                return (
                    None,
                    {
                        "table": table_name,
                        "cell": loc,
                        "rule": f"{funct_name} function error",
                        "level": "error",
                        "message": err,
                        "kill": True,
                    },
                )
            x += 1

    elif funct_name == "under":
        # under(tree, value)
        if len(args) != 2:
            # must have exactly two values
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": "under must have exactly two arguments",
                    "kill": True,
                },
            )
        tree_loc = args[0]
        success, err = is_table_column(table_details, 1, tree_loc)
        if not success:
            # first value must be a table.column dict
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": err,
                    "kill": True,
                },
            )

        trees = config["trees"]
        tree_name = f'{tree_loc["table"]}.{tree_loc["column"]}'
        if tree_name not in trees:
            # tree must have already been defined
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": f"{tree_name} must be defined as a tree in 'field'",
                    "kill": True,
                },
            )
        top_level = args[1]
        if not isinstance(top_level, str):
            # second value must be a string
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": f"argument 2 must be a string",
                    "kill": True,
                },
            )
        tree = trees[tree_name]
        tree_values = list(tree.keys())
        tree_values.extend(list(itertools.chain.from_iterable(tree.values())))
        if top_level not in tree_values:
            # second value must exist in tree
            return (
                None,
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": f"{funct_name} function error",
                    "level": "error",
                    "message": f"argument 2 ({top_level}) must exist in tree {tree_name}",
                    "kill": True,
                },
            )

    return function, errors


def validate_condition(config, table_name, loc, condition, parse_cond=True):
    """Validate a condition.

    :param config: valve config dictionary
    :param table_name: name of table that condition is defined in
    :param condition: text or parsed condition
    :param loc: A1 location of condition
    :param parse_cond: if true, parse a text condition
    :return: None on error or parsed condition on success, error messages
    """
    errors = []
    if parse_cond:
        parsed_condition, err = parse(condition)
    else:
        parsed_condition, err = condition, None

    datatypes = config["datatypes"]

    if not parsed_condition:
        # Condition could not be parsed - unexpected input
        msg = f"unable to parse '{condition}' due to unexpected input"
        if err:
            msg = f"unable to parse due to unexpected input:\n{err}"
        errors.append(
            {
                "table": table_name,
                "cell": loc,
                "rule": "invalid condition",
                "level": "error",
                "message": msg,
                "kill": True,
            }
        )
        return None, errors

    # Validate specifics
    if "datatype" in parsed_condition:
        dt = parsed_condition["datatype"]
        if dt not in datatypes.keys():
            errors.append(
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": "unknown datatype",
                    "level": "error",
                    "message": f"datatype '{dt}' is not defined in the datatype table",
                    "kill": True,
                }
            )
            return None, errors
    elif "disjunction" in parsed_condition:
        valid = True
        # Parse each sub-condition and check if they are valid
        for sub_cond in parsed_condition["disjunction"]:
            parsed_sub, add_errors = validate_condition(
                config, table_name, loc, sub_cond, parse_cond=False
            )
            errors.extend(add_errors)
            if not parsed_sub:
                valid = False
        if not valid:
            return None, errors
    elif "function" in parsed_condition:
        parsed_function, err = validate_function(config, table_name, loc, parsed_condition)
        if not parsed_function:
            errors.append(err)
            return None, errors
        return parsed_function, errors
    elif "negation" in parsed_condition:
        valid = True
        for sub_cond in parsed_condition["negation"]:
            parsed_sub, add_errors = validate_condition(
                config, table_name, loc, sub_cond, parse_cond=False
            )
            errors.extend(add_errors)
            if not parsed_sub:
                valid = False
        if not valid:
            return None, errors
    else:
        # Something is wrong with the Lark grammar
        raise Exception("Unknown expression: " + str(parsed_condition))
    return parsed_condition, errors


def validate_level(level):
    """Validate a level entry. The level must be one of (case-insensitive): error, warn, or info.

    :param level: logging level
    :return: True if level is valid or False
    """
    if not level or level == "":
        return False
    elif level.lower() not in ["error", "warn", "info"]:
        return False
    return True


def validate_tree_type(
    table_details, field_table, fn_row_idx, fn_col_idx, tree_table, tree_column, tree_function
):
    """Validate a 'tree' field type and build the tree.

    :param table_details: dictionary of table name -> details (rows, fields)
    :param field_table: path to field table
    :param tree_table: name of table to build tree from
    :param tree_column: name of column in table to build tree from
    :param fn_row_idx: row that 'tree' appears in from field
    :param fn_col_idx: column that 'tree' appears in from field
    :param tree_function: the parsed field type (tree function)
    :return: tree dictionary or None on error, list of errors
    """
    errors = []
    args = tree_function["function"]["tree"]
    if len(args) != 1:
        # tree(...) must have exactly one argument
        # logging.error("The `tree` function accepts exactly one argument")
        errors.append(
            {
                "table": field_table,
                "cell": idx_to_a1(fn_row_idx, fn_col_idx),
                "rule": "tree function error",
                "level": "error",
                "message": f"the `tree` function must have exactly one argument",
                "kill": True,
            }
        )
        return None, errors
    tree_table_name = args[0]["table"]
    if tree_table_name != tree_table:
        # logging.error("The table in `tree` must be the same as the `table` value")
        errors.append(
            {
                "table": field_table,
                "cell": idx_to_a1(fn_row_idx, fn_col_idx),
                "rule": "tree function error",
                "level": "error",
                "message": f"the table name provided in the `tree` function ({tree_table_name}) "
                f"must be the same as the value in the 'table' column ({tree_table})",
                "kill": True,
            }
        )
        return None, errors
    else:
        child_column = args[0]["column"]
        return build_tree(
            tree_table,
            table_details[tree_table]["rows"],
            table_details[tree_table]["fields"].index(tree_column),
            tree_column,
            child_column,
        )


# ---- LARK PARSING ----


def parse_args(arg):
    """Parse an argument of a 'function' expression.

    :param arg: list representing one argument provided to a function.
                This list is created from the Lark parser output.
    :return: str or dict representation of arg, or None on error
    """
    found_label = False
    search_label = False
    search_field = False
    search_int = False
    table = None
    itm = None

    for itm in arg:
        if isinstance(itm, int):
            continue

        # if itm == "function":

        if itm == "integer":
            search_int = True
            continue

        if search_int:
            return int(itm)

        if itm == "field":
            # Field is a table.column pair
            search_field = True
            continue

        if search_field:
            if not found_label and itm == "label":
                found_label = True
                continue
            if not table:
                # Table always comes first
                table = itm
                found_label = False
                continue
            else:
                # As long as we've bypassed table and label, the item is the column name
                return {"table": table, "column": itm}

        if itm == "label":
            # Ignore the first instance of "label" - part of grammar
            # We cannot globally ignore "label" because that may be a datatype
            search_label = True
            continue

        if search_label:
            # No "field" means that this is just a string arg
            return itm

    if table and itm:
        return {"table": table, "column": itm}
    else:
        return None


def parse_function(function):
    """Parse a 'function' list to a dictionary.

    :param function: 'function' list from Lark parsing
    :return: dictionary representation of the list
    """
    args_level = 0
    cur_level = 0
    func_level = 0

    func_name = None
    search_name = False

    cur_arg = None
    in_func = False

    args = []

    for itm in function:
        if isinstance(itm, int):
            cur_level = itm

        if itm == "function_name" and not func_name:
            search_name = True
            continue
        if search_name:
            func_name = itm
            search_name = False
            continue

        if itm == "arguments":
            args_level = cur_level

        if itm == "argument" and cur_arg is None:
            cur_arg = []
            continue

        if cur_arg is not None:
            if itm == "function":
                in_func = True
                if cur_level < args_level:
                    if cur_arg:
                        parsed, err = parse_function_arg(cur_arg)
                        if parsed:
                            args.append(parsed)
                        else:
                            return None, err
                    func_level = cur_level
                    cur_arg = []
                else:
                    cur_arg.append("function")
                continue

            if in_func and cur_level > func_level:
                cur_arg.append(itm)
                continue

            if itm == "argument" and cur_level - 1 == args_level:
                if cur_arg:
                    in_func = False
                    parsed, err = parse_function_arg(cur_arg)
                    if parsed:
                        args.append(parsed)
                    else:
                        return None, err
                cur_arg = []
                continue
            cur_arg.append(itm)

    if cur_arg:
        parsed, err = parse_function_arg(cur_arg)
        if parsed:
            args.append(parsed)
        else:
            return None, err

    return {"function": {func_name: args}}, None


def parse_function_arg(cur_arg):
    """Parse a function that has been provided as an argument to another function.

    :param cur_arg: list of parsed values for function
    :return: parsed function as dictionary or None, error message on failure
    """
    popped = cur_arg.pop(0)
    if cur_arg[0] == "function_name":
        cur_arg.insert(0, popped)
        cur_arg.insert(0, "function")
    if cur_arg[0] == "function":
        return parse_function(cur_arg)
    else:
        parsed = parse_args(cur_arg)
        if parsed:
            return parsed, None
        else:
            return None, "Unable to parse args: " + cur_arg


def parse_sub_conditions(sub_conds):
    """Parse sub-conditions within a disjunction or negation.

    :param sub_conds: list representing all sub-conditions.
                      This list is created from the Lark parser output.
    :return: list of dict(s) (None on error), error message (None on success)
    """
    search_type = False
    search_expr = False
    expr_type = None
    found_label = False
    function = None
    parsed_set = []
    in_negation = False
    for itm in sub_conds:
        if itm == "expression":
            # Reset
            if function:
                # Parse previous function, if it exists
                function.insert(0, "function_name")
                parsed_function, err = parse_function(function)
                if not parsed_function:
                    return None, err
                if in_negation:
                    parsed_set.append({"negation": parsed_function})
                    in_negation = False
                else:
                    parsed_set.append(parsed_function)
            expr_type = None
            search_expr = True
            function = None
            search_type = False
            continue

        if search_expr:
            if isinstance(itm, int):
                continue
            if search_type:
                expr_type = itm
                search_expr = False
                search_type = False
                continue
            if itm == "type":
                search_type = True
                function = None
                continue
            if itm == "negation":
                in_negation = True
            search_expr = False
            continue

        if itm == "type":
            # Next item will be the type, either function or datatype
            search_type = True
            function = None
            continue
        if search_type and not isinstance(itm, int):
            # Grab "expression" type and continue
            expr_type = itm
            search_type = False
            continue

        if expr_type:
            # Parse based on the expression type
            if expr_type == "datatype":
                if not found_label and itm == "label":
                    found_label = True
                    continue
                if isinstance(itm, int):
                    continue
                if in_negation and not isinstance(itm, int):
                    parsed_set.append({"negation": {"datatype": itm}})
                    in_negation = False
                else:
                    parsed_set.append({"datatype": itm})
                expr_type = None
                search_type = False
            elif expr_type == "function":
                if function is not None:
                    function.append(itm)
                else:
                    function = []
    if function:
        # Make sure to get last function if it exists
        function.insert(0, "function_name")
        parsed_function, err = parse_function(function)
        if not parsed_function:
            return None, err
        if in_negation:
            parsed_set.append({"negation": parsed_function})
        else:
            parsed_set.append(parsed_function)
    return parsed_set, None


def parse(text):
    """Parse a text function or datatype.

    :param text: text to parse
    :return: parsed text as dict (None on error), error message (None on success)
    """
    parser = Lark(
        """
        start: expression
        expression: negation | disjunction | type
        negation: "not" expression
        disjunction: expression ("or" expression)+
        type: function | datatype
        function: function_name "(" arguments ")"
        function_name: WORD
        arguments: argument ("," argument)*
        argument: field | label | integer | function
        field: label "." label
        datatype: label
        label: WORD | ESCAPED_STRING
        integer: INTEGER
        INTEGER : /[0-9]+/

        %import common.WORD
        %import common.ESCAPED_STRING
        %ignore " "           // Disregard spaces in text
    """
    )

    try:
        t = parser.parse(text)
    except UnexpectedInput as e:
        return None, e.get_context(text)

    tree_list_unparsed = tree2list(t, ">")
    tree_list_unparsed = [x for x in tree_list_unparsed if x != "\n" and x != "\t"]
    tree_list = []

    for itm in tree_list_unparsed:
        if re.match(r"^>+$", itm):
            tree_list.append(len(itm) - 1)
        else:
            tree_list.append(itm)
    # Remove the first levels & their ints (start and expression)
    tree_list.pop(0)
    tree_list.pop(0)
    tree_list.pop(0)
    tree_list.pop(0)
    tree_list.pop(0)

    # Get the first value and transform the list into a dict
    v = tree_list.pop(0)
    if v == "type":
        tree_list.pop(0)
        expr_type = tree_list.pop(0)
        if expr_type == "datatype":
            return {"datatype": tree_list[-1]}, None
        else:
            p_funct, err = parse_function(tree_list)
            if not p_funct:
                return None, err
            return p_funct, None
    elif v == "disjunction":
        p_set, err = parse_sub_conditions(tree_list)
        if not p_set:
            return None, err
        return {"disjunction": p_set}, None
    elif v == "negation":
        p_set, err = parse_sub_conditions(tree_list)
        if not p_set:
            return None, err
        return {"negation": p_set}, None


# ---- CONDITION VALIDATION ----


def is_datatype(datatypes, datatype, value):
    """Determine if the value is of datatype.

    :param datatypes: dictionary of datatype names -> details
    :param datatype: datatype that value should be
    :param value: value to validate
    :return: True if value is datatype or False otherwise, optional replacement when False
    """
    # First build a list of ancestors
    ancestor_dts = build_datatype_ancestors(datatypes, datatype)
    ancestor_dts.insert(0, datatype)
    for dt in ancestor_dts:
        re_pattern = datatypes[dt].get("match")[1:-1]
        if re_pattern:
            if not re.match(re_pattern, value):
                fix = datatypes[dt].get("replace")
                match = re.match(r".?/(.+)/(.+)/.?", fix)
                if match:
                    pattern = match.group(1)
                    replace = match.group(2)
                    return False, re.sub(pattern, replace, value)
                return False, None
    return True, None


def meets_condition(
    config, condition, unparsed_condition, value, when_value=None,
):
    """Determine if the value meets the condition.

    :param config: valve config dictionary
    :param condition: parsed condition to check (as dict)
    :param unparsed_condition: unparsed text of condition for error messages
    :param value: value to check
    :param when_value: lookup value, required only for 'lookup' function
    :return: True if value meets condition, error message on False
    """
    condition_type = list(condition.keys())[0]
    datatypes = config["datatypes"]
    if condition_type == "datatype":
        datatype = condition["datatype"]
        # Check if condition is met, potentially get a replacement
        value_meets_condition, replace = is_datatype(datatypes, datatype, value)
        if value_meets_condition is False:
            return False, f"'{value}' is not of datatype '{unparsed_condition}'"

    elif condition_type == "function":
        success, err = run_function(config, condition["function"], value, lookup_value=when_value)
        if not success:
            return False, f"{unparsed_condition} failed: {err}"

    elif condition_type == "negation":
        # Negations may be one or more
        for c in condition["negation"]:
            if not meets_condition(config, c, "", value, when_value=when_value)[0]:
                # As long as one is "NOT" met, this passes
                return True, None
        # If we get here, the negation conditions were not met
        return False, f"'{value}' does not meet any of the criteria: {unparsed_condition}"

    elif condition_type == "disjunction":
        for c in condition["disjunction"]:
            if meets_condition(config, c, "", value, when_value=when_value)[0]:
                return True, None
        return False, f"'{value}' does not meet any of the criteria: {unparsed_condition}"

    else:
        # This should be prevented in validate_condition
        raise Exception("unknown condition type: " + condition_type)
    return True, None


def run_function(config, function, value, lookup_value=None):
    """Run a VALVE function for the provided value.

    :param config: valve config dictionary
    :param function: function to run (as parsed dictionary)
    :param value: value to run function on
    :param lookup_value: required for lookup function
    :return: True if value passes function, error message on False
    """
    table_details = config["table_details"]
    funct_name = list(function.keys())[0]
    args = function[funct_name]

    if funct_name == "CURIE":
        return CURIE(table_details, args, value)
    elif funct_name == "in":
        return in_set(args, value)
    elif funct_name == "from":
        return from_table_column(table_details, args, value)
    elif funct_name == "list":
        return for_each_list(config, args, value, lookup_value=lookup_value)
    elif funct_name == "lookup":
        return lookup(table_details, args, value, lookup_value)
    elif funct_name == "split":
        return split(config, args, value, lookup_value=lookup_value)
    elif funct_name == "under":
        trees = config["trees"]
        return under(trees, args, value)
    else:
        # This should never be reached in normal operation
        # validate_condition already checks that this is OK
        raise Exception("Unknown function: " + funct_name)


# ---- VALVE FUNCTIONS ----


def CURIE(table_details, args, value):
    """Method for the VALVE 'CURIE' function. The value must be a CURIE and the prefix of the value
    must be in the table.column pair defined by the only argument.

    :param table_details: dictionary of table name -> details
    :param args: arguments provided to CURIE
    :param value: value to run CURIE on
    :return: True if value passes CURIE, error message on False
    """
    table_name = args[0]["table"]
    column_name = args[0]["column"]
    prefixes = []
    for row in table_details[table_name]["rows"]:
        prefixes.append(row[column_name])
    if ":" not in value:
        return False, f"'{value}' is not a CURIE"
    value_prefix = value.split(":")[0]
    if value_prefix not in prefixes:
        return False, f"prefix '{value_prefix}' is not in {table_name}.{column_name}"
    return True, None


def in_set(args, value):
    """Method for the VALVE 'in' function. The value must be one of the arguments.

    :param args: arguments provided to in
    :param value: value to run in on
    :return: True if value passes in, error message on False"""
    valid = False
    for allowed_value in args:
        if value == allowed_value:
            valid = True
            break
    if not valid:
        return False, f"'{value}' is not one of: " + ", ".join(args)
    return True, None


def from_table_column(table_details, args, value):
    """Method for the VALVE 'from' function.

    The value must exist in the table.column pair defined by the only argument.

    :param table_details: dictionary of table name -> details
    :param args: arguments provided to from
    :param value: value to run from on
    :return: True if value passes from, error message on False"""
    table_name = args[0]["table"]
    column_name = args[0]["column"]
    source_rows = table_details[table_name]["rows"]
    allowed_values = [x[column_name] for x in source_rows if column_name in x]
    if value not in allowed_values:
        return False, f"'{value}' is not in {table_name}.{column_name}"
    return True, None


def for_each_list(config, args, value, lookup_value=None):
    """Method for the VALVE 'list' function.

    Split the value on the first argument and perform the function provided as the second argument
    on all values.

    :param config: valve config dictionary
    :param args: arguments provided to list
    :param value: value to run list on
    :param lookup_value: value required for 'lookup' when 'lookup' is used as the sub-function
    :return: True if value passes list, error message on False"""
    split_char = args[0]
    sub_funct = args[1]["function"]
    errs = []
    for v in value.split(split_char):
        success, err = run_function(config, sub_funct, v, lookup_value=lookup_value)
        if not success:
            errs.append(err)
    if errs:
        return False, "\n".join(errs)
    return True, None


def lookup(table_details, args, value, lookup_value):
    """Method for VALVE 'lookup' function.

    The lookup value is found in the first table.column pair (first argument), then the allowed
    value is retrived from the second table.column (second argument) pair on the same row. The
    provided value must be exactly the same as the found value. Both tables must be the same.

    :param table_details: dictionary of table name -> details
    :param args: arguments provided to lookup
    :param value: value to run lookup on
    :param lookup_value: value to lookup in the target table.column pair
    :return: True if value passes lookup, error message on False"""
    if not lookup_value:
        raise Exception("A lookup_value is required for a lookup function")
    table_name = args[0]["table"]
    column_name = args[0]["column"]
    column_name_2 = args[1]["column"]
    for row in table_details[table_name]["rows"]:
        maybe_value = row[column_name]
        if maybe_value == lookup_value:
            check_value = row[column_name_2]
            if value != check_value:
                return False, f"'{value}' must be '{check_value}'"
            return True, None
    return False, f"'{lookup_value}' not present in {table_name}.{column_name}"


def split(config, args, value, lookup_value=None):
    """Method for VALVE 'split' function.

    Split the value on the first argument. The number of values after the split must match the
    number provided by the second argument. Iterate through the split values and perform the
    corresponding function from the remaining arguments.

    :param config: valve config dictionary
    :param args: arguments provided to split
    :param value: value to run split on
    :param lookup_value: value required for 'lookup' when 'lookup' is used as a sub-function
    :return: True if value passes split, error message on False"""
    split_char = args[0]
    split_count = int(args[1])
    value_split = value.split(split_char)
    if len(value_split) != split_count:
        return False, f"value must have {split_count} elements when split on '{split_char}'"
    errs = []
    x = 0
    while x < split_count:
        v = value_split[x]
        sub_funct = args[x + 2]["function"]
        success, err = run_function(config, sub_funct, v, lookup_value=lookup_value)
        if not success:
            errs.append(err)
        x += 1
    if errs:
        return False, "\n".join(errs)
    return True, None


def under(trees, args, value):
    """Method for VALVE 'under' function.

    Retrieve the tree defined by the first argument (a table.column pair). The value must be a
    descendant of the second argument.

    :param trees: dictionary of tree name -> dictionary of parent node -> list of children nodes
    :param args: arguments provided to under
    :param value: value to run under on
    :return: True if value passes under, error message on False"""
    table_name = args[0]["table"]
    column_name = args[0]["column"]
    tree_name = f"{table_name}.{column_name}"
    if tree_name not in trees:
        # This has already been validated for CLI users
        raise Exception(f"A tree for {tree_name} is not defined")

    tree = trees[tree_name]
    top_level = args[1]
    descendants = build_table_descendants(tree, top_level)
    descendants.insert(0, top_level)
    if value not in descendants:
        return False, f"'{value}' is not under '{top_level}' from {tree_name}"
    return True, None


# ---- MAIN METHODS ----


def validate_table(config, table, fields, rules):
    """Run VALVE validation on a table.

    :param config: valve config dictionary
    :param table: path to table
    :param fields: {field-name: type, ...}
    :param rules: dictionary of rules
    :return: list of errors
    """
    errors = []
    table_details = config["table_details"]
    sep = "\t"
    if table.endswith("csv"):
        sep = ","
    with open(table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        row_idx = 0
        for row in reader:
            col_idx = 1
            for field, value in row.items():
                if not value:
                    value = ""
                # Check for field type
                if field in fields:
                    # Get the expected field type
                    # This will be validated based on the given datatypes
                    parsed_type = fields[field]["parsed"]
                    unparsed = fields[field]["unparsed"]

                    # all values in this field must match the type
                    mc, err_message = meets_condition(config, parsed_type, unparsed, value)
                    if not mc:
                        field_id = fields[field]["field ID"]
                        errors.append(
                            {
                                "table": table,
                                "cell": idx_to_a1(row_idx + 1, col_idx),
                                "rule ID": "field:" + str(field_id),
                                "level": "error",
                                "message": err_message,
                            }
                        )
                # Check for rules
                if field in rules:
                    # Check if the value meets any of the conditions
                    for rule in rules[field]:
                        # Run meets_condition without logging
                        # as the then-cond check is only run if the value matches the type
                        if meets_condition(config, rule["when_condition"], "", value)[0]:
                            # The "when" value meets the condition - validate the "then" value
                            table = rule["table"]
                            column = rule["column"]

                            # Retrieve the "then" value to check if it meets the "then condition"
                            check_value = table_details[table]["rows"][row_idx][column]
                            check_col_idx = table_details[table]["fields"].index(column)
                            success, err_message = meets_condition(
                                config,
                                rule["then_condition"],
                                rule["unparsed_then"],
                                check_value,
                                when_value=value,
                            )
                            if not success:
                                errors.append(
                                    {
                                        "table": table,
                                        "cell": idx_to_a1(row_idx + 2, check_col_idx + 1),
                                        "rule ID": "rule:" + str(rule["rule ID"]),
                                        "rule": rule["message"],
                                        "level": rule["level"],
                                        "message": err_message,
                                    }
                                )
                col_idx += 1
            row_idx += 1
    return errors


def write_errors(output, errors):
    """Write errors to a file.

    :param output: path to write errors to
    :param errors: list of dictionaries of error messages
    """
    sep = "\t"
    if output.endswith("csv"):
        sep = ","
    with open(output, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["table", "cell", "rule ID", "rule", "level", "message", "suggestion"],
            delimiter=sep,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(errors)


def main():
    p = ArgumentParser()
    p.add_argument("-D", "--directory", required=True)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    source_dir = args.directory
    datatype_table = None
    field_table = None
    rule_table = None
    tables = []
    for f in os.listdir(source_dir):
        fname = os.path.splitext(f)[0]
        path = os.path.join(source_dir, f)
        if fname == "datatype":
            datatype_table = path
        elif fname == "field":
            field_table = path
        elif fname == "rule":
            rule_table = path
        else:
            tables.append(path)

    if not datatype_table:
        raise RuntimeError("A 'datatype' TSV or CSV must be included in " + source_dir)
    if not field_table:
        raise RuntimeError("A 'field' TSV or CSV must be included in " + source_dir)
    if not rule_table:
        raise RuntimeError("A 'rule' TSV or CSV must be included in " + source_dir)
    if not tables:
        raise RuntimeError("Additional tables to validate must be included in " + source_dir)

    setup_errors = []
    table_details = get_table_details(tables)

    datatypes, add_errors = read_datatype_table(datatype_table)
    setup_errors.extend(add_errors)

    config = {"table_details": table_details, "datatypes": datatypes}

    table_fields, trees, add_errors = read_field_table(config, field_table)
    setup_errors.extend(add_errors)

    config["trees"] = trees

    table_rules, add_errors = read_rule_table(config, rule_table,)
    setup_errors.extend(add_errors)

    # Check for true setup errors and stop process if they exist
    kill = False
    errors = []
    for e in setup_errors:
        if "kill" in e:
            kill = True
            del e["kill"]
        errors.append(e)
    if kill:
        write_errors(args.output, errors)
        logging.critical(f"VALVE setup failed with {len(errors)} errors!")
        sys.exit(1)

    config = {"datatypes": datatypes, "table_details": table_details, "trees": trees}

    for table in tables:
        tname = os.path.splitext(os.path.basename(table))[0]
        fields = table_fields.get(tname, [])
        rules = table_rules.get(tname, [])
        add_errors = validate_table(config, table, fields, rules)
        errors.extend(add_errors)

    write_errors(args.output, errors)
    if errors:
        logging.critical(f"VALVE completed with {len(errors)} problems found!")


if __name__ == "__main__":
    main()
