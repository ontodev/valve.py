import csv
import itertools
import logging
import os
import re
import sys

from argparse import ArgumentParser
from collections import defaultdict
from .parse import parse


# TODO
#  - handle numeric datatypes (later)
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


def has_ancestor(tree, ancestor, node):
    """Check whether a node has an ancestor (or self) in a tree.

    :param tree: a dictionary from chidren to sets of parents
    :param ancestor: the ancestor to look for
    :param node: the node to start from
    :return: True if it has the ancestor, False otherwise"""
    if node == ancestor:
        return True
    if node not in tree:
        return False
    parents = tree[node]
    if ancestor in parents:
        return True
    for parent in parents:
        if has_ancestor(tree, ancestor, parent):
            return True
    return False


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


# ---- INPUT TABLES ----


def get_table_details(tables, row_start=2):
    """Build a dictionary of table details.

    :param tables: list of table paths
    :param row_start: row number that contents to validate start on
    :return: dict of table name -> {fields, rows}
    """
    table_details = {}
    for table in tables:
        sep = "\t"
        if table.endswith("csv"):
            sep = ","
        row_idx = row_start - 2
        with open(table, "r") as f:
            reader = csv.DictReader(f, delimiter=sep)
            table_name = os.path.splitext(os.path.basename(table))[0]
            table_details[table_name] = {
                "fields": reader.fieldnames,
                "rows": list(reader)[row_idx:],
            }
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
    table_name = os.path.splitext(os.path.basename(datatype_table))[0]

    # Read the datatypes from the sheet
    datatypes = {}
    with open(datatype_table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames
        missing = list(set(datatype_headers) - set(headers))
        if missing:
            raise Exception("Missing required headers for 'datatype: " + ", ".join(missing))
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
                    "table": table_name,
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
                    "table": table_name,
                    "cell": idx_to_a1(idx, headers.index("level") + 1),
                    "rule": "unknown level",
                    "message": "the 'level' must be one of: ERROR, WARN, INFO",
                    "kill": True,
                }
            )
    return datatypes, errors


def read_field_table(config, field_table, row_start=2):
    """Build a dictionary of fields.

    :param config: valve config dictionary containing table_details
    :param field_table: path to the 'field' table
    :param row_start: row number that contents to validate start on
    :return: dictionary of table-name -> field-name -> types
    """
    errors = []
    sep = "\t"
    if field_table.endswith("csv"):
        sep = ","
    table_name = os.path.splitext(os.path.basename(field_table))[0]

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
                        "table": table_name,
                        "cell": idx_to_a1(idx, headers.index("column") + 1),
                        "rule": "duplicate column",
                        "message": "this column value is already defined in 'field'",
                        "kill": True,
                    }
                )
            else:
                # Parse the field type
                parsed_type = parse(row["type"])
                success, err = validate_condition(config, parsed_type)
                if not parsed_type:
                    # Type could not be parsed - grammar issue
                    errors.append(
                        {
                            "table": table_name,
                            "cell": idx_to_a1(idx, headers.index("type") + 1),
                            "rule": "invalid type",
                            "message": err,
                            "kill": True,
                        }
                    )
                    continue
                if parsed_type["type"] == "function" and parsed_type["name"] == "tree":
                    # Special processing for `tree` function
                    # This does not get added to field_types,
                    # but a tree is built and added to global trees
                    tree, add_errors = validate_tree_type(config, idx, table, column, parsed_type, row_start=row_start)
                    for err in add_errors:
                        if "table" not in err:
                            err["table"] = (table_name,)
                            err["cell"] = (headers.index("type") + 1,)
                            err["rule"] = "tree function error"
                            err["level"] = "ERROR"
                            err["kill"] = True
                        errors.append(err)
                    if tree:
                        # Add tree to config for further tree iterations
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
    table_name = os.path.splitext(os.path.basename(rule_table))[0]

    table_columns = {
        table_name: details["fields"] for table_name, details in config["table_details"].items()
    }

    table_rules = {}
    with open(rule_table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames
        missing = list(set(rule_headers) - set(headers))
        if missing:
            raise Exception("Missing required headers for 'field': " + ", ".join(missing))

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
                        "table": table_name,
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
                            "table": table_name,
                            "cell": when_column_loc,
                            "rule": "unknown column",
                            "message": f"the provided column must exist in '{when_table}'",
                            "kill": True,
                        }
                    )

            # Validate the when condition
            when_condition = row["when condition"]
            parsed_when_condition = parse(when_condition)
            success, err = validate_condition(config, parsed_when_condition)
            if not success:
                # when-cond could not be parsed
                errors.append(
                    {
                        "table": table_name,
                        "cell": idx_to_a1(idx, headers.index("when condition") + 1),
                        "rule": "invalid condition",
                        "message": err,
                        "kill": True,
                    }
                )
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
            parsed_then_condition = parse(then_condition)
            success, err = validate_condition(config, parsed_then_condition)
            if not success:
                # then-cond could not be parsed
                errors.append(
                    {
                        "table": table_name,
                        "cell": idx_to_a1(idx, headers.index("then condition") + 1),
                        "rule": "invalid condition",
                        "message": err,
                        "kill": True,
                    }
                )
                continue

            # Validate the message level
            level = row.get("level", "")
            if not validate_level(level):
                errors.append(
                    {
                        "table": table_name,
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
                        "table": table_name,
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
                            "table": table_name,
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
                    "unparsed_when": when_condition,
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


def build_tree(
    config, fn_row_idx, table_name, parent_column, child_column, row_start=2, add_tree_name=None, split_char="|",
):
    """Build a hierarchy for the `tree` function while validating the values.

    :param config: valve config dictionary
    :param fn_row_idx: row of tree function in 'field' table
    :param table_name: table name
    :param parent_column: name of column that 'Parent' values are in
    :param child_column: name of column that 'Child' values are in
    :param row_start: row number that contents to validate start on
    :param add_tree_name: optional name of tree to add to
    :param split_char: character to split parent values on
    :return: map of child -> parents, list of errors (if any)
    """
    errors = []

    table_details = config["table_details"]
    rows = table_details[table_name]["rows"]
    col_idx = table_details[table_name]["fields"].index(parent_column)
    trees = config.get("trees", {})
    tree = defaultdict(set)
    if add_tree_name:
        if add_tree_name not in trees:
            errors.append(
                {"message": f"{add_tree_name} must be defined before using it in a function"}
            )
            return None, errors
        tree = trees.get(add_tree_name, defaultdict(set))

    allowed_values = [row[child_column] for row in rows]
    allowed_values.extend(list(tree.keys()))
    row_idx = row_start
    for row in rows:
        parent = row[parent_column]
        child = row[child_column]
        if not parent or parent.strip() == "":
            if child not in tree:
                tree[child] = set()
            row_idx += 1
            continue
        parents = [parent]
        if split_char:
            parents = parent.split(split_char)
        for parent in parents:
            if parent not in allowed_values:
                # show an error on the parent value, but the parent still appears in the tree
                msg = (
                    f"'{parent}' from {table_name}.{parent_column} must exist in {table_name}."
                    + child_column
                )
                if add_tree_name:
                    msg += f" or {add_tree_name} tree"
                errors.append(
                    {
                        "table": table_name,
                        "cell": idx_to_a1(row_idx, col_idx + 1),
                        "rule ID": "field:" + str(fn_row_idx),
                        "level": "ERROR",
                        "message": msg,
                    }
                )
            if child not in tree:
                tree[child] = set()
            tree[child].add(parent)
        row_idx += 1
    return tree, errors


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


def validate_function(config, function):
    """Validate a function.

    :param config: valve config dictionary
    :param function: parsed function as dictionary
    :return: parsed function or None on error, error table entry on error
    """
    errors = []
    funct_name = function["name"]
    if funct_name not in funct_names:
        return False, f"function name ({funct_name}) must be one of: " + ",".join(funct_names)

    # Special validation for each function
    args = function["args"]
    if funct_name == "CURIE":
        # CURIE(table.column)
        if len(args) != 1:
            # must have exactly one value
            return False, "`CURIE` must have exactly one argument"
        if args[0]["type"] != "field":
            # value must be a table.column dict
            return False, "`CURIE` argument must be a table.column pair"

    elif funct_name == "from":
        # from(table.column)
        if len(args) != 1:
            # must have exactly one value
            return False, "`from` must have exactly one argument"
        if args[0]["type"] != "field":
            # value must be a table.column dict
            return False, "`from` argument must be a table.column pair"

    elif funct_name == "in":
        # in("x", "y", "z", ...)
        x = 1
        for arg in args:
            # all args must be strings, not dicts
            if not isinstance(arg, str):
                return False, f"`in` argument {x} must be a string"
        x += 1

    elif funct_name == "list":
        # list(split, funct)
        if len(args) != 2:
            # must have exactly two values
            return False, "`list` must have exactly two arguments"
        if not isinstance(args[0], str):
            return False, "`list` argument 1 must be a string"

        # second value must be a valid function
        success, err = validate_function(config, args[1])
        if not success:
            return False, "`list` argument 2 must be a valid function: " + err

    elif funct_name == "lookup":
        # lookup(table.column, table.column)
        # Validate that the arguments are table-columns and the tables are the same
        if len(args) != 2:
            return False, "`lookup` must have exactly two arguments"
        x = 0
        while x < 2:
            if args[x]["type"] != "field":
                return False, f"`lookup` argument {x} must be a table.column pair"
            x += 1

        table_name = args[0]["table"]
        table_name_2 = args[1]["table"]
        if table_name != table_name_2:
            return (
                False,
                f"the first table name ({table_name}) must be the same as "
                f"the second table name ({table_name_2})",
            )

    elif funct_name == "split":
        # split(split, int, funct, funct, ...)
        if len(args) < 4:
            return False, "`split` must have at least four arguments"
        if not isinstance(args.pop(0), str):
            # first value must be a string
            return False, "`split` argument 1 must be a string"
        try:
            funct_count = int(args.pop(0))
        except ValueError:
            # second value must be a number (passed as str)
            return False, "`split` argument 2 must be a whole number"
        if len(args) != funct_count:
            # rem args must be equal to the last value
            return False, f"`split` must include {funct_count} functions"
        x = 3
        for arg in args:
            # rem args must be valid functions
            success, err = validate_function(config, arg)
            if not success:
                return False, f"`split` argument {x} must be a valid function: " + err
            x += 1

    elif funct_name == "under":
        # under(tree, value)
        if len(args) != 2:
            # must have exactly two values
            return False, "`under` must have exactly two arguments"
        tree_loc = args[0]
        if tree_loc["type"] != "field":
            return False, f"`under` argument 1 must be a table.column pair"

        trees = config["trees"]
        tree_name = f'{tree_loc["table"]}.{tree_loc["column"]}'
        if tree_name not in trees:
            # tree must have already been defined
            return False, f"`under` argument 1 '{tree_name}' must be defined as a tree in 'field'"
        top_level = args[1]
        if not isinstance(top_level, str):
            # second value must be a string
            return False, "`under` argument 2 must be a string"

        tree = trees[tree_name]
        tree_values = list(tree.keys())
        tree_values.extend(list(itertools.chain.from_iterable(tree.values())))
        if top_level not in tree_values:
            # second value must exist in tree
            return False, f"`under` argument 2 ({top_level}) must exist in tree {tree_name}"

    return function, errors


def validate_condition(config, parsed_condition):
    """Validate a condition.

    :param config: valve config dictionary
    :param parsed_condition: parsed condition to validate
    :return: None on error or parsed condition on success, error message
    """
    datatypes = config["datatypes"]
    cond_type = parsed_condition["type"]

    if cond_type == "datatype":
        dt = parsed_condition["name"]
        if dt not in datatypes.keys():
            return False, f"datatype '{dt}' must be defined in the datatype table"
    elif cond_type == "function":
        return validate_function(config, parsed_condition)
    elif cond_type == "negation":
        return validate_condition(config, parsed_condition["expression"])
    elif cond_type == "disjunction":
        # Parse each sub-condition and check if they are valid
        for sub_cond in parsed_condition["disjuncts"]:
            success, err = validate_condition(config, sub_cond)
            if not success:
                # Break on error
                return False, err
    else:
        raise Exception("Unknown condition type: " + cond_type)

    return True, None


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


def validate_tree_type(config, fn_row_idx, tree_table, tree_column, tree_function, row_start=2):
    """Validate a 'tree' field type and build the tree.

    :param config:
    :param tree_table: name of table to build tree from
    :param tree_column: name of column in table to build tree from
    :param fn_row_idx: row that 'tree' appears in from field
    :param tree_function: the parsed field type (tree function)
    :param row_start: row number that contents to validate start on
    :return: tree dictionary or None on error, list of errors
    """
    errors = []
    args = tree_function["args"]
    if 1 > len(args) > 3:
        errors.append({"message": "the `tree` function must have between one and three arguments"})
        return None, errors

    # first arg is always table.column
    tree_arg = args.pop(0)
    if "table" not in tree_arg:
        errors.append(
            {"message": "the first argument of the `tree` function must be a table.column pair"}
        )
        return None, errors

    tree_table_name = tree_arg["table"]
    if tree_table_name != tree_table:
        # logging.error("The table in `tree` must be the same as the `table` value")
        errors.append(
            {
                "message": f"the table name provided in the `tree` function ({tree_table_name}) "
                f"must be the same as the value in the 'table' column ({tree_table})",
            }
        )
        return None, errors
    else:
        # Parse the rest of the args
        add_tree_name = None
        split_char = None
        if args:
            x = 0
            while x < len(args):
                arg = args[x]
                if "split" in arg:
                    split_char = arg["split"]
                elif "table" in arg:
                    add_tree_name = f'{arg["table"]}.{arg["column"]}'
                else:
                    errors.append(
                        {"message": f"`tree` arguments must be table.column pair or split=CHAR"}
                    )
                    return None, errors
                x += 1
        child_column = tree_arg["column"]
        return build_tree(
            config,
            fn_row_idx,
            tree_table,
            tree_column,
            child_column,
            row_start=row_start,
            add_tree_name=add_tree_name,
            split_char=split_char,
        )


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
    config, condition, unparsed_condition, value, when_condition=None, when_value=None
):
    """Determine if the value meets the condition.

    :param config: valve config dictionary
    :param condition: parsed condition to check (as dict)
    :param unparsed_condition: unparsed text of condition for error messages
    :param value: value to check
    :param when_value: "when value" to prompt checking rule
    :param when_condition: "when condition" to prompt checking rule
    :return: True if value meets condition, error message on False
    """
    condition_type = condition["type"]
    datatypes = config["datatypes"]
    if condition_type == "datatype":
        datatype = condition["name"]
        # Check if condition is met, potentially get a replacement
        value_meets_condition, replace = is_datatype(datatypes, datatype, value)
        if value_meets_condition is False:
            if when_condition:
                return (
                    False,
                    f"because '{when_value}' is '{when_condition}', '{value}' must be of datatype "
                    f"'{unparsed_condition}'",
                )
            return False, f"'{value}' must be of datatype '{unparsed_condition}'"

    elif condition_type == "function":
        success, err = run_function(config, condition, value, lookup_value=when_value)
        if not success:
            if when_condition:
                return False, f"because '{when_value}' is '{when_condition}', {err}"
            return False, err

    elif condition_type == "negation":
        # Negations may be one or more
        if not meets_condition(config, condition["expression"], "", value, when_value=when_value)[
            0
        ]:
            # As long as one is "NOT" met, this passes
            return True, None
        # If we get here, the negation conditions were not met
        if unparsed_condition == "not blank":
            if when_condition:
                return False, f"because '{when_value}' is '{when_condition}', value cannot be blank"
            return False, "value cannot be blank"
        if when_condition:
            return (
                False,
                f"because '{when_value}' is '{when_condition}', '{value}' must be "
                f"{unparsed_condition}",
            )
        return False, f"'{value}' must be {unparsed_condition}"

    elif condition_type == "disjunction":
        for c in condition["disjuncts"]:
            if meets_condition(config, c, "", value, when_value=when_value)[0]:
                return True, None
        if when_condition:
            return (
                False,
                f"because '{when_value}' is '{when_condition}', '{value}' must meet one of: "
                f"{unparsed_condition}",
            )
        return False, f"'{value}' must meet one of: {unparsed_condition}"

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
    funct_name = function["name"]
    args = function["args"]

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
        return False, f"prefix '{value_prefix}' must be in {table_name}.{column_name}"
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
        return False, f"'{value}' must be one of: " + ", ".join(args)
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
        return False, f"'{value}' must be in {table_name}.{column_name}"
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
    sub_funct = args[1]
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
    return False, f"'{lookup_value}' must present in {table_name}.{column_name}"


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
        return False, f"'{args[1]}' must have {split_count} elements when split on '{split_char}'"
    errs = []
    x = 0
    while x < split_count:
        v = value_split[x]
        sub_funct = args[x + 2]
        success, err = run_function(config, sub_funct, v, lookup_value=lookup_value)
        if not success:
            errs.append(err)
        x += 1
    if errs:
        return False, " & ".join(errs)
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
    ancestor = args[1]
    if has_ancestor(tree, ancestor, value):
        return True, None
    else:
        return False, f"'{value}' must be equal to or under '{ancestor}' from {tree_name}"


# ---- VALIDATION ----


def collect_distinct(table_details, table, errors):
    """Collect distinct error messages and write the rows with distinct errors to a new table. The
    new table will be [table_name]_distinct. Return the distinct errors with updated locations in
    the new table.

    :param table_details: table name -> details (rows, fields)
    :param table: path to table with errors
    :param errors: all errors from the table
    :return: updated distinct errors from the table
    """
    distinct = {}
    for error in errors:
        if error["message"] not in distinct:
            distinct[error["message"]] = error

    logging.info(f"{len(distinct)} distinct error(s) found in {table}")

    error_rows = defaultdict(list)
    for error in distinct.values():
        row = int(error["cell"][1:])
        error_rows[row].append(error)
    errors = []

    basename = os.path.basename(table)
    dirname = os.path.dirname(table)
    table_name = os.path.splitext(basename)[0]
    table_ext = os.path.splitext(basename)[1]
    sep = "\t"
    if table_ext == ".csv":
        sep = ","
    output = f"{dirname}/{table_name}_distinct{table_ext}"
    logging.info("writing rows with errors to " + output)

    fields = table_details[table_name]["fields"]
    rows = table_details[table_name]["rows"]
    with open(output, "w") as g:
        writer = csv.DictWriter(g, fields, delimiter=sep, lineterminator="\n")
        writer.writeheader()
        row_idx = 2
        new_idx = 2
        for row in rows:
            if row_idx in error_rows.keys():
                writer.writerow(row)
                for error in error_rows[row_idx]:
                    error["table"] = table_name + "_distinct"
                    error["cell"] = error["cell"][0:1] + str(new_idx)
                    errors.append(error)
                new_idx += 1
            row_idx += 1
    return errors


def validate_table(config, table, fields, rules, row_start=2):
    """Run VALVE validation on a table.

    :param config: valve config dictionary
    :param table: path to table
    :param fields: {field-name: type, ...}
    :param rules: dictionary of rules
    :param row_start: row number that contents to validate start on
    :return: list of errors
    """
    errors = []
    table_details = config["table_details"]
    sep = "\t"
    if table.endswith("csv"):
        sep = ","
    table_name = os.path.splitext(os.path.basename(table))[0]
    with open(table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        row_idx = row_start
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
                                "table": table_name,
                                "cell": idx_to_a1(row_idx, col_idx),
                                "rule ID": "field:" + str(field_id),
                                "level": "ERROR",
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
                            check_value = table_details[table]["rows"][row_idx - row_start][column]
                            check_col_idx = table_details[table]["fields"].index(column)
                            success, err_message = meets_condition(
                                config,
                                rule["then_condition"],
                                rule["unparsed_then"],
                                check_value,
                                when_condition=rule["unparsed_when"],
                                when_value=value,
                            )
                            if not success:
                                errors.append(
                                    {
                                        "table": table_name,
                                        "cell": idx_to_a1(row_idx, check_col_idx + 1),
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
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(errors)


def valve(source_dir, output, row_start=2, distinct=False):
    """Main VALVE method.

    :param source_dir: directory containing config files & tables to validate
    :param output: path to output errors to
    :param row_start: row number that contents to validate start on
    :param distinct: if True, collect distinct errors
    :return: True if VALVE completed (with or without errors), False if VALVE configuration failed
    """
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
        elif path.endswith(".csv") or path.endswith(".tsv"):
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

    table_fields, trees, add_errors = read_field_table(config, field_table, row_start=row_start)
    setup_errors.extend(add_errors)

    config["trees"] = trees

    table_rules, add_errors = read_rule_table(config, rule_table)
    setup_errors.extend(add_errors)

    # Check for true setup errors and stop process if they exist
    kill = False

    for e in setup_errors:
        if "kill" in e:
            kill = True
    if kill:
        write_errors(output, setup_errors)
        logging.critical(f"VALVE setup failed with {len(setup_errors)} errors!")
        return False

    config = {"datatypes": datatypes, "table_details": table_details, "trees": trees}

    errors = []
    for table in tables:
        logging.info("validating " + table)
        tname = os.path.splitext(os.path.basename(table))[0]
        fields = table_fields.get(tname, [])
        rules = table_rules.get(tname, [])

        # Validate and return errors
        add_errors = validate_table(config, table, fields, rules, row_start=row_start)

        # Add any non-kill errors that were found during setup
        add_errors.extend([x for x in setup_errors if x["table"] == tname])
        logging.info(f"{add_errors} errors found in {table}")

        if add_errors and distinct:
            # Update errors to only be distinct messages in a new table
            update_errors = collect_distinct(table_details, table, add_errors)
            errors.extend(update_errors)
        elif not distinct:
            errors.extend(add_errors)

    write_errors(output, errors)
    if errors:
        logging.error(f"VALVE completed with {len(errors)} problems found!")
    return True


def main():
    p = ArgumentParser()
    p.add_argument(
        "-D", "--directory", help="Directory containing config and tables", required=True
    )
    p.add_argument(
        "-d",
        "--distinct",
        help="Collect the first of each distinct error messages and write to a separate table",
        action="store_true",
    )
    p.add_argument(
        "-r", "--row-start", help="Index of first row in tables to validate", type=int, default=2
    )
    p.add_argument("-o", "--output", help="CSV or TSV to write error messages to", required=True)
    args = p.parse_args()

    success = valve(args.directory, args.output, row_start=args.row_start, distinct=args.distinct)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
