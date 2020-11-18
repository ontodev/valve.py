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
    "parents",
    "match",
    "level",
]
# Other allowed values: description, instructions, replace

# Required headers for 'field' table
field_headers = ["table", "column", "condition"]

# Required headers for 'rule' table
rule_headers = [
    "table",
    "when column",
    "when condition",
    "then column",
    "then condition",
]
# Other allowed values: level, description, note

# Supported function names
funct_names = ["CURIE", "distinct", "in", "sub", "list", "lookup", "split", "tree", "under"]


# ---- MISC HELPERS ----


def build_datatype_ancestors(datatypes, datatype):
    """Recursively build a list of ancestor datatypes for a given datatype.

    :param datatypes: map of datatype name -> details
    :param datatype: datatype to get ancestors of
    :return: list of ancestor datatypes
    """
    ancestors = []
    parents = datatypes[datatype].get("parents")
    for p in parents:
        ancestors.append(p)
        ancestors.extend(build_datatype_ancestors(datatypes, p))
    return ancestors


def has_ancestor(tree, ancestor, node, direct=False):
    """Check whether a node has an ancestor (or self) in a tree.

    :param tree: a dictionary from children to sets of parents
    :param ancestor: the ancestor to look for
    :param node: the node to start from
    :param direct: if True, only look at direct parents, not full ancestors
    :return: True if it has the ancestor, False otherwise"""
    if node == ancestor and not direct:
        return True
    if node not in tree:
        return False
    parents = tree[node]
    if ancestor in parents:
        return True
    if direct:
        return False
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
        with open(table, "r") as f:
            reader = csv.DictReader(f, delimiter=sep)
            table_name = os.path.splitext(os.path.basename(table))[0]
            table_details[table_name] = {
                "path": table,
                "fields": reader.fieldnames,
                "rows": list(reader)[row_start - 2 :],
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
    datatypes = defaultdict(dict)
    with open(datatype_table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames
        missing = list(set(datatype_headers) - set(headers))
        if missing:
            raise Exception("Missing required column for 'datatype' table: " + ", ".join(missing))
        idx = 2
        for row in reader:
            dt = row["datatype"]
            if not re.match(r"^(?![0-9])[A-Za-z0-9-_]+$", dt):
                errors.append(
                    {
                        "table": table_name,
                        "cell": idx_to_a1(idx, headers.index("datatype") + 1),
                        "rule": "invalid datatype name",
                        "message": "the datatype must use only alphanumeric characters, dashes, "
                                   "and underscores and must not start with an integer",
                        "kill": True,
                    }
                )
            del row["datatype"]

            parents_str = row["parents"].strip()
            parents = []
            if parents_str != "":
                parents = parents_str.split(" ")
            row["parents"] = parents
            row["idx"] = idx
            datatypes[dt] = row
            idx += 1

    # Validate the datatypes
    dt_names = datatypes.keys()
    for dt, details in datatypes.items():
        idx = details["idx"]
        for p in details["parents"]:
            if p not in dt_names:
                errors.append(
                    {
                        "table": table_name,
                        "cell": idx_to_a1(idx, headers.index("parents") + 1),
                        "rule": "unknown parent datatype",
                        "message":
                        f" parent datatype ({p}) must be defined in the 'datatype' sheet",
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
        missing = list(set(field_headers) - set(headers))
        if missing:
            raise Exception("Missing required columns for 'rule' table: " + ", ".join(missing))

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
                # Parse the field condition
                parsed_type = parse(row["condition"])
                success, err = validate_condition(config, parsed_type)
                if not success:
                    errors.append(
                        {
                            "table": table_name,
                            "cell": idx_to_a1(idx, headers.index("condition") + 1),
                            "rule": "invalid condition",
                            "message": err,
                            "kill": True,
                        }
                    )
                    continue
                if parsed_type["type"] == "function" and parsed_type["name"] == "tree":
                    # Special processing for `tree` function
                    # This does not get added to field_types,
                    # but a tree is built and added to global trees
                    tree, add_errors = validate_tree_type(
                        config, idx, table, column, parsed_type, row_start=row_start
                    )
                    for err in add_errors:
                        if "table" not in err:
                            err.update(
                                {
                                    "table": table_name,
                                    "cell": idx_to_a1(idx, headers.index("condition") + 1),
                                    "rule": "`tree` function error",
                                    "level": "ERROR",
                                    "kill": True,
                                }
                            )
                        errors.append(err)
                    if tree:
                        # Add tree to config for further tree iterations
                        trees[f"{table}.{column}"] = tree
                        config["trees"] = trees
                    continue
                elif parsed_type["type"] == "function" and parsed_type["name"] == "distinct":
                    # Validate the args
                    args = parsed_type["args"]
                    success, err = validate_distinct(args)
                    if err:
                        errors.append(
                            {
                                "table": table_name,
                                "cell": idx_to_a1(idx, headers.index("condition") + 1),
                                "rule": "`distinct` function error",
                                "level": "ERROR",
                                "message": err,
                                "kill": True,
                            }
                        )
                        continue
                    success, add_errors = distinct(
                        config["table_details"], args, table, column, row_start=row_start,
                    )
                    if not success:
                        for err in add_errors:
                            err.update(
                                {
                                    "rule ID": "field:" + str(idx),
                                    "rule": "non-distinct value(s)",
                                    "level": "ERROR",
                                }
                            )
                            errors.append(err)
                    # Set the first arg (an expression) to the type for this table.column
                    parsed_type = args[0]

                field_types[column] = {
                    "parsed": parsed_type,
                    "unparsed": row["condition"],
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
            raise Exception("Missing required columns for 'rule' table: " + ", ".join(missing))

        idx = 1
        for row in reader:
            idx += 1
            # Validate the when table.column (check that these exist)
            table = row["table"]
            table_loc = idx_to_a1(idx, headers.index("table") + 1)
            when_column = row["when column"]
            when_column_loc = idx_to_a1(idx, headers.index("when column") + 1)

            if table not in table_columns.keys():
                errors.append(
                    {
                        "table": table_name,
                        "cell": table_loc,
                        "rule": "unknown table",
                        "message": "the table must exist in the input",
                        "kill": True,
                    }
                )
            else:
                if when_column not in table_columns[table]:
                    errors.append(
                        {
                            "table": table_name,
                            "cell": when_column_loc,
                            "rule": "unknown column",
                            "message": f"the provided column must exist in '{table}'",
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
            if table in table_rules:
                column_rules = table_rules[table]
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
            then_column = row["then column"]
            then_column_loc = idx_to_a1(idx, headers.index("then column") + 1)
            if then_column not in table_columns[table]:
                errors.append(
                    {
                        "table": table_name,
                        "cell": then_column_loc,
                        "rule": "unknown column",
                        "message": f"the provided column must exist in '{table}'",
                        "kill": True,
                    }
                )

            # Add this condition to the dicts
            rules.append(
                {
                    "when_condition": parsed_when_condition,
                    "unparsed_when": when_condition,
                    "table": table,
                    "column": then_column,
                    "then_condition": parsed_then_condition,
                    "unparsed_then": then_condition,
                    "level": level,
                    "message": row.get("description", None),
                    "rule ID": idx,
                }
            )
            column_rules[when_column] = rules
            table_rules[table] = column_rules

    return table_rules, errors


# ---- INPUT VALIDATION ----


def build_tree(
    config,
    fn_row_idx,
    table_name,
    parent_column,
    child_column,
    row_start=2,
    add_tree_name=None,
    split_char=None,
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


def validate_expression(config, funct_name, pos, arg):
    """Validate that an argument is a function or datatype.

    :param config: valve config dictionary
    :param funct_name: name of function currently being validated
    :param pos: position of this argument
    :param arg: argument to validate
    :return: True on success False on error, error message on error
    """
    if not isinstance(arg, dict):
        return False, f"`{funct_name}` argument {pos} ({arg}) must be a valid datatype or function"
    if arg["type"] == "datatype":
        datatypes = config["datatypes"]
        dt_name = arg["name"]
        if dt_name not in datatypes:
            return (
                False,
                f"`{funct_name}` argument {pos} datatype ({dt_name}) must be a defined datatype",
            )
    else:
        success, err = validate_function(config, arg)
        if not success:
            return (
                False,
                f"`{funct_name}` argument {pos} must be a valid datatype or function: " + err,
            )
    return True, None


def validate_function(config, function):
    """Validate a function.

    :param config: valve config dictionary
    :param function: parsed function as dictionary
    :return: True on success False on error, error message on error
    """
    errors = []
    funct_name = function["name"]
    if funct_name not in funct_names:
        return False, f"function name ({funct_name}) must be one of: " + ",".join(funct_names)

    table_details = config["table_details"]

    # Special validation for each function
    args = function["args"]
    if funct_name == "CURIE":
        # CURIE(table.column)
        x = 1
        for arg in args:
            if (not isinstance(arg, str) and not isinstance(arg, dict)) or (
                isinstance(arg, dict) and arg["type"] != "field"
            ):
                return False, f"`CURIE` argument {x} must be a table.column pair or a string"
            if isinstance(arg, dict):
                success, err = validate_table_column(table_details, "CURIE", x, arg)
                if not success:
                    return False, err
            x += 1

    elif funct_name == "in":
        # in("x", "y", "z", ...)
        x = 1
        for arg in args:
            if (not isinstance(arg, str) and not isinstance(arg, dict)) or (
                isinstance(arg, dict) and arg["type"] != "field"
            ):
                return False, f"`in` argument {x} must be a table.column pair or a string"
            if isinstance(arg, dict):
                success, err = validate_table_column(table_details, "in", x, arg)
                if not success:
                    return False, err
            x += 1

    elif funct_name == "sub":
        # sub(regex, expr)
        if len(args) != 2:
            return False, "`sub` must have exactly two arguments"
        if not isinstance(args[0], dict) or args[0]["type"] != "regex":
            return False, "`sub` argument 1 must be a regex pattern"
        flag_str = args[0]["flags"]
        if not re.match(r"[agix]+", flag_str):
            return False, "`sub` regex flag(s) must be one or more of: a, g, i, or x"

        # second value must be a valid function or datatype
        success, err = validate_expression(config, "sub", 2, args[1])
        if not success:
            return False, err

    elif funct_name == "list":
        # list(split, expr)
        if len(args) != 2:
            # must have exactly two values
            return False, "`list` must have exactly two arguments"
        if not isinstance(args[0], str):
            return False, "`list` argument 1 must be a string"

        # second value must be a valid function or datatype
        success, err = validate_expression(config, "list", 2, args[1])
        if not success:
            return False, err

    elif funct_name == "lookup":
        # lookup(table, column, column)
        if len(args) != 3:
            return False, "`lookup` must have exactly three arguments"
        table = args[0]
        if not isinstance(table, str) or table not in table_details:
            return False, "`lookup` argument 1 must be a table name"
        headers = table_details[table]["fields"]
        x = 1
        while x < 3:
            arg = args[x]
            if not isinstance(arg, str) or arg not in headers:
                return False, f"`lookup` argument {x + 1} must be a column name in '{table}'"
            x += 1

    elif funct_name == "split":
        # split(split, int, expr, expr, ...)
        if len(args) < 4:
            return False, "`split` must have at least four arguments"
        if not isinstance(args[0], str):
            # first value must be a string
            return False, "`split` argument 1 must be a string"
        try:
            funct_count = int(args[1])
        except ValueError:
            # second value must be a number (passed as str)
            return False, "`split` argument 2 must be a whole number"
        if len(args) - 2 != funct_count:
            # rem args must be equal to the last value
            return False, f"`split` must include {funct_count} functions"
        x = 2
        while x < len(args):
            # rem args must be valid functions or datatypes
            success, err = validate_expression(config, "split", x + 1, args[x])
            if not success:
                return False, err
            x += 1

    elif funct_name == "under":
        # under(tree, value)
        if len(args) != 2 and len(args) != 3:
            # can have two or three args
            return False, "`under` must have either two or three arguments"
        tree_loc = args[0]
        if not isinstance(tree_loc, dict) or tree_loc["type"] != "field":
            return False, f"`under` argument 1 must be a table.column pair"

        tree_name = f'{tree_loc["table"]}.{tree_loc["column"]}'
        if "trees" not in config:
            return False, f"a `tree` for {tree_name} must be defined in order to use `under`"
        trees = config["trees"]

        if tree_name not in trees:
            # tree must have already been defined
            return False, f"`under` argument 1 '{tree_name}' must be defined as a tree in 'field'"
        top_level = args[1]
        if not isinstance(top_level, str):
            # second value must be a string
            return False, "`under` argument 2 must be a string"

        if len(args) == 3:
            if args[2]["type"] != "named_arg":
                return False, "if provided, `under` argument 3 must be `direct=bool`"
            if args[2]["name"] != "direct":
                return False, "`under` only accepts named argument `direct`"
            if args[2]["value"].lower() not in ["true", "false"]:
                return False, "in `under` argument 3, the value of `direct=` must be true or false"

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


def validate_distinct(args):
    """Validate arguments to the `distinct` function.

    :param args: list of args passed to `distinct`
    :return: True if valid or False if not, error message on False
    """
    expr = args[0]
    if expr["type"] != "datatype" and expr["type"] != "function":
        return False, "`distinct` argument 1 must be a datatype or function"
    if len(args) > 1:
        arg_idx = 2
        for arg in args[1:]:
            if not isinstance(arg, dict) or arg["type"] != "field":
                return False, f"`distinct` argument {arg_idx} must be a table.column pair"
            arg_idx += 1
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


def validate_table_column(table_details, fn_name, arg_pos, arg):
    """Validate a table.column arg pair.

    :param table_details: dictionary of table name -> details
    :param fn_name: name of function to validate
    :param arg_pos: position of argument in function
    :param arg: argument dict
    :return: True if valid False if not, error message on False
    """
    table_name = arg["table"]
    column_name = arg["column"]
    if table_name not in table_details:
        return (
            False,
            f"`{fn_name}` argument {arg_pos} must use a valid table name "
            f"({table_name} is not in inputs)",
        )
    if column_name not in table_details[table_name]["fields"]:
        return (
            False,
            f"`{fn_name}` argument {arg_pos} must use a field name from {table_name} "
            f"('{column_name}' is not in fields)",
        )
    return True, None


def validate_tree_type(config, fn_row_idx, table_name, parent_column, tree_function, row_start=2):
    """Validate a 'tree' field type and build the tree.

    :param config: dict of VALVE config
    :param table_name: name of table to build tree from
    :param parent_column: name of column in table to build tree from
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

    # first arg is column
    child_column = args[0]
    if not isinstance(child_column, str):
        errors.append(
            {"message": "the first argument of the `tree` function must be a column name"}
        )
        return None, errors

    # Parse the rest of the args
    add_tree_name = None
    split_char = None
    if args:
        x = 1
        while x < len(args):
            arg = args[x]
            arg_type = arg["type"]
            if arg_type == "named_arg":
                if arg["name"] == "split":
                    split_char = arg["value"]
                else:
                    errors.append({"message": f"`tree` named argument must be split=CHAR"})
                    return None, errors
            elif arg_type == "field":
                add_tree_name = f'{arg["table"]}.{arg["column"]}'
            else:
                errors.append(
                    {"message": f"`tree` arguments must be table.column pair or split=CHAR"}
                )
                return None, errors
            x += 1
    return build_tree(
        config,
        fn_row_idx,
        table_name,
        parent_column,
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
        return in_set(table_details, args, value)
    elif funct_name == "sub":
        return substitute(config, args, value, lookup_value=lookup_value)
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
    must be in the table.column pair or string defined by the arg (1+ args)

    :param table_details: dictionary of table name -> details
    :param args: arguments provided to CURIE
    :param value: value to run CURIE on
    :return: True if value passes CURIE, error message on False
    """
    prefixes = []
    # Get prefixes from args - either strings or table.column pairs
    for arg in args:
        if isinstance(arg, str):
            prefixes.append(arg)
            continue
        table_name = arg["table"]
        column_name = arg["column"]
        for row in table_details[table_name]["rows"]:
            prefixes.append(row[column_name])
    if ":" not in value:
        return False, f"'{value}' is not a CURIE"
    value_prefix = value.split(":")[0]
    if value_prefix not in prefixes:
        return False, f"prefix '{value_prefix}' must be one of: " + ", ".join(prefixes)
    return True, None


def distinct(table_details, args, table, column, row_start=2):
    """Method for the VALVE 'distinct' function.

    :param table_details: dictionary of table name -> details
    :param args: arguments provided to distinct
    :param table: table to run distinct on
    :param column: column to run distinct on
    :param row_start: row number of row that values to validate start on
    :return: True if values pass distinct, list of errors on False
    """
    base_rows = table_details[table]["rows"]
    base_headers = table_details[table]["fields"]
    base_values = [x[column] for x in base_rows if x.get(column, None)]

    # extra columns to check - {table: {column: values}}
    errors = []
    with_values = defaultdict(dict)
    if len(args) > 1:
        for itm in args[1:]:
            t = itm["table"]
            c = itm["column"]
            trows = table_details[t]["rows"]
            values = [x[c] for x in trows if x.get(c, None)]
            if t not in with_values:
                with_values[t] = dict()
            with_values[t].update({c: values})

    # Check external table.columns for values that are duplicate to anything in table.column
    duplicate_values = defaultdict(set)
    for t, col_values in with_values.items():
        for c, values in col_values.items():
            headers = table_details[t]["fields"]
            idx = row_start
            for v in values:
                if v in base_values:
                    base_loc = idx_to_a1(
                        base_values.index(v) + row_start, base_headers.index(column) + 1
                    )
                    if v not in duplicate_values:
                        duplicate_values[v] = set()
                    duplicate_values[v].add(f"{t}:{idx_to_a1(idx, headers.index(c) + 1)}")
                    duplicate_values[v].add(f"{table}:{base_loc}")
                idx += 1

    # Check the table.column for duplicate values
    if len(base_values) > len(set(base_values)):
        # Create a dict of value -> indexes
        value_to_idxs = defaultdict(list)
        for i, v in enumerate(base_values):
            value_to_idxs[v].append(i)
        value_to_idxs = {k: v for k, v in value_to_idxs.items() if len(v) > 1}

        # Add these to the duplicate_values map
        for val, idxs in value_to_idxs.items():
            if val not in duplicate_values:
                duplicate_values[val] = set()
            for i in idxs:
                duplicate_values[val].add(
                    f"{table}:{idx_to_a1(i + row_start, base_headers.index(column) + 1)}"
                )

    # Create the error messages
    for value, locs in duplicate_values.items():
        for loc in locs:
            t = loc.split(":")[0]
            a1 = loc.split(":")[1]
            other_locs = locs.copy()
            other_locs.remove(loc)
            errors.append(
                {
                    "table": t,
                    "cell": a1,
                    "message": f"'{value}' must be distinct with value(s) at: "
                    + ", ".join(other_locs),
                }
            )
    if errors:
        return False, errors
    return True, None


def in_set(table_details, args, value):
    """Method for the VALVE 'in' function. The value must be one of the arguments.

    :param table_details: dictionary of table name -> details
    :param args: arguments provided to in
    :param value: value to run in on
    :return: True if value passes in, error message on False"""
    allowed = []
    for arg in args:
        if isinstance(arg, str):
            if value == arg:
                return True, None
            allowed.append(f'"{arg}"')
        else:
            table_name = arg["table"]
            column_name = arg["column"]
            source_rows = table_details[table_name]["rows"]
            allowed_values = [x[column_name] for x in source_rows if column_name in x]
            if value in allowed_values:
                return True, None
            allowed.append(f"{table_name}.{column_name}")
    return False, f"'{value}' must be in: " + ", ".join(allowed)


def substitute(config, args, value, lookup_value=None):
    """Method for the VALVE 'sub' function.

    Substitute match with replacement, then evaluate the expression.

    :param config: valve config dictionary
    :param args: arguments provided to list
    :param value: value to run list on
    :param lookup_value: value required for 'lookup' when 'lookup' is used as the sub-function
    :return: True if value passes list, error message on False"""
    regex = args[0]
    subfunc = args[1]
    pattern = regex["pattern"]

    # Handle any regex flags
    flags = regex["flags"]
    count = 1
    ignore_case = False
    if flags:
        if "g" in flags:
            # Set count to zero to replace all matches
            count = 0
            flags = flags.replace("g", "")
        if "i" in flags:
            # Use python flags instead
            # (?i) does not work if there are no alpha characters in pattern
            ignore_case = True
            flags = flags.replace("i", "")
        if flags:
            # a and x flags can be inserted into the pattern
            pattern = f"?({flags}){pattern}"

    if ignore_case:
        value = re.sub(pattern, regex["replace"], value, count=count, flags=re.IGNORECASE)
    else:
        value = re.sub(pattern, regex["replace"], value, count=count)

    # Handle the expression (dataype or function)
    if subfunc["type"] == "datatype":
        datatypes = config["datatypes"]
        datatype = subfunc["name"]
        value_is_datatype = is_datatype(datatypes, datatype, value)[0]
        if not value_is_datatype:
            return False, f"substituted value '{value}' must be of datatype {datatype}"
        return True, None
    else:
        return run_function(config, subfunc, value, lookup_value=lookup_value)


def for_each_list(config, args, value, lookup_value=None):
    """Method for the VALVE 'list' function.

    Split the value on the first argument and perform the function or datatype check provided as the
    second argument on all values.

    :param config: valve config dictionary
    :param args: arguments provided to list
    :param value: value to run list on
    :param lookup_value: value required for 'lookup' when 'lookup' is used as the sub-expression
    :return: True if value passes list, error message on False"""
    split_char = args[0]
    expr = args[1]
    expr_name = expr["name"]
    datatypes = config["datatypes"]
    errs = []
    for v in value.split(split_char):
        if expr["type"] == "datatype":
            success, _ = is_datatype(datatypes, expr_name, v)
            if not success:
                errs.append(f"sub-value '{v}' must be of datatype '{expr_name}'")
        else:
            success, err = run_function(config, expr, v, lookup_value=lookup_value)
            if not success:
                errs.append(err)
    if errs:
        return False, "\n".join(errs)
    return True, None


def lookup(table_details, args, value, lookup_value):
    """Method for VALVE 'lookup' function.

    The lookup value is found in the first column (second argument), then the allowed
    value is retrieved from the second column (third argument) pair on the same row. The
    provided value must be exactly the same as the found value.

    :param table_details: dictionary of table name -> details
    :param args: arguments provided to lookup
    :param value: value to run lookup on
    :param lookup_value: value to lookup in the target table.column pair
    :return: True if value passes lookup, error message on False"""
    if not lookup_value:
        raise Exception("A lookup_value is required for a lookup function")
    table_name = args[0]
    column_name = args[1]
    column_name_2 = args[2]
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
    corresponding function or datatype match from the remaining arguments.

    :param config: valve config dictionary
    :param args: arguments provided to split
    :param value: value to run split on
    :param lookup_value: value required for 'lookup' when 'lookup' is used as a sub-expression
    :return: True if value passes split, error message on False"""
    split_char = args[0]
    split_count = int(args[1])
    value_split = value.split(split_char)
    if len(value_split) != split_count:
        return False, f"'{args[1]}' must have {split_count} elements when split on '{split_char}'"
    errs = []
    datatypes = config["datatypes"]
    x = 0
    while x < split_count:
        v = value_split[x]
        expr = args[x + 2]
        if expr["type"] == "datatype":
            datatype = expr["name"]
            success, _ = is_datatype(datatypes, datatype, v)
            if not success:
                errs.append(f"sub-value '{v}' must be of datatype '{datatype}'")
        else:
            success, err = run_function(config, expr, v, lookup_value=lookup_value)
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
    direct = False
    if len(args) == 3 and args[2]["value"].lower() == "true":
        direct = True
    if has_ancestor(tree, ancestor, value, direct=direct):
        return True, None
    else:
        if direct:
            return False, f"'{value}' must be a direct subclass of '{ancestor}' from {tree_name}"
        return False, f"'{value}' must be equal to or under '{ancestor}' from {tree_name}"


# ---- VALIDATION ----


def collect_distinct_messages(table_details, output_dir, table, messages):
    """Collect distinct messages and write the rows with distinct messages to a new table. The
    new table will be [table_name]_distinct. Return the distinct messages with updated locations in
    the new table.

    :param table_details: table name -> details (rows, fields)
    :param output_dir: directory to write distinct tables to
    :param table: path to table with messages
    :param messages: all messages from the table
    :return: updated distinct messages from the table
    """
    distinct_messages = {}
    for msg in messages:
        if msg["message"] not in distinct_messages:
            distinct_messages[msg["message"]] = msg

    logging.info(f"{len(distinct_messages)} distinct error(s) found in {table}")

    message_rows = defaultdict(list)
    for msg in distinct_messages.values():
        row = int(msg["cell"][1:])
        message_rows[row].append(msg)
    messages = []

    basename = os.path.basename(table)
    table_name = os.path.splitext(basename)[0]
    table_ext = os.path.splitext(basename)[1]
    sep = "\t"
    if table_ext == ".csv":
        sep = ","
    output = os.path.join(output_dir, f"{table_name}_distinct{table_ext}")
    logging.info("writing rows with errors to " + output)

    fields = table_details[table_name]["fields"]
    rows = table_details[table_name]["rows"]
    with open(output, "w") as g:
        writer = csv.DictWriter(g, fields, delimiter=sep, lineterminator="\n")
        writer.writeheader()
        row_idx = 2
        new_idx = 2
        for row in rows:
            if row_idx in message_rows.keys():
                writer.writerow(row)
                for msg in message_rows[row_idx]:
                    msg["table"] = table_name + "_distinct"
                    msg["cell"] = msg["cell"][0:1] + str(new_idx)
                    messages.append(msg)
                new_idx += 1
            row_idx += 1
    return messages


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
    table_name = os.path.splitext(os.path.basename(table))[0]
    row_idx = row_start
    for row in table_details[table_name]["rows"]:
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


def write_messages(output, messages):
    """Write validation messages to a file.

    :param output: path to write errors to
    :param messages: list of dictionaries of validation messages
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
        writer.writerows(messages)


def get_config_from_tables(paths, row_start=2):
    """Create a VALVE config dict from a list of paths.

    :param paths: input paths
    :param row_start: row number that contents to validate start on
    :return: config dict
    """
    datatype_table = None
    field_table = None
    rule_table = None
    tables = []
    for dir_or_file in paths:
        files = []
        if os.path.isdir(dir_or_file):
            for f in os.listdir(dir_or_file):
                path = os.path.join(dir_or_file, f)
                if not path.endswith(".csv") and not path.endswith(".tsv"):
                    continue
                files.append(path)
        else:
            files.append(dir_or_file)

        for f in files:
            fname = os.path.splitext(os.path.basename(f))[0]
            if fname == "datatype":
                if datatype_table:
                    raise RuntimeError(
                        f"More than one 'datatype' table found: {f}, {datatype_table}"
                    )
                datatype_table = f
            elif fname == "field":
                if field_table:
                    raise RuntimeError(f"More than one 'field' table found: {f}, {field_table}")
                field_table = f
            elif fname == "rule":
                if rule_table:
                    raise RuntimeError(f"More than one 'rule' table found: {f}, {rule_table}")
                rule_table = f
            else:
                tables.append(f)

    if not datatype_table:
        raise RuntimeError("A 'datatype' TSV or CSV must be included in the input directory(ies)")
    if not field_table:
        raise RuntimeError("A 'field' TSV or CSV must be included in the input directory(ies)")
    if not tables:
        raise RuntimeError(
            "Additional tables to validate must be included in the input directory(ies)"
        )

    setup_errors = []
    table_details = get_table_details(tables, row_start=row_start)

    datatypes, add_errors = read_datatype_table(datatype_table)
    setup_errors.extend(add_errors)

    config = {"table_details": table_details, "datatypes": datatypes}

    table_fields, trees, add_errors = read_field_table(config, field_table, row_start=row_start)
    setup_errors.extend(add_errors)

    config["trees"] = trees

    table_rules = {}
    if rule_table:
        table_rules, add_errors = read_rule_table(config, rule_table)
        setup_errors.extend(add_errors)

    return {
        "datatypes": datatypes,
        "table_details": table_details,
        "table_fields": table_fields,
        "table_rules": table_rules,
        "trees": trees,
        "errors": setup_errors,
    }


def validate(o, row_start=2, distinct_messages=None):
    """Main VALVE method.

    :param o: inputs or config object
    :param row_start: row number that contents to validate start on
    :param distinct_messages: output directory to write distinct message tables to, or None
    :return: True if VALVE completed (with or without errors), False if VALVE configuration failed
    """

    if isinstance(o, list):
        config = get_config_from_tables(o, row_start=row_start)
    elif isinstance(o, dict):
        config = o
    else:
        raise Exception(
            "`validate` accepts a list of paths or a config object, not " + type(o).__name__
        )

    table_details = config["table_details"]
    table_fields = config["table_fields"]
    table_rules = config["table_rules"]
    setup_errors = config["errors"]

    # Check for true setup errors and stop process if they exist
    for e in setup_errors:
        if "kill" in e:
            logging.critical(f"VALVE setup failed with {len(setup_errors)} errors!")
            return setup_errors

    errors = []
    for table in table_details.keys():
        logging.info("validating " + table)
        tname = os.path.splitext(os.path.basename(table))[0]
        fields = table_fields.get(tname, [])
        rules = table_rules.get(tname, [])

        # Validate and return errors
        add_errors = validate_table(config, table, fields, rules, row_start=row_start)

        # Add any non-kill errors that were found during setup
        add_errors.extend([x for x in setup_errors if x["table"] == tname])
        logging.info(f"{add_errors} errors found in {table}")

        if add_errors and distinct_messages:
            # Update errors to only be distinct messages in a new table
            table_path = table_details[table]["path"]
            update_errors = collect_distinct_messages(
                table_details, distinct_messages, table_path, add_errors
            )
            errors.extend(update_errors)
        elif not distinct_messages:
            errors.extend(add_errors)
    if errors:
        logging.error(f"VALVE completed with {len(errors)} problems found!")
    return errors


def main():
    p = ArgumentParser()
    p.add_argument("paths", help="Paths to input directories and/or files", nargs="+")
    p.add_argument(
        "-d",
        "--distinct",
        help="Collect each distinct error messages and write to a table in provided directory",
    )
    p.add_argument(
        "-r", "--row-start", help="Index of first row in tables to validate", type=int, default=2
    )
    p.add_argument("-o", "--output", help="CSV or TSV to write error messages to", required=True)
    args = p.parse_args()

    messages = validate(args.paths, row_start=args.row_start, distinct_messages=args.distinct)
    write_messages(args.output, messages)
    for msg in messages:
        if "level" in msg and msg["level"].lower() == "error":
            sys.exit(1)


if __name__ == "__main__":
    main()
