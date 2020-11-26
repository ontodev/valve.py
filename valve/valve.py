import csv
import inspect
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


# ---- MISC HELPERS ----


def build_datatype_ancestors(datatypes, datatype):
    """Recursively build a list of ancestor datatypes for a given datatype.

    :param datatypes: map of datatype name -> details
    :param datatype: datatype to get ancestors of
    :return: list of ancestor datatypes
    """
    ancestors = []
    parent = datatypes[datatype].get("parent")
    if parent:
        ancestors.append(parent)
        ancestors.extend(build_datatype_ancestors(datatypes, parent))
    return ancestors


def error(config, table, column, row_idx, message, level="ERROR", suggestion=None):
    row_start = config["row_start"]
    col_idx = config["table_details"][table]["fields"].index(column)
    d = {
        "table": table,
        "cell": idx_to_a1(row_start + row_idx, col_idx + 1),
        "level": level,
        "message": message,
    }
    if suggestion:
        d["suggestion"] = suggestion
    return d


def get_indexes(seq, item):
    """Return all indexes of an item in a sequence.

    :param seq: sequence to check
    :param item: item to find indexes of
    :return: list of indexes
    """
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def has_ancestor(tree, ancestor, node, direct=False):
    """Check whether a node has an ancestor (or self) in a tree.

    :param tree: a dictionary from children to sets of parents
    :param ancestor: the ancestor to look for
    :param node: the node to start from
    :param direct: if True, only look at direct parents, not full ancestors
    :return: True if it has the ancestor, False otherwise
    """
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


def parsed_to_str(condition):
    """Convert a parsed condition back to its original string.

    :param condition: parsed condition to convert
    :return: string version of condition
    """
    cond_type = condition["type"]
    if cond_type == "string":
        val = condition["value"]
        if " " in val:
            return f'"{val}"'
        return val
    if cond_type == "field":
        table = condition["table"]
        col = condition["column"]
        if " " in table:
            table = f'"{table}"'
        if " " in col:
            col = f'"{col}"'
        return f"{table}.{col}"
    if cond_type == "named_arg":
        name = condition["name"]
        val = condition["value"]
        if " " in val:
            val = f'"{val}"'
        return f"{name}={val}"
    if cond_type == "regex":
        pattern = condition["pattern"]
        flags = condition["flags"]
        if "replace" in condition:
            replace = condition["replace"]
            return f"s/{pattern}/{replace}/{flags}"
        return f"/{pattern}/{flags}"
    if cond_type == "function":
        args = []
        for arg in condition["args"]:
            args.append(parsed_to_str(arg))
        name = condition["name"]
        args = ", ".join(args)
        return f"{name}({args})"
    raise Exception("Unknown condition type: " + cond_type)


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
    datatypes = {}
    with open(datatype_table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames
        missing = list(set(datatype_headers) - set(headers))
        if missing:
            raise Exception("Missing required column for 'datatype' table: " + ", ".join(missing))
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
        if not level_is_valid(level):
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


def read_field_table(config, field_table):
    """Build a dictionary of fields.

    :param config: valve config dictionary containing table_details
    :param field_table: path to the 'field' table
    :return: dictionary of table-name -> field-name -> types
    """
    row_start = config["row_start"]
    errors = []
    sep = "\t"
    if field_table.endswith("csv"):
        sep = ","
    table_name = os.path.splitext(os.path.basename(field_table))[0]

    table_details = config["table_details"]

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
            table_to_validate = row["table"]
            if table_to_validate not in table_details and table_to_validate != "*":
                errors.append(
                    {
                        "table": table_name,
                        "cell": idx_to_a1(idx, headers.index("table") + 1),
                        "rule": "missing table",
                        "message": f"table '{table_to_validate}' does not exist in inputs",
                        "kill": True,
                    }
                )
                continue
            column = row["column"]
            if table_to_validate != "*":
                if column not in table_details[table_to_validate]["fields"]:
                    errors.append(
                        {
                            "table": table_name,
                            "cell": idx_to_a1(idx, headers.index("table") + 1),
                            "rule": "missing column",
                            "message": f"'{column}' does not exist in table '{table_to_validate}'",
                            "kill": True,
                        }
                    )
                    continue

            # Dict of field name -> its type (parsed)
            if table_to_validate in table_fields:
                field_types = table_fields[table_to_validate]
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
                err = check_condition(config, parsed_type)
                if err:
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
                    tree_opts, err = get_tree_options(parsed_type)
                    if err:
                        errors.append(
                            {
                                "table": table_name,
                                "cell": idx_to_a1(idx + row_start, headers.index(column) + 1),
                                "rule": "`tree` function error",
                                "level": "ERROR",
                                "message": err,
                                "kill": True,
                            }
                        )
                        continue
                    tree, errs = build_tree(
                        config,
                        idx,
                        table_to_validate,
                        column,
                        tree_opts["child_column"],
                        add_tree_name=tree_opts["add_tree_name"],
                        split_char=tree_opts["split_char"],
                    )
                    errors.extend(errs)
                    if tree:
                        trees[f"{table_to_validate}.{column}"] = tree
                        config["trees"] = trees
                    continue

                field_types[column] = {
                    "parsed": parsed_type,
                    "field ID": idx,
                }
                table_fields[table_to_validate] = field_types

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
            table_to_validate = row["table"]
            table_loc = idx_to_a1(idx, headers.index("table") + 1)
            when_column = row["when column"]
            when_column_loc = idx_to_a1(idx, headers.index("when column") + 1)

            if table_to_validate not in table_columns.keys():
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
                if when_column not in table_columns[table_to_validate]:
                    errors.append(
                        {
                            "table": table_name,
                            "cell": when_column_loc,
                            "rule": "unknown column",
                            "message": f"'{when_column}' must exist in '{table_to_validate}'",
                            "kill": True,
                        }
                    )

            # Validate the when condition
            when_condition = row["when condition"]
            parsed_when_condition = parse(when_condition)
            err = check_condition(config, parsed_when_condition)
            if err:
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
            if table_to_validate in table_rules:
                column_rules = table_rules[table_to_validate]
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
            err = check_condition(config, parsed_then_condition)
            if err:
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
            if not level_is_valid(level):
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
            if then_column not in table_columns[table_to_validate]:
                errors.append(
                    {
                        "table": table_name,
                        "cell": then_column_loc,
                        "rule": "unknown column",
                        "message": f"'{then_column}' must exist in '{table_to_validate}'",
                        "kill": True,
                    }
                )

            # Add this condition to the dicts
            rules.append(
                {
                    "when_condition": parsed_when_condition,
                    "column": then_column,
                    "then_condition": parsed_then_condition,
                    "level": level,
                    "message": row.get("description", None),
                    "rule ID": idx,
                }
            )
            column_rules[when_column] = rules
            table_rules[table_to_validate] = column_rules
    return table_rules, errors


# ---- INPUT VALIDATION ----


def build_tree(
    config, fn_row_idx, table_name, parent_column, child_column, add_tree_name=None, split_char="|",
):
    """Build a hierarchy for the `tree` function while validating the values.

    :param config: valve config dictionary
    :param fn_row_idx: row of tree function in 'field' table
    :param table_name: table name
    :param parent_column: name of column that 'Parent' values are in
    :param child_column: name of column that 'Child' values are in
    :param add_tree_name: optional name of tree to add to
    :param split_char: character to split parent values on
    :return: map of child -> parents, list of errors (if any)
    """
    errors = []
    table_details = config["table_details"]
    row_start = config["row_start"]
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


def check_condition(config, parsed_condition):
    """Validate a condition. Return an error message if not valid.

    :param config: valve config dictionary
    :param parsed_condition: parsed condition to validate
    :return: None on success, error message on failure
    """
    datatypes = config["datatypes"]
    cond_type = parsed_condition["type"]

    if cond_type == "string":
        dt = parsed_condition["value"]
        if dt not in datatypes.keys():
            return f"datatype '{dt}' must be defined in the datatype table"
    elif cond_type == "function":
        return check_function(config, parsed_condition)
    else:
        raise Exception("Unknown condition type: " + cond_type)
    return None


def check_expression(config, funct_name, pos, arg):
    """Check that an argument is a function or datatype. Return an error message if not.

    :param config: valve config dictionary
    :param funct_name: name of function currently being validated
    :param pos: position of this argument
    :param arg: argument to validate
    :return: None on success, error message on failure
    """
    if not is_type(arg, ["string", "function"]):
        return False, f"`{funct_name}` argument {pos} ({arg}) must be a valid datatype or function"
    if arg["type"] == "string":
        datatypes = config["datatypes"]
        dt_name = arg["value"]
        if dt_name not in datatypes:
            return f"`{funct_name}` argument {pos} ({dt_name}) must be a defined datatype"
    else:
        err = check_function(config, arg)
        if err:
            return f"`{funct_name}` argument {pos} must be a valid datatype or function: " + err
    return None


def check_function(config, function):
    """Validate a function and return an error message if not valid.

    :param config: valve config dictionary
    :param function: parsed function as dictionary
    :return: None on success or error message
    """
    functions = config["functions"]
    funct_name = function["name"]
    # Add tree & distinct to functions (resolved at top-level)
    allowed_funct_names = list(functions.keys()) + ["tree"]
    if funct_name not in allowed_funct_names:
        return f"function name ({funct_name}) must be one of: " + ",".join(allowed_funct_names)

    if funct_name not in builtin_functions.keys():
        # No validation for args to custom functions
        return None

    table_details = config["table_details"]

    # Special validation for each builtin function
    args = function["args"]
    if funct_name == "any":
        # any(expr, expr, ...)
        x = 1
        for arg in args:
            err = check_expression(config, "any", x, arg)
            if err:
                return err
            x += 1

    elif funct_name == "CURIE":
        # CURIE(table.column)
        x = 1
        for arg in args:
            if not is_type(arg, ["string", "field"]):
                return f"`CURIE` argument {x} must be a table.column pair or a string"
            if arg["type"] == "field":
                return check_table_column(table_details, "CURIE", x, arg)
            x += 1

    elif funct_name == "distinct":
        # distinct(expr, [table-column, ...])
        err = check_expression(config, "distinct", 1, args[0])
        if err:
            return err
        if len(args) > 1:
            arg_idx = 2
            for arg in args[1:]:
                if not is_type(arg, ["field"]):
                    return f"`distinct` argument {arg_idx} must be a table.column pair"
                arg_idx += 1

    elif funct_name == "in":
        # in("x", "y", "z", ...)
        x = 1
        for arg in args:
            if not is_type(arg, ["string", "field"]):
                return f"`in` argument {x} must be a table.column pair or a string"
            if arg["type"] == "field":
                return check_table_column(table_details, "in", x, arg)
            x += 1

    elif funct_name == "not":
        # not(expr, expr, ...)
        x = 1
        for arg in args:
            err = check_expression(config, "any", x, arg)
            if err:
                return err
            x += 1

    elif funct_name == "sub":
        # sub(regex, expr)
        if len(args) != 2:
            return "`sub` must have exactly two arguments"
        if not is_type(args[0], ["regex"]):
            return "`sub` argument 1 must be a regex pattern"
        flag_str = args[0]["flags"]
        if not re.match(r"[agix]+", flag_str):
            return "`sub` regex flag(s) must be one or more of: a, g, i, or x"

        # second value must be a valid function or datatype
        err = check_expression(config, "sub", 2, args[1])
        if err:
            return err

    elif funct_name == "list":
        # list(split, expr)
        if len(args) != 2:
            # must have exactly two values
            return "`list` must have exactly two arguments"
        if not is_type(args[0], ["string"]):
            return "`list` argument 1 must be a string"

        # second value must be a valid function or datatype
        err = check_expression(config, "list", 2, args[1])
        if err:
            return err

    elif funct_name == "lookup":
        # lookup(table, column, column)
        if len(args) != 3:
            return "`lookup` must have exactly three arguments"
        if not is_type(args[0], ["string"]):
            return "`lookup` argument 1 must be a table name"
        table = args[0]["value"]
        if table not in table_details:
            return f"`lookup` argument 1 table name ({table}) must be a table in the inputs"
        headers = table_details[table]["fields"]
        x = 1
        while x < 3:
            arg = args[x]
            if not is_type(arg, ["string"]) or arg["value"] not in headers:
                return f"`lookup` argument {x + 1} must be a column name in '{table}'"
            x += 1

    elif funct_name == "under":
        # under(tree, value)
        if len(args) != 2 and len(args) != 3:
            # can have two or three args
            return "`under` must have either two or three arguments"
        tree_loc = args[0]
        if not is_type(tree_loc, ["field"]):
            return f"`under` argument 1 must be a table.column pair"

        tree_name = f'{tree_loc["table"]}.{tree_loc["column"]}'
        if "trees" not in config:
            return f"a `tree` for {tree_name} must be defined in order to use `under`"
        trees = config["trees"]

        if tree_name not in trees:
            # tree must have already been defined
            return f"`under` argument 1 '{tree_name}' must be defined as a tree in 'field'"
        if not is_type(args[1], ["string"]):
            # second value must be a string
            return "`under` argument 2 must be a string"
        top_level = args[1]["value"]

        if len(args) == 3:
            if not is_type(args[2], ["named_arg"]):
                return "if provided, `under` argument 3 must be `direct=bool`"
            if args[2]["name"] != "direct":
                return "`under` only accepts named argument `direct`"
            if args[2]["value"].lower() not in ["true", "false"]:
                return "in `under` argument 3, the value of `direct=` must be true or false"

        tree = trees[tree_name]
        tree_values = list(tree.keys())
        tree_values.extend(list(itertools.chain.from_iterable(tree.values())))
        if top_level not in tree_values:
            # second value must exist in tree
            return f"`under` argument 2 ({top_level}) must exist in tree {tree_name}"

    return None


def check_table_column(table_details, fn_name, arg_pos, arg):
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
        return f"`{fn_name}` argument {arg_pos} must use a valid table ({table_name} not in inputs)"
    if column_name not in table_details[table_name]["fields"]:
        return (
            f"`{fn_name}` argument {arg_pos} must use a field from {table_name} "
            f"('{column_name}' not in fields)"
        )
    return None


def get_tree_options(tree_function):
    """Validate a 'tree' field type and return the options from the args.

    :param tree_function: the parsed field type (tree function)
    :return: options & error message (on failure)
    """
    args = tree_function["args"]
    if 1 > len(args) > 3:
        return None, "the `tree` function must have between one and three arguments"

    # first arg is column
    child_column = args[0]
    if not is_type(child_column, ["string"]):
        return None, "the first argument of the `tree` function must be a column name"

    # Parse the rest of the args
    add_tree_name = None
    split_char = None
    if args:
        x = 1
        while x < len(args):
            arg = args[x]
            if "name" in arg and arg["name"] == "split":
                split_char = arg["value"]
            elif "table" in arg:
                add_tree_name = f'{arg["table"]}.{arg["column"]}'
            else:
                return None, f"`tree` argument {x + 1} must be table.column pair or split=CHAR"
            x += 1
    return (
        {
            "child_column": child_column["value"],
            "add_tree_name": add_tree_name,
            "split_char": split_char,
        },
        None,
    )


def is_type(arg, types):
    """Validate that an arg is one of the given types.

    :param arg: argument to validate
    :param types: valid type or types as list
    :return: True if arg is dict and has "type" in the list
    """
    if not isinstance(arg, dict):
        return False
    for t in types:
        if arg["type"] == t:
            return True
    return False


def level_is_valid(level):
    """Validate a level entry. The level must be one of (case-insensitive): error, warn, or info.

    :param level: logging level
    :return: True if level is valid or False
    """
    if not level or level == "":
        return False
    elif level.lower() not in ["error", "warn", "info"]:
        return False
    return True


# ---- CONDITION VALIDATION ----


def check_value(config, condition, table, column, row_idx, value):
    """Determine if the value meets the condition.

    :param config: valve config dictionary
    :param condition: parsed condition to check (as dict)
    :param table: table name
    :param column: column name
    :param row_idx: row number from values
    :param value: value to check
    :return: List of messages (empty on success)
    """
    condition_type = condition["type"]
    datatypes = config["datatypes"]
    if condition_type == "string":
        datatype = condition["value"]
        # Check if condition is met, potentially get a replacement
        value_meets_condition, replace = is_datatype(datatypes, datatype, value)
        if value_meets_condition is False:
            unparsed_condition = parsed_to_str(condition)
            message = f"'{value}' must be of datatype '{unparsed_condition}'"
            return [error(config, table, column, row_idx, message)]

    elif condition_type == "function":
        return run_function(config, condition, table, column, row_idx, value)

    else:
        # This should be prevented in validate_condition
        raise Exception("unknown condition type: " + condition_type)
    return []


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
                if fix:
                    parsed = parse(fix)
                    # TODO - handle flags
                    pattern = parsed["pattern"]
                    replace = parsed["replace"]
                    return False, re.sub(pattern, replace, value)
                return False, None
    return True, None


def run_function(config, function, table, column, row_idx, value):
    """Run a VALVE function for the provided value.

    :param config: valve config dictionary
    :param function: function to run (as parsed dictionary)
    :param table: table to run distinct on
    :param column: column to run distinct on
    :param row_idx: current row number
    :param value: value to run function on
    :return: List of messages (empty on success)
    """
    functions = config["functions"]
    funct_name = function["name"]
    args = function["args"]
    if funct_name not in functions:
        raise Exception("Unknown function: " + funct_name)
    fn = functions[funct_name]
    if not fn:
        raise Exception(f"Function '{funct_name}' must be defined")

    return fn(config, args, table, column, row_idx, value)


# ---- VALVE FUNCTIONS ----


def any_of(config, args, table, column, row_idx, value):
    """Method for the VALVE 'any' function.

    :param config: valve config dictionary
    :param args: arguments provided to not
    :param table: table to run distinct on
    :param column: column to run distinct on
    :param row_idx: current row number
    :param value: value to run any on
    :return: List of messages (empty on success)
    """
    conditions = []
    for arg in args:
        messages = check_value(config, arg, table, column, row_idx, value)
        if not messages:
            # As long as one is met, this passes
            return []
        conditions.append(parsed_to_str(arg))
    # If we get here, no condition was met
    message = f"'{value}' must meet one of: " + ", ".join(conditions)
    return [error(config, table, column, row_idx, message)]


def CURIE(config, args, table, column, row_idx, value):
    """Method for the VALVE 'CURIE' function. The value must be a CURIE and the prefix of the value
    must be in the table.column pair or string defined by the arg (1+ args)

    :param config: valve config dictionary
    :param args: arguments provided to CURIE
    :param table: table to run distinct on
    :param column: column to run distinct on
    :param row_idx: current row number
    :param value: value to run CURIE on
    :return: List of messages (empty on success)
    """
    table_details = config["table_details"]
    prefixes = []
    # Get prefixes from args - either strings or table.column pairs
    for arg in args:
        if arg["type"] == "string":
            prefixes.append(arg["value"])
            continue
        table_name = arg["table"]
        column_name = arg["column"]
        for row in table_details[table_name]["rows"]:
            prefixes.append(row[column_name])
    if ":" not in value:
        message = f"'{value}' must be a CURIE"
        return [error(config, table, column, row_idx, message)]
    value_prefix = value.split(":")[0]
    if value_prefix not in prefixes:
        message = f"prefix '{value_prefix}' must be one of: " + ", ".join(prefixes)
        return [error(config, table, column, row_idx, message)]
    return []


def distinct(config, args, table, column, row_idx, value):
    """Method for the VALVE 'distinct' function. This is run over all rows, rather than one value.

    :param config: valve config dictionary
    :param args: arguments provided to distinct
    :param table: table to run distinct on
    :param column: column to run distinct on
    :param row_idx: current row number
    :param value: value to run distinct on
    :return: List of messages (empty on success)
    """
    table_details = config["table_details"]
    row_start = config["row_start"]
    base_rows = table_details[table]["rows"]
    base_headers = table_details[table]["fields"]
    base_values = [x.get(column, None) for x in base_rows]

    duplicate_locs = set()
    value_indexes = get_indexes(base_values, value)
    if len(value_indexes) > 1:
        col_idx = base_headers.index(column) + 1
        for idx in value_indexes:
            if idx == row_idx:
                continue
            duplicate_locs.add(f"{table}:{idx_to_a1(idx + row_start, col_idx)}")

    # extra table-columns to check
    if len(args) > 1:
        for itm in args[1:]:
            t = itm["table"]
            c = itm["column"]
            trows = table_details[t]["rows"]
            theaders = table_details[t]["fields"]
            tvalues = [x.get(c, None) for x in trows]
            if value in tvalues:
                value_indexes = get_indexes(tvalues, value)
                col_idx = theaders.index(c) + 1
                for idx in value_indexes:
                    duplicate_locs.add(f"{t}:{idx_to_a1(idx + row_start, col_idx)}")

    # Create the error messages
    if duplicate_locs:
        message = f"'{value}' must be distinct with value(s) at: " + ", ".join(duplicate_locs)
        return [error(config, table, column, row_idx, message)]
    return []


def in_set(config, args, table, column, row_idx, value):
    """Method for the VALVE 'in' function. The value must be one of the arguments.

    :param config: valve config dictionary
    :param args: arguments provided to in
    :param table: table name
    :param column: column name
    :param row_idx: row number from values
    :param value: value to run in on
    :return: List of messages (empty on success)
    """
    table_details = config["table_details"]
    allowed = []
    for arg in args:
        if arg["type"] == "string":
            arg_val = arg["value"]
            if value == arg_val:
                return []
            allowed.append(f'"{arg_val}"')
        else:
            table_name = arg["table"]
            column_name = arg["column"]
            source_rows = table_details[table_name]["rows"]
            allowed_values = [x[column_name] for x in source_rows if column_name in x]
            if value in allowed_values:
                return []
            allowed.append(f"{table_name}.{column_name}")
    message = f"'{value}' must be in: " + ", ".join(allowed)
    return [error(config, table, column, row_idx, message)]


def not_any_of(config, args, table, column, row_idx, value):
    """Method for the VALVE 'not' function.

    :param config: valve config dictionary
    :param args: arguments provided to not
    :param table: table name
    :param column: column name
    :param row_idx: row number from values
    :param value: value to run not on
    :return: List of messages (empty on success)
    """
    for arg in args:
        messages = check_value(config, arg, table, column, row_idx, value)
        if not messages:
            # If any condition *is* met (no errors), this fails
            unparsed = parsed_to_str(arg)
            message = f"'{value}' must not be '{unparsed}'"
            if unparsed == "blank":
                message = f"value must not be blank"
            return [error(config, table, column, row_idx, message)]
    return []


def substitute(config, args, table, column, row_idx, value):
    """Method for the VALVE 'sub' function.

    Substitute match with replacement, then evaluate the expression.

    :param config: valve config dictionary
    :param args: arguments provided to list
    :param table: table name
    :param column: column name
    :param row_idx: row number from values
    :param value: value to run list on
    :return: List of messages (empty on success)
    """
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
    if subfunc["type"] == "string":
        datatypes = config["datatypes"]
        datatype = subfunc["value"]
        value_is_datatype = is_datatype(datatypes, datatype, value)[0]
        if not value_is_datatype:
            message = f"substituted value '{value}' must be of datatype {datatype}"
            return [error(config, table, column, row_idx, message)]
        return []
    else:
        return run_function(config, subfunc, table, column, row_idx, value)


def for_each_list(config, args, table, column, row_idx, value):
    """Method for the VALVE 'list' function.

    Split the value on the first argument and perform the function or datatype check provided as the
    second argument on all values.

    :param config: valve config dictionary
    :param args: arguments provided to list
    :param table: table name
    :param column: column name
    :param row_idx: row number from values
    :param value: value to run list on
    :return: List of messages (empty on success)
    """
    split_char = args[0]["value"]
    expr = args[1]
    datatypes = config["datatypes"]
    errs = []
    for v in value.split(split_char):
        if expr["type"] == "string":
            datatype = expr["value"]
            success, _ = is_datatype(datatypes, datatype, v)
            if not success:
                errs.append(f"sub-value '{v}' must be of datatype '{datatype}'")
        else:
            err = run_function(config, expr, table, column, row_idx, v)
            if err:
                errs.append(err)
    if errs:
        message = "\n".join(errs)
        return [error(config, table, column, row_idx, message)]
    return []


def lookup(config, args, table, column, row_idx, value):
    """Method for VALVE 'lookup' function.

    The lookup value is found in the first column (second argument), then the allowed
    value is retrieved from the second column (third argument) pair on the same row. The
    provided value must be exactly the same as the found value.

    :param config: valve config dictionary
    :param args: arguments provided to lookup
    :param table: table name
    :param column: column name
    :param row_idx: row number from values
    :param value: value to run lookup on
    :return: List of messages (empty on success)
    """
    table_details = config["table_details"]
    col_idx = table_details[table]["fields"].index(column)
    table_rules = config["table_rules"][table]
    lookup_value = None
    for when_column, rules in table_rules.items():
        for rule in rules:
            if rule["column"] != column or rule["then_condition"].get("name", "") != "lookup":
                continue
            lookup_value = table_details[table]["rows"][row_idx][when_column]
            break

    if not lookup_value:
        raise Exception(f"Unable to find lookup function for {table}.{column} in rule table")

    search_table = args[0]["value"]
    search_column = args[1]["value"]
    return_column = args[2]["value"]
    for row in table_details[search_table]["rows"]:
        maybe_value = row[search_column]
        if maybe_value == lookup_value:
            expected = row[return_column]
            if value != expected:
                message = f"'{value}' must be '{expected}'"
                return [error(config, table, column, row_idx, message, suggestion=expected)]
            return []
    message = f"'{value}' must present in {search_table}.{return_column}"
    return [error(config, table, column, row_idx, message)]


def under(config, args, table, column, row_idx, value):
    """Method for VALVE 'under' function.

    Retrieve the tree defined by the first argument (a table.column pair). The value must be a
    descendant of the second argument.

    :param config: valve config dictionary
    :param args: arguments provided to under
    :param table: table name
    :param column: column name
    :param row_idx: row number from values
    :param value: value to run under on
    :return: List of messages (empty on success)
    """
    trees = config["trees"]
    table_name = args[0]["table"]
    column_name = args[0]["column"]
    tree_name = f"{table_name}.{column_name}"
    if tree_name not in trees:
        # This has already been validated for CLI users
        raise Exception(f"A tree for {tree_name} is not defined")
    tree = trees[tree_name]
    ancestor = args[1]["value"]
    direct = False
    if len(args) == 3 and args[2]["value"].lower() == "true":
        direct = True
    if has_ancestor(tree, ancestor, value, direct=direct):
        return []

    message = f"'{value}' must be equal to or under '{ancestor}' from {tree_name}"
    if direct:
        message = f"'{value}' must be a direct subclass of '{ancestor}' from {tree_name}"
    return [error(config, table, column, row_idx, message)]


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


def validate_table(config, table):
    """Run VALVE validation on a table.

    :param config: valve config dictionary
    :param table: path to table
    :return: list of errors
    """
    errors = []
    table_name = os.path.splitext(os.path.basename(table))[0]
    table_details = config["table_details"]

    fields = config["table_fields"].get(table, {})
    fields.update(config.get("*", {}))
    rules = config["table_rules"].get(table, {})
    rules.update(config.get("*", {}))

    row_idx = 0
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
                # all values in this field must match the type
                messages = check_value(config, parsed_type, table_name, field, row_idx, value)
                if messages:
                    field_id = fields[field]["field ID"]
                    for m in messages:
                        m.update({"rule ID": "field:" + str(field_id), "level": "ERROR"})
                        errors.append(m)

            # Check for rules
            if field in rules:
                # Check if the value meets any of the conditions
                for rule in rules[field]:
                    when_condition = rule["when_condition"]
                    # Run meets_condition without logging
                    # as the then-cond check is only run if the value matches the type
                    messages = check_value(
                        config, when_condition, table_name, field, row_idx, value
                    )
                    if not messages:
                        # The "when" value meets the condition - validate the "then" value
                        then_column = rule["column"]

                        # Retrieve the "then" value to check if it meets the "then condition"
                        then_value = row[then_column]
                        messages = check_value(
                            config,
                            rule["then_condition"],
                            table_name,
                            then_column,
                            row_idx,
                            then_value,
                        )
                        if messages:
                            for m in messages:
                                msg = (
                                    f"because '{value}' is '{parsed_to_str(when_condition)}', "
                                    + m["message"]
                                )
                                m.update(
                                    {
                                        "rule ID": "rule:" + str(rule["rule ID"]),
                                        "rule": rule["message"],
                                        "level": rule["level"],
                                        "message": msg,
                                    }
                                )
                                errors.append(m)
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


def get_config_from_tables(paths, row_start=2, functions=None):
    """Create a VALVE config dict from a list of paths.

    :param paths: input paths
    :param row_start: row number that contents to validate start on
    :param functions: dict of function name -> functions to add
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

    if not functions:
        functions = builtin_functions
    else:
        for funct_name, function in functions.items():
            validate_custom_function(funct_name, function)
        functions.update(builtin_functions)

    setup_errors = []
    table_details = get_table_details(tables, row_start=row_start)

    datatypes, add_errors = read_datatype_table(datatype_table)
    setup_errors.extend(add_errors)

    # Init config dict
    config = {
        "table_details": table_details,
        "datatypes": datatypes,
        "functions": functions,
        "row_start": row_start,
    }

    table_fields, trees, add_errors = read_field_table(config, field_table)
    setup_errors.extend(add_errors)

    # Trees may be used in rules, add to config
    config["trees"] = trees

    table_rules = {}
    if rule_table:
        table_rules, add_errors = read_rule_table(config, rule_table)
        setup_errors.extend(add_errors)

    # Add the rest of the configuration to dict
    config.update(
        {"table_fields": table_fields, "table_rules": table_rules, "errors": setup_errors}
    )
    return config


def validate_custom_function(funct_name, function):
    if funct_name in builtin_functions:
        raise Exception(f"Cannot use builtin function name '{funct_name}'")
    params = list(inspect.signature(function).parameters.keys())
    if params[0] != "config":
        raise Exception(f"'{funct_name}' argument 1 must be config")
    if params[1] != "args":
        raise Exception(f"'{funct_name}' argument 2 must be args")
    if params[2] != "table":
        raise Exception(f"'{funct_name}' argument 3 must be value")
    if params[3] != "column":
        raise Exception(f"'{funct_name}' argument 4 must be value")
    if params[4] != "row_idx":
        raise Exception(f"'{funct_name}' argument 5 must be value")
    if params[5] != "value":
        raise Exception(f"'{funct_name}' argument 6 must be value")


def validate(o, row_start=2, distinct_messages=None, functions=None):
    """Main VALVE method.

    :param o: inputs or config object
    :param row_start: row number that contents to validate start on
    :param distinct_messages: output directory to write distinct message tables to, or None
    :param functions: dict of function name -> function to add to builtins
    :return: True if VALVE completed (with or without errors), False if VALVE configuration failed
    """

    if isinstance(o, list):
        config = get_config_from_tables(o, row_start=row_start, functions=functions)
    elif isinstance(o, dict):
        config = o
        # Update any functions with the builtins
        if "functions" in config:
            functions = config["functions"]
            for funct_name, function in functions.items():
                validate_custom_function(funct_name, function)
            functions.update(builtin_functions)
            config["functions"] = functions
        else:
            config["functions"] = builtin_functions
        if "row_start" not in config:
            config["row_start"] = row_start
    else:
        raise Exception(
            "`validate` accepts a list of paths or a config object, not " + type(o).__name__
        )

    table_details = config["table_details"]
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

        # Validate and return errors
        add_errors = validate_table(config, table)

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


# Supported functions (distinct & tree are handled at top-level)
builtin_functions = {
    "any": any_of,
    "CURIE": CURIE,
    "distinct": distinct,
    "in": in_set,
    "not": not_any_of,
    "sub": substitute,
    "list": for_each_list,
    "lookup": lookup,
    "under": under,
}


if __name__ == "__main__":
    main()
