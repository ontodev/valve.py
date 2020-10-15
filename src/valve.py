import csv
import logging
import os
import re
import sys

from argparse import ArgumentParser
from lark import Lark, Tree, Token
from lark.exceptions import UnexpectedInput


# TODO - rule names in the output should be table name + row
# TODO - handle numeric datatypes (later)


# Required headers for 'datatype' table
datatype_headers = [
    "datatype",
    "parent",
    "match",
    "level",
    "description",
    "instructions",
    "replace",
]

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
    "level",
    "description",
    "note",
]

# Supported function names
funct_names = ["CURIE", "from", "in", "lookup", "tree", "under"]

errors = []
trees = {}
kill = False


# ---- MISC HELPERS ----


def build_datatype_ancestors(datatype, datatypes, ancestors):
    """"""
    parent = datatypes[datatype].get("parent")
    if parent:
        ancestors.append(parent)
        build_datatype_ancestors(parent, datatypes, ancestors)
    ancestors.reverse()


def build_table_descendants(tree, node, descendants):
    """"""
    children = tree.get(node)
    if not children:
        return
    for c in children:
        if c not in descendants:
            descendants.append(c)
            build_table_descendants(tree, c, descendants)


def idx_to_a1(row, col):
    """Convert a row & column to A1 notation. Adapted from gspread.utils."""
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


def get_table_details(tables):
    """

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
    """

    :param datatype_table: path to datatype table
    :return: dict of datatype -> {parent, match, level, description, instructions, replace}
    """
    global errors, kill
    sep = "\t"
    if datatype_table.endswith("csv"):
        sep = ","

    # Read the datatypes from the sheet
    datatypes = {}
    with open(datatype_table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames
        if headers != datatype_headers:
            raise Exception(
                f"Headers for 'datatype' at {datatype_table} do not match required fields"
            )
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
            kill = True
            errors.append(
                {
                    "table": datatype_table,
                    "cell": idx_to_a1(idx, headers.index("parent") + 1),
                    "rule": "unknown parent datatype",
                    "message": "the parent datatype must be defined in the 'datatype' sheet",
                }
            )

        level = details["level"].lower().strip()
        validate_level(level, datatype_table, idx_to_a1(idx, headers.index("level") + 1))
    return datatypes


def read_field_table(field_table, table_details, datatypes):
    """:return: {table-name: {field-name: type, ...}, ...}"""
    global errors, kill
    sep = "\t"
    if field_table.endswith("csv"):
        sep = ","

    # Dict of table name -> field types in that table
    table_fields = {}
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
                kill = True
                errors.append(
                    {
                        "table": field_table,
                        "cell": idx_to_a1(idx, headers.index("column") + 1),
                        "rule": "duplicate column",
                        "message": "this column value is already defined in 'field'",
                    }
                )
            else:
                # Parse the field type
                parsed_type = validate_condition(
                    row["type"], field_table, idx_to_a1(idx, headers.index("type") + 1), datatypes
                )
                if not parsed_type:
                    # Type could not be parsed - grammar issue
                    continue
                if "function" in parsed_type and "tree" in parsed_type["function"]:
                    # Special processing for `tree` function
                    # This does not get added to field_types,
                    # but a tree is built and added to global trees
                    validate_tree_type(
                        field_table,
                        idx,
                        headers.index("type") + 1,
                        parsed_type,
                        table,
                        column,
                        table_details,
                    )
                    continue
                field_types[column] = {
                    "parsed": parsed_type,
                    "unparsed": row["type"],
                    "field ID": idx,
                }
                table_fields[table] = field_types
    return table_fields


def read_rule_table(rule_table, table_columns, datatypes):
    """
    :param rule_table: path to rule table
    :param table_columns: all table names -> columns
    :param datatypes:
    :return: {table-name:
                {column-name:
                    {condition:
                        {"table": then-table,
                         "column": then-column,
                         "condition": then-condition,
                         "level": level,
                         "message": message},
                     ...},
                 ...},
              ...}
    """
    global errors, kill
    sep = "\t"
    if rule_table.endswith("csv"):
        sep = ","

    table_rules = {}
    with open(rule_table, "r") as f:
        reader = csv.DictReader(f, delimiter=sep)
        headers = reader.fieldnames
        if headers != rule_headers:
            raise Exception(f"Headers for 'field' at {rule_table} do not match required fields")

        idx = 1
        for row in reader:
            idx += 1
            # Validate the when table.column (check that these exist)
            when_table = row["when table"]
            when_table_loc = idx_to_a1(idx, headers.index("when table") + 1)
            when_column = row["when column"]
            when_column_loc = idx_to_a1(idx, headers.index("when column") + 1)
            validate_table_column(
                when_table, when_column, table_columns, when_table_loc, when_column_loc
            )

            # Validate the when condition
            when_condition = row["when condition"]
            parsed_when_condition = validate_condition(
                when_condition,
                rule_table,
                idx_to_a1(idx, headers.index("when condition") + 1),
                datatypes,
            )
            if not parsed_when_condition:
                # when-cond could not be parsed
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
            parsed_then_condition = validate_condition(
                then_condition,
                rule_table,
                idx_to_a1(idx, headers.index("then condition") + 1),
                datatypes,
            )
            if not parsed_then_condition:
                # then-cond could not be parsed
                continue

            # Validate the message level
            level = row["level"].lower().strip()
            validate_level(level, rule_table, idx_to_a1(idx, headers.index("level") + 1))

            # Validate the when table.column (check that these exist)
            then_table = row["then table"]
            then_table_loc = idx_to_a1(idx, headers.index("then table") + 1)
            then_column = row["then column"]
            then_column_loc = idx_to_a1(idx, headers.index("then column") + 1)
            validate_table_column(
                then_table, then_column, table_columns, then_table_loc, then_column_loc
            )

            # Add this condition to the dicts
            rules.append(
                {
                    "when condition": parsed_when_condition,
                    "table": then_table,
                    "column": then_column,
                    "then condition": parsed_then_condition,
                    "level": level,
                    "message": row["description"],
                    "rule ID": idx,
                }
            )
            column_rules[when_column] = rules
            table_rules[when_table] = column_rules

    return table_rules


# ---- INPUT VALIDATION ----


def build_tree(table, rows, row_idx, col_idx, parent_column, child_column):
    """Build a hierarchy for the `tree` function while validating the values.

    :param table:
    :param rows:
    :param col_idx:
    :param parent_column:
    :param child_column:
    :return:
    """
    global errors, trees
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
                    "rule": "field " + str(row_idx),
                    "level": "error",
                    "message": f"the value of {parent_column} must be defined in {child_column}",
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
    trees[f"{table}.{parent_column}"] = tree


def validate_function(parsed_condition, table_name, loc):
    """Validate a function.

    :param parsed_condition:
    :param table_name:
    :param loc:
    :return:
    """
    global errors, kill
    funct_name = list(parsed_condition["function"].keys())[0]
    if funct_name not in funct_names:
        kill = True
        errors.append(
            {
                "table": table_name,
                "cell": loc,
                "rule": "unknown function",
                "level": "error",
                "message": f"function name ({funct_name}) must be one of: "
                           + ",".join(funct_names),
            }
        )
        return None
    # Special validation for each function
    # TODO - validation for CURIE, from, in, under ...
    args = parsed_condition["function"][funct_name]
    if funct_name == "lookup":
        # Validate that the arguments are table-columns and the tables are the same
        if "table" not in args[0] or "table" not in args[1]:
            kill = True
            errors.append(
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": "invalid `lookup` argument",
                    "level": "error",
                    "message": "the arguments for `lookup` must be 'table.column' format",
                }
            )
            return None
        table_name = args[0]["table"]
        table_name_2 = args[1]["table"]
        if table_name != table_name_2:
            kill = True
            errors.append(
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": "invalid `lookup` argument",
                    "level": "error",
                    "message": f"the first table name ({table_name}) must be the same as "
                               f"the second table name ({table_name_2})",
                }
            )
            return None
    return parsed_condition


def validate_condition(condition, table_name, loc, datatypes, parse_cond=True):
    """

    :param condition:
    :param table_name: name of table that condition is defined in
    :param loc: A1 location of condition
    :param datatypes:
    :param parse_cond:
    :return: None on error, parsed condition on success
    """
    global errors, kill
    if parse_cond:
        parsed_condition, err = parse(condition)
    else:
        parsed_condition, err = condition, None

    if not parsed_condition:
        # Condition could not be parsed - unexpected input
        msg = f"unable to parse '{condition}' due to unexpected input"
        if err:
            msg = f"unable to parse due to unexpected input:\n{err}"
        kill = True
        errors.append(
            {
                "table": table_name,
                "cell": loc,
                "rule": "invalid condition",
                "level": "error",
                "message": msg,
            }
        )
        return None

    # Validate specifics
    if "datatype" in parsed_condition:
        dt = parsed_condition["datatype"]
        if dt not in datatypes.keys():
            kill = True
            errors.append(
                {
                    "table": table_name,
                    "cell": loc,
                    "rule": "unknown datatype",
                    "level": "error",
                    "message": f"datatype '{dt}' is not defined in the datatype table",
                }
            )
            return None
    elif "disjunction" in parsed_condition:
        valid = True
        # Parse each sub-condition and check if they are valid
        for sub_cond in parsed_condition["disjunction"]:
            parsed_sub = validate_condition(sub_cond, table_name, loc, datatypes, parse_cond=False)
            if not parsed_sub:
                valid = False
        if not valid:
            return None
    elif "function" in parsed_condition:
        return validate_function(parsed_condition, table_name, loc)
    elif "negation" in parsed_condition:
        valid = True
        for sub_cond in parsed_condition["negation"]:
            parsed_sub = validate_condition(sub_cond, table_name, loc, datatypes, parse_cond=False)
            if not parsed_sub:
                valid = False
        if not valid:
            return None
    else:
        # Something is wrong with the Lark grammar
        raise Exception("Unknown expression: " + str(parsed_condition))
    return parsed_condition


def validate_level(level, table_name, loc):
    """

    :param level: logging level
    :param table_name: name of table that level is defined in
    :param loc: A1 location of level
    :return:
    """
    global errors, kill
    if level == "":
        kill = True
        errors.append(
            {
                "table": table_name,
                "cell": loc,
                "rule": "missing level",
                "message": "the 'level' must be one of: ERROR, WARN, INFO",
            }
        )
        return False
    elif level.lower() not in ["error", "warn", "info"]:
        kill = True
        errors.append(
            {
                "table": table_name,
                "cell": loc,
                "rule": "unknown level",
                "message": "the 'level' must be one of: ERROR, WARN, INFO",
            }
        )
        return False
    return True


def validate_table_column(table, column, table_fields, table_loc, column_loc):
    """Validate a table-column pair. Ensure that the table exists in the input and that the column
    exists in the table.

    :param table: given table name
    :param column: given column name
    :param table_fields: headers of given table
    :param table_loc: A1 location of table name
    :param column_loc: A1 location of column name
    :return: True if valid
    """
    valid = True
    if table not in table_fields.keys():
        errors.append(
            {
                "table": "rule",
                "cell": table_loc,
                "rule": "unknown table",
                "message": "the table must exist in the input",
            }
        )
        valid = False
    else:
        if column not in table_fields[table]:
            errors.append(
                {
                    "table": "rule",
                    "cell": column_loc,
                    "rule": "unknown column",
                    "message": f"the provided column must exist in '{table}'",
                }
            )
            valid = False
    return valid


def validate_tree_type(field_table, row_idx, col_idx, parsed_type, table, column, table_details):
    """

    :param field_table:
    :param loc:
    :param parsed_type:
    :param table:
    :param column:
    :param table_details:
    :return:
    """
    global errors, kill
    args = parsed_type["function"]["tree"]
    if len(args) != 1:
        # tree(...) must have exactly one argument
        # logging.error("The `tree` function accepts exactly one argument")
        kill = True
        errors.append(
            {
                "table": field_table,
                "cell": idx_to_a1(row_idx, col_idx),
                "rule": "tree function error",
                "level": "error",
                "message": f"the `tree` function must have exactly one argument",
            }
        )
        return False
    tree_table = args[0]["table"]
    if tree_table != table:
        # logging.error("The table in `tree` must be the same as the `table` value")
        kill = True
        errors.append(
            {
                "table": field_table,
                "cell": idx_to_a1(row_idx, col_idx),
                "rule": "tree function error",
                "level": "error",
                "message": f"the table name provided in the `tree` function ({tree_table}) "
                f"must be the same as the value in the 'table' column ({table})",
            }
        )
        return False
    else:
        child_column = args[0]["column"]
        build_tree(
            table,
            table_details[table]["rows"],
            row_idx,
            table_details[table]["fields"].index(column),
            column,
            child_column,
        )
        return True


# ---- LARK PARSING ----


def parse_args(arg):
    """Parse an argument of a 'function' expression.

    :param arg: list representing one argument provided to a function.
                This list is created from the Lark parser output.
    :return: str of dict representation of arg, or None on error
    """
    found_label = False
    search_label = False
    search_field = False
    table = None
    itm = None

    for itm in arg:
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
    """

    :param function: list representing the function.
                     This list is created from the Lark parser output.
    :return: dict of function name -> list of arguments (None on error),
             error message (None on success)
    """
    search_name = False
    search_args = False
    args = []
    cur_arg = None
    funct_name = None

    for itm in function:
        if itm == "function_name":
            # Next element will be the name
            search_name = True
            continue
        if search_name:
            # Save function name
            funct_name = itm
            search_name = False
            continue

        if itm == "arguments":
            # Following items will be arguments
            search_args = True
            continue

        if search_args:
            if itm == "argument":
                # Beginning of an argument
                if cur_arg is not None:
                    # Parse previous argument if tracked
                    parsed = parse_args(cur_arg)
                    if parsed:
                        args.append(parsed)
                    else:
                        # Arguments could not be parsed
                        return None, f"Unable to parse arguments: {cur_arg}"
                # Reset current arg
                cur_arg = []
                continue
            if cur_arg is not None:
                cur_arg.append(itm)

    if cur_arg is not None:
        # Make sure to parse the last entry in the list
        parsed = parse_args(cur_arg)
        if parsed:
            args.append(parsed)
        else:
            # Arguments could not be parsed
            return None, f"Unable to parse arguments: {cur_arg}"

    return {funct_name: args}, None


def parse_sub_conditions(sub_conds):
    """

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
                    parsed_set.append({"negation": {"function": parsed_function}})
                    in_negation = False
                else:
                    parsed_set.append({"function": parsed_function})
            expr_type = None
            search_expr = True
            function = None
            continue

        if search_expr:
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
        if search_type:
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
                if in_negation:
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
            parsed_set.append({"negation": {"function": parsed_function}})
        else:
            parsed_set.append({"function": parsed_function})
    return parsed_set, None


def lark_to_list(v, parsed_text):
    """Parse Lark output into a list."""
    if isinstance(v, list):
        for itm in v:
            lark_to_list(itm, parsed_text)
    elif isinstance(v, Tree):
        parsed_text.append(v.data)
        lark_to_list(v.children, parsed_text)
    elif isinstance(v, Token):
        if v.type == "ESCAPED_STRING":
            # Remove quotes and escape characters
            val = v.value[1:-1]
            r = re.compile(r"\\(.)")
            r.sub(r"\1", val)
            parsed_text.append(val)
        parsed_text.append(v.value)


def parse(text):
    """

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
        argument: field | label
        field: label "." label
        datatype: label
        label: WORD | ESCAPED_STRING

        %import common.WORD
        %import common.ESCAPED_STRING
        %ignore " "           // Disregard spaces in text
    """
    )
    parsed_text = []
    try:
        t = parser.parse(text)
    except UnexpectedInput as e:
        return None, e.get_context(text)

    # Transform Lark parser output (Tree) to list
    lark_to_list(t.children, parsed_text)
    # Output always begins with "expression"
    parsed_text.pop(0)

    # Get the first value and transform the list into a dict
    v = parsed_text.pop(0)
    if v == "type":
        expr_type = parsed_text.pop(0)
        if expr_type == "datatype":
            return {"datatype": parsed_text[1:][0]}, None
        else:
            p_funct, err = parse_function(parsed_text)
            if not p_funct:
                return None, err
            return {"function": p_funct}, None
    elif v == "disjunction":
        p_set, err = parse_sub_conditions(parsed_text)
        if not p_set:
            return None, err
        return {"disjunction": p_set}, None
    elif v == "negation":
        p_set, err = parse_sub_conditions(parsed_text)
        if not p_set:
            return None, err
        return {"negation": p_set}, None


# ---- CONDITION VALIDATION ----


def is_datatype(value, datatype, datatypes):
    """Determine if the value is of datatype.

    :param value:
    :param datatype:
    :param datatypes:
    :return:
    """
    # First build a list of ancestors
    ancestor_dts = [datatype]
    build_datatype_ancestors(datatype, datatypes, ancestor_dts)
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
    value,
    condition,
    datatypes,
    table_details,
    when_value=None,
    log=False,
    table=None,
    loc=None,
    level=None,
    field=None,
):
    """Determine if the value meets the condition.

    :param value:
    :param condition: dict of parsed condition
    :param datatypes:
    :param table_details:
    :param when_value:
    :param log:
    :param table:
    :param loc:
    :param level:
    :param field:
    :return:
    """
    condition_type = list(condition.keys())[0]

    if condition_type == "datatype":
        datatype = condition["datatype"]
        # Check if condition is met, potentially get a replacement
        value_meets_condition, replace = is_datatype(value, datatype, datatypes)
        if log and value_meets_condition is False:
            unparsed = field["unparsed"]
            field_id = field["field ID"]
            errors.append(
                {
                    "table": table,
                    "cell": loc,
                    "rule": "field " + str(field_id),
                    "level": level,
                    "message": f"this value is not of datatype '{unparsed}'",
                    "suggestion": replace,
                }
            )
        return value_meets_condition

    elif condition_type == "function":
        if not run_function(value, condition["function"], table_details, lookup_value=when_value):
            if log:
                unparsed = field["unparsed"]
                field_id = field["field ID"]
                errors.append(
                    {
                        "table": table,
                        "cell": loc,
                        "rule": "field " + str(field_id),
                        "level": level,
                        "message": f"this value does not pass function '{unparsed}'",
                    }
                )
            return False
        return True

    elif condition_type == "negation":
        # Negations may be one or more
        for c in condition["negation"]:
            if not meets_condition(value, c, datatypes, table_details):
                # As long as one is "NOT" met, this passes
                return True
        # If we get here, the negation conditions were not met
        if log:
            unparsed = field["unparsed"]
            field_id = field["field ID"]
            errors.append(
                {
                    "table": table,
                    "cell": loc,
                    "rule": "field " + str(field_id),
                    "level": level,
                    "message": f"this value does not meet any of the criteria: '{unparsed}'",
                }
            )
        return False

    elif condition_type == "disjunction":
        for c in condition["disjunction"]:
            if meets_condition(value, c, datatypes, table_details):
                return True
        if log:
            unparsed = field["unparsed"]
            field_id = field["field ID"]
            errors.append(
                {
                    "table": table,
                    "cell": loc,
                    "rule": "field " + str(field_id),
                    "level": level,
                    "message": f"this value does not meet any of the criteria: '{unparsed}'",
                }
            )
        return False

    else:
        # This should be prevented in validate_condition
        raise Exception("unknown condition type: " + condition_type)


def run_function(value, function, table_details, lookup_value=None):
    """Run a function for the provided value.

    :param value:
    :param function:
    :param table_details:
    :param lookup_value: required for lookup function
    :return:
    """
    global errors, trees
    funct_name = list(function.keys())[0]
    args = function[funct_name]
    if funct_name == "CURIE":
        # A CURIE with a prefix in the given table-column
        # Only one arg (table-column w/prefixes)
        table_name = args[0]["table"]
        column_name = args[0]["column"]
        prefixes = []
        for row in table_details[table_name]["rows"]:
            prefixes.append(row[column_name])
        value_prefix = value.split(":")[0]
        if value_prefix not in prefixes:
            return False
        return True
    elif funct_name == "in":
        # One of the values in the args
        valid = False
        for allowed_value in args:
            if value == allowed_value:
                valid = True
                break
        if not valid:
            return False
        return True
    elif funct_name == "from":
        # TODO - validate one argument, table & column exist
        table_name = args[0]["table"]
        column_name = args[0]["column"]
        allowed_values = table_details[table_name]
        pass
    elif funct_name == "lookup":
        if not lookup_value:
            raise Exception("A lookup_value is required for a lookup function")
        # table-column, allowed value
        # given value is the same as table-column, look for the allowed value
        table_name = args[0]["table"]
        column_name = args[0]["column"]
        column_name_2 = args[1]["column"]
        for row in table_details[table_name]["rows"]:
            maybe_value = row[column_name]
            if maybe_value == lookup_value:
                check_value = row[column_name_2]
                if value != check_value:
                    return False
        return True
    elif funct_name == "under":
        # Same datatype as given table-column
        # Equal to given table-column values or their descendants
        # table-column tree value = arg[0], parent to look under = arg[1]
        # Basically calls tree(table-column) and looks under parent in that tree
        # logging.warning("The 'under' function is not yet implemented")
        table_name = args[0]["table"]
        column_name = args[0]["column"]
        tree_name = f"{table_name}.{column_name}"
        if tree_name not in trees:
            # TODO - this should already be validated
            logging.error(f"A tree for {tree_name} is not defined")
            return False
        tree = trees[tree_name]
        top_level = args[1]
        descendants = [top_level]
        build_table_descendants(tree, top_level, descendants)
        if value not in descendants:
            return False
        return True
    else:
        # This should never be reached in normal operation
        # validate_condition already checks that this is OK
        raise Exception("Unknown function: " + funct_name)


# ---- MAIN METHODS ----


def run_validation(table, table_details, datatypes, fields, rules):
    """
    :param table:
    :param table_details:
    :param datatypes:
    :param fields: {field-name: type, ...}
    :param rules:
    """
    global errors, trees
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

                    # Run meets_condition with logging
                    # as all values in this field must match the type
                    meets_condition(
                        value,
                        parsed_type,
                        datatypes,
                        table_details,
                        log=True,
                        table=table,
                        loc=idx_to_a1(row_idx + 1, col_idx),
                        field=fields[field],
                    )
                # Check for rules
                if field in rules:
                    # Check if the value meets any of the conditions
                    for rule in rules[field]:
                        parsed_condition = rule["when condition"]
                        # Run meets_condition without logging
                        # as the then-cond check is only run if the value matches the type
                        if meets_condition(value, parsed_condition, datatypes, table_details):
                            # The "when" value meets the condition - validate the "then" value
                            table = rule["table"]
                            column = rule["column"]

                            # Retrieve the "then" value to check if it meets the "then condition"
                            check_value = table_details[table]["rows"][row_idx][column]
                            check_col_idx = table_details[table]["fields"].index(column)
                            if not meets_condition(
                                check_value,
                                rule["then condition"],
                                datatypes,
                                table_details,
                                when_value=value,
                            ):
                                errors.append(
                                    {
                                        "table": table,
                                        "cell": idx_to_a1(row_idx + 2, check_col_idx + 1),
                                        "rule": "rule " + str(rule["rule ID"]),
                                        "level": rule["level"],
                                        "message": rule["message"],
                                    }
                                )
                col_idx += 1
            row_idx += 1


def write_errors(output):
    global errors
    sep = "\t"
    if output.endswith("csv"):
        sep = ","
    with open(output, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["table", "cell", "rule", "level", "message", "suggestion"],
            delimiter=sep,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(errors)


def main():
    p = ArgumentParser()
    p.add_argument("-d", "--datatype")
    p.add_argument("-f", "--field")
    p.add_argument("-r", "--rule")
    p.add_argument("-t", "--tables", nargs="+")
    p.add_argument("-o", "--output")
    args = p.parse_args()

    table_details = get_table_details(args.tables)
    datatypes = read_datatype_table(args.datatype)
    table_fields = read_field_table(args.field, table_details, datatypes)
    table_rules = read_rule_table(
        args.rule,
        {table_name: details["fields"] for table_name, details in table_details.items()},
        datatypes,
    )

    if kill:
        # If there are any errors in configuration, quit now with exit code 1
        write_errors(args.output)
        logging.critical("VALVE configuration validation completed with errors!")
        sys.exit(1)

    for table in args.tables:
        tname = os.path.splitext(os.path.basename(table))[0]
        fields = table_fields.get(tname, [])
        rules = table_rules.get(tname, [])
        run_validation(table, table_details, datatypes, fields, rules)

    write_errors(args.output)
    if errors:
        logging.critical(f"VALVE completed with {len(errors)} problems found!")


if __name__ == "__main__":
    main()
