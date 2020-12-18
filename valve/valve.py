import copy
import csv
import inspect
import json
import jsonschema
import logging
import os
import re
import sys

from argparse import ArgumentParser
from collections import defaultdict
from .parse import parse


# Schema setup
json_path = os.path.join(os.path.dirname(__file__), "resources", "valve.json")
with open(json_path, encoding="UTF-8") as fp:
    main_schema = json.load(fp)

del main_schema["type"]
argument_schema = copy.deepcopy(main_schema)
argument_schema["items"] = {"$ref": "#/definitions/argument"}
datatype_schema = copy.deepcopy(main_schema)
datatype_schema["items"] = {"$ref": "#/definitions/datatype"}
field_schema = copy.deepcopy(main_schema)
field_schema["items"] = {"$ref": "#/definitions/field"}
rule_schema = copy.deepcopy(main_schema)
rule_schema["items"] = {"$ref": "#/definitions/rule"}


def validate(paths, add_functions=None, distinct_messages=None, row_start=2):
    """Run VALVE validation over a list of paths.

    :param paths: paths to files or directories to validate
    :param add_functions: dict of additional custom functions
    :param distinct_messages: path to directory for distinct tables
    :param row_start: row number that data to validate starts on
    :return: list of validation messages (empty on success)
    """
    # Register functions
    functions = copy.deepcopy(default_functions)
    if add_functions:
        if not isinstance(add_functions, dict):
            raise Exception("Value for add_functions must be a dict")
        for funct_name, details in add_functions.items():
            # Raise exception on issue
            check_custom_function(funct_name, details)
        functions.update(add_functions)

    # Check for directories and list their entire contents
    fixed_paths = []
    for p in paths:
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.endswith("tsv") or f.endswith("csv"):
                    fixed_paths.append(os.path.join(p, f))
        else:
            fixed_paths.append(p)

    # Load all tables, error on duplicates
    table_details = get_table_details(fixed_paths, row_start=row_start)
    config = {"functions": functions, "table_details": table_details, "row_start": row_start}

    # Load datatype, field, and rule - stop process on any problems
    setup_messages = configure_datatypes(config)
    setup_messages.extend(configure_fields(config))
    setup_messages.extend(configure_rules(config))
    kill_messages = [x for x in setup_messages if x["table"] in ["datatype", "field", "rule"]]
    if kill_messages:
        logging.error(f"VALVE configuration completed with {len(kill_messages)} errors")
        return kill_messages

    # Run validation
    messages = []
    for table in table_details.keys():
        if table in ["datatype", "field", "rule"]:
            continue
        logging.info("validating " + table)

        # Validate and return errors
        add_messages = [x for x in setup_messages if x["table"] == table]
        add_messages.extend(validate_table(config, table))
        logging.info(f"{len(add_messages)} problems found in table {table}")

        if add_messages and distinct_messages:
            # Update errors to only be distinct messages in a new table
            table_path = table_details[table]["path"]
            update_errors = collect_distinct_messages(
                table_details, distinct_messages, table_path, add_messages
            )
            messages.extend(update_errors)
        elif not distinct_messages:
            messages.extend(add_messages)

    if messages:
        logging.error(f"VALVE completed with {len(messages)} problems found!")
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
    rules = None
    if "table_rules" in config:
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
                messages = validate_condition(
                    config, parsed_type, table_name, field, row_idx, value
                )
                if messages:
                    field_id = fields[field]["field ID"]
                    for m in messages:
                        m.update({"rule ID": "field:" + str(field_id), "level": "ERROR"})
                        errors.append(m)

            # Check for rules
            if rules and field in rules:
                # Check if the value meets any of the conditions
                for rule in rules[field]:
                    when_condition = rule["when_condition"]
                    # Run meets_condition without logging
                    # as the then-cond check is only run if the value matches the type
                    messages = validate_condition(
                        config, when_condition, table_name, field, row_idx, value
                    )
                    if not messages:
                        # The "when" value meets the condition - validate the "then" value
                        then_column = rule["column"]

                        # Retrieve the "then" value to check if it meets the "then condition"
                        then_value = row[then_column]

                        messages = validate_condition(
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


# ---------- CONFIGURATION ----------


def build_condition(config, condition):
    """Build a parsed condition from a condition string.

    :param config: valve config dict
    :param condition: condition string
    :return: parsed condition
    """
    parsed, err = parse_condition(config, condition)
    if err:
        raise Exception(err)
    return parsed


def check_config_contents(config, table, conditions, rows):
    """Check the rows of a configuration table.

    :param config: valve config dict
    :param table: table to check (datatype, field, or rule)
    :param conditions: list of conditions (column to condition pairs)
    :param rows: table rows
    :return: list of messages (empty on success)
    """
    messages = []
    parsed_conditions = []
    for column, condition in conditions:
        parsed_conditions.append([column, build_condition(config, condition)])
    row_idx = config["row_start"]
    for row in rows:
        for column, condition in parsed_conditions:
            messages.extend(
                validate_condition(config, condition, table, column, row_idx, row[column])
            )
        row_idx += 1
    return messages


def configure_datatypes(config):
    """Add datatypes to config.

    :param config: valve config dict
    :return: list of messages (empty on success)
    """
    if "datatype" not in config["table_details"]:
        raise Exception(f"missing table 'datatype'")
    rows = config["table_details"]["datatype"]["rows"]

    # Check structure & contents of datatype table
    messages = check_rows(config, datatype_schema, "datatype", rows)
    config["datatypes"] = default_datatypes
    messages.extend(check_config_contents(config, "datatype", datatype_conditions, rows))

    for row in rows:
        datatype = row
        match = datatype.get("match")
        if match:
            pat = re.match(r"/(.+)/", match).group(1)
            datatype["match"] = re.compile(pat)
        config["datatypes"][datatype["datatype"]] = datatype

    return messages


def configure_fields(config):
    """Add fields to config.

    :param config: valve config dict
    :return: list of messages (empty on success)
    """
    if "field" not in config["table_details"]:
        raise Exception(f"missing table 'field'")
    rows = config["table_details"]["field"]["rows"]

    # Check structure & contents of field table
    messages = check_rows(config, field_schema, "field", rows)
    messages.extend(check_config_contents(config, "field", field_conditions, rows))

    config["trees"] = {}
    table_fields = {}
    row_idx = config["row_start"] - 1
    for row in rows:
        row_idx += 1
        table = row["table"]
        column = row["column"]
        field_types = table_fields.get(table, {})
        if table != "*":
            if table not in config["table_details"]:
                messages.append(
                    error(config, "field", "table", row_idx, f"unrecognized table '{table}'")
                )
                continue
            if column not in config["table_details"][table]["fields"]:
                messages.append(
                    error(
                        config,
                        "field",
                        "column",
                        row_idx,
                        f"unrecognized column '{column}' for table '{table}'",
                    )
                )
                continue

        # Check that this table.column pair has not already been defined
        if column in field_types:
            messages.append(
                error(
                    config,
                    "field",
                    "column",
                    row_idx,
                    f"Multiple condition defined for {table}.{column}",
                )
            )
            continue

        # Parse and validate the condition
        condition = row["condition"]
        parsed_condition, err = parse_condition(config, condition)
        if err:
            messages.append(error(config, "field", "condition", row_idx, err))
            continue

        if parsed_condition["type"] == "function" and parsed_condition["name"] == "tree":
            # Build a tree and add it to config
            tree_opts, err = get_tree_options(parsed_condition)
            if err:
                messages.append(error(config, "field", "condition", row_idx, err))
                continue
            tree, errs = build_tree(
                config,
                row_idx,
                table,
                column,
                tree_opts["child_column"],
                add_tree_name=tree_opts["add_tree_name"],
                split_char=tree_opts["split_char"],
            )
            messages.extend(errs)
            if tree:
                config["trees"][f"{table}.{column}"] = tree
        else:
            # Otherwise add to table_fields
            field_types[column] = {
                "parsed": parsed_condition,
                "field ID": row_idx,
            }
            table_fields[table] = field_types
    config["table_fields"] = table_fields
    return messages


def configure_rules(config):
    """Add rules to config.

    :param config: valve config dict
    :return: list of messages (empty on success)
    """
    if "rule" not in config["table_details"]:
        # Rule table is optional
        return []
    rows = config["table_details"]["rule"]["rows"]

    # Check structure & contents of field table
    messages = check_rows(config, rule_schema, "field", rows)
    messages.extend(check_config_contents(config, "rule", rule_conditions, rows))

    table_rules = {}
    row_idx = config["row_start"] - 1
    for row in rows:
        row_idx += 1
        table = row["table"]
        if table not in config["table_details"]:
            messages.append(
                error(config, "rule", "table", row_idx, f"unrecognized table '{table}'")
            )
            continue

        column_rules = table_rules.get(table, {})

        when_column = row["when column"]
        if when_column not in config["table_details"][table]["fields"]:
            messages.append(
                error(
                    config,
                    "rule",
                    "when column",
                    row_idx,
                    f"unrecognized column '{when_column}' for table '{table}'",
                )
            )
            continue

        rules = column_rules.get(when_column, [])

        then_column = row["then column"]
        if then_column not in config["table_details"][table]["fields"]:
            messages.append(
                error(
                    config,
                    "rule",
                    "then column",
                    row_idx,
                    f"unrecognized column '{then_column}' for table '{table}'",
                )
            )
            continue

        # Parse and validate the conditions
        when_condition = row["when condition"]
        parsed_when_condition, err = parse_condition(config, when_condition)
        if err:
            messages.append(error(config, "rule", "when condition", row_idx, err))
            continue
        then_condition = row["then condition"]
        parsed_then_condition, err = parse_condition(config, then_condition)
        if err:
            messages.append(error(config, "rule", "then condition", row_idx, err))
            continue

        # Add this condition to the dicts
        rules.append(
            {
                "when_condition": parsed_when_condition,
                "column": then_column,
                "then_condition": parsed_then_condition,
                "level": row.get("level", "ERROR"),
                "message": row.get("description", None),
                "rule ID": row_idx,
            }
        )
        column_rules[when_column] = rules
        table_rules[table] = column_rules

    config["table_rules"] = table_rules
    return messages


def get_table_details(paths, row_start=2):
    """Build a dictionary of table details.

    :param paths: list of table paths
    :param row_start: row number that contents to validate start on
    :return: dict of table name -> {fields, rows}
    """
    tables = {}
    for path in paths:
        sep = "\t"
        if path.endswith("csv"):
            sep = ","
        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter=sep)
            name = os.path.splitext(os.path.basename(path))[0]
            if name in tables:
                raise Exception(f"Duplicate table name '{name}'")
            tables[name] = {
                "path": path,
                "fields": reader.fieldnames,
                "rows": list(reader)[row_start - 2 :],
            }
    return tables


# ---------- CHECKS ----------


def check_custom_function(funct_name, details):
    """Check a custom function. Raise Exception on any problem.

    :param funct_name: name of function used in inputs
    :param details: dict containing usage, validate (function), check (function)
    """
    if "validate" not in details:
        raise Exception(f"Dict entry for '{funct_name}' requires a 'validate' key")
    function = details["validate"]
    if funct_name in default_functions:
        raise Exception(f"Cannot use builtin function name '{funct_name}'")
    params = list(inspect.signature(function).parameters.keys())
    if params[0] != "config":
        raise Exception(f"'{funct_name}' argument 1 must be config")
    if params[1] != "args":
        raise Exception(f"'{funct_name}' argument 2 must be args")
    if params[2] != "table":
        raise Exception(f"'{funct_name}' argument 3 must be table")
    if params[3] != "column":
        raise Exception(f"'{funct_name}' argument 4 must be column")
    if params[4] != "row_idx":
        raise Exception(f"'{funct_name}' argument 5 must be row_idx")
    if params[5] != "value":
        raise Exception(f"'{funct_name}' argument 6 must be value")


def check_function(config, parsed):
    """Check a function.

    :param config: valve config dict
    :param parsed: parsed function
    :return: string error message or None on success
    """
    condition = parsed_to_str(parsed)
    name = parsed["name"]
    if name not in config["functions"]:
        return f"unrecognized function '{name}'"
    function = config["functions"][name]

    try:
        jsonschema.validate(parsed["args"], argument_schema)
    except jsonschema.exceptions.ValidationError as err:
        message = err.message
        if err.parent and err.parent.validator == "oneOf":
            position = err.parent.path[0] + 1
            instance = err.parent.instance
            try:
                instance = parsed_to_str(instance)
            except ValueError:
                pass
            message = (
                f"argument {position} '{instance}' is not one of the allowed types for "
                f"'{function['usage']}' in '{condition}'"
            )
        return message
    for arg in parsed["args"]:
        if arg["type"] == "function":
            err = check_function(config, arg)
            if err:
                return err
        elif arg["type"] == "field":
            table = arg["table"]
            if table not in config["table_details"]:
                return f"unrecognized table '{table}'"
            column = arg["column"]
            if column not in config["table_details"][table]["fields"]:
                return f"unrecognized column '{column}' in table '{table}'"
    if "check" in function:
        c = function["check"]
        if isinstance(c, list):
            return check_args(name, parsed["args"], function["check"])
        elif callable(c):
            return c(config, parsed["args"])
        else:
            raise Exception(f"'check' value for {name} must be a list or function")


def check_args(name, args, expected):
    """Check a list of args for a function against a list of expected types.

    :param name: name of function to check
    :param args: function args
    :param expected: list of expected types
    :return: error messages or None on success
    """
    i = 0
    errors = []
    itr = iter(expected)
    e = next(itr)
    add_msg = ""
    while True:
        if e.endswith("*"):
            # zero or more
            e = e[:-1]
            for a in args[i:]:
                if not check_arg(a, e):
                    errors.append(f"optional argument {i + 1} must be of type '{e}'{add_msg}")
                i += 1
        elif e.endswith("?"):
            # zero or one
            e = e[:-1]
            if len(args) <= i:
                # this is OK here
                break
            if not check_arg(args[i], e):
                try:
                    add_msg = f" or '{e}'"
                    e = next(itr)
                    continue
                except StopIteration:
                    # no other expected args, add error
                    errors.append(f"optional argument {i + 1} must be of type '{e}'{add_msg}")
                    break
        elif e.endswith("+"):
            # one or more
            e = e[:-1]
            if len(args) <= i:
                errors.append(f"requires one or more '{e}'{add_msg} at argument {i + 1}")
                break
            for a in args[i:]:
                if not check_arg(a, e):
                    errors.append(f"argument {i + 1} must be of type '{e}'{add_msg}")
                i += 1
        else:
            # exactly one
            if len(args) <= i:
                errors.append(f"requires one '{e}'{add_msg} at argument {i + 1}")
                break
            if not check_arg(args[i], e):
                errors.append(f"argument {i + 1} must be of type '{e}'{add_msg}")
        try:
            i += 1
            e = next(itr)
        except StopIteration:
            break
    if i < len(args):
        errors.append(f"expects {i} argument(s), but {len(args)} were given")
    if errors:
        return name + " " + "; ".join(errors)


def check_arg(arg, expected):
    """Check that an arg is of expected type.

    :param arg: arg to check
    :param expected: expected type
    :return: True if expected type, False otherwise
    """
    if " or " in expected:
        # remove optional parentheses
        m = re.match(r"\((.+)\)", expected)
        if m:
            expected = m.group(1)
        valid = False
        for e in expected.split(" or "):
            if check_arg(arg, e):
                valid = True
                break
        return valid
    if expected.startswith("named:"):
        if arg["type"] != "named_arg":
            return False
        if arg["key"] != expected[6:]:
            return False
    elif expected in ["field", "string", "regex"]:
        if arg["type"] != expected:
            return False
    elif expected == "expression":
        if arg["type"] not in ["function", "string"]:
            return False
    else:
        raise Exception("Unknown argument type: " + expected)
    return True


def check_lookup(config, args):
    """Check the arguments passed to the lookup function.

    :param config: valve config dict
    :param args: list of parsed args
    :return: error message or None on success
    """
    errors = []
    i = 0
    table = None
    while i < 3 and i < len(args):
        a = args[i]
        i += 1
        if not check_arg(a, "string"):
            errors.append(f"argument {i} must be of type 'string'")
            continue
        if i == 1:
            table = a["value"]
            if a["value"] not in config["table_details"]:
                errors.append(f"argument 1 must be a table in inputs")
                return "lookup " + "; ".join(errors)
        if table and i > 1 and a["value"] not in config["table_details"][table]["fields"]:
            errors.append(f"argument {i} must be a column in '{table}'")
    if len(args) != 3:
        errors.append(f"expects 3 arguments, but {len(args)} were passed")
    if errors:
        return "lookup " + "; ".join(errors)


def check_rows(config, schema, table, rows):
    """Check the rows of a configuration file based on a JSON schema.

    :param config: valve config dict
    :param schema: JSON schema to validate structure
    :param table: table to validate
    :param rows: rows
    :return: list of messages (empty on success)
    """
    try:
        jsonschema.validate(rows, schema)
    except jsonschema.exceptions.ValidationError as err:
        if err.validator == "required":
            msg = err.message.replace("property", "column")
        else:
            msg = err.message
        if len(err.path) > 1:
            row_idx, column = list(err.path)[0:2]
            return [error(config, table, column, row_idx, msg)]
        return [{"table": table, "level": "ERROR", "message": msg}]
    return []


def find_datatype_ancestors(datatypes, datatype):
    """Recursively build a list of ancestor datatypes for a given datatype.

    :param datatypes: map of datatype name -> details
    :param datatype: datatype to get ancestors of
    :return: list of ancestor datatypes
    """
    ancestors = []
    parent = datatypes[datatype].get("parent")
    if parent:
        ancestors.append(parent)
        ancestors.extend(find_datatype_ancestors(datatypes, parent))
    return ancestors


def parse_condition(config, condition):
    """Check a condition and return the parsed condition.

    :param config: valve config dict
    :param condition: unparsed condition to check
    :return: parsed condition, error message
    """
    parsed = parse(condition)
    if parsed["type"] == "function":
        err = check_function(config, parsed)
        if err:
            return None, err
    elif parsed["type"] == "string":
        name = parsed["value"]
        if name not in config["datatypes"]:
            return None, f"unrecognized datatype '{name}'"
    else:
        return None, f"invalid condition '{condition}'"
    return parsed, None


# --------- CONDITION VALIDATION ----------


def validate_condition(config, condition, table, column, row_idx, value):
    """Run validation for a condition on a value.

    :param config: valve config dict
    :param condition: parsed condition
    :param table: table name
    :param column: column name
    :param row_idx: row number
    :param value: value to validate
    :return: list of messages (empty on success)
    """
    if condition["type"] == "function":
        name = condition["name"]
        args = condition["args"]
        function = config["functions"][name]
        return function["validate"](config, args, table, column, row_idx, value)
    elif condition["type"] == "string":
        return validate_datatype(config, condition, table, column, row_idx, value)
    else:
        raise Exception(f"invalid condition '{condition}'")


def validate_datatype(config, condition, table, column, row_idx, value):
    """Determine if the value is of datatype.

    :param config: valve config dict
    :param condition: parsed condition (datatype)
    :param table: table name
    :param column: column name
    :param row_idx: row number
    :param value: value to validate
    :return: list of messages (empty on success)
    """
    # First build a list of ancestors
    datatypes = config["datatypes"]
    name = condition["value"]
    ancestors = find_datatype_ancestors(datatypes, name)
    ancestors.insert(0, name)
    for name in ancestors:
        datatype = datatypes[name]
        description = datatype.get("description", name)
        level = datatype.get("level", "ERROR")
        if datatype.get("match"):
            if not datatype["match"].match(value):
                suggestion = None
                if datatype.get("replace"):
                    m = re.match(
                        r"s/(.+[^\\]|.*(?<!/)/.*[^\\])/(.+[^\\]|.*(?<!/)/.*[^\\])/(.*)",
                        datatype["replace"],
                    )
                    suggestion = re.sub(m.group(1), m.group(2), value)
                return [
                    error(
                        config,
                        table,
                        column,
                        row_idx,
                        description,
                        level=level,
                        suggestion=suggestion,
                    )
                ]
    return []


# ---------- VALVE FUNCTIONS ----------


def validate_any(config, args, table, column, row_idx, value):
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
        messages = validate_condition(config, arg, table, column, row_idx, value)
        if not messages:
            # As long as one is met, this passes
            return []
        conditions.append(parsed_to_str(arg))
    # If we get here, no condition was met
    message = f"'{value}' must meet one of: " + ", ".join(conditions)
    return [error(config, table, column, row_idx, message)]


def validate_concat(config, args, table, column, row_idx, value):
    """Method for the VALVE 'concat' function.

    :param config: valve config dictionary
    :param args: arguments provided to distinct
    :param table: table to run distinct on
    :param column: column to run distinct on
    :param row_idx: current row number
    :param value: value to run concat on
    :return: List of messages (empty on success)
    """
    datatypes = config["datatypes"]
    validate_conditions = []
    validate_values = []
    rem = value
    for arg in args:
        if arg["type"] == "string":
            arg_val = arg["value"]
            if arg_val in datatypes:
                validate_conditions.append(arg)
                continue
            validate_values.append(rem.split(arg_val, 1)[0])
            try:
                rem = rem.split(arg_val, 1)[1]
            except IndexError:
                # Does not match pattern given in concat
                return [
                    error(
                        config,
                        table,
                        column,
                        row_idx,
                        f"'{value}' must contain substring '{arg_val}'",
                    )
                ]
        else:
            validate_conditions.append(arg)
    if rem:
        validate_values.append(rem)
    idx = 0
    messages = []
    while idx < len(validate_values):
        v = validate_values[idx]
        condition = validate_conditions[idx]
        messages.extend(validate_condition(config, condition, table, column, row_idx, v))
        idx += 1
    return messages


def validate_distinct(config, args, table, column, row_idx, value):
    """Method for the VALVE 'distinct' function.

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


def validate_in(config, args, table, column, row_idx, value):
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


def validate_list(config, args, table, column, row_idx, value):
    """Method for the VALVE 'list' function.

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
    errs = []
    for v in value.split(split_char):
        errs.extend(validate_condition(config, expr, table, column, row_idx, v))
    if errs:
        messages = []
        for err in errs:
            messages.append(err["message"])
        message = "; ".join(messages)
        return [error(config, table, column, row_idx, message)]
    return []


def validate_lookup(config, args, table, column, row_idx, value):
    """Method for VALVE 'lookup' function.

    :param config: valve config dictionary
    :param args: arguments provided to lookup
    :param table: table name
    :param column: column name
    :param row_idx: row number from values
    :param value: value to run lookup on
    :return: List of messages (empty on success)
    """
    table_details = config["table_details"]
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


def validate_not(config, args, table, column, row_idx, value):
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
        messages = validate_condition(config, arg, table, column, row_idx, value)
        if not messages:
            # If any condition *is* met (no errors), this fails
            unparsed = parsed_to_str(arg)
            message = f"'{value}' must not be '{unparsed}'"
            if unparsed == "blank":
                message = f"value must not be blank"
            return [error(config, table, column, row_idx, message)]
    return []


def validate_sub(config, args, table, column, row_idx, value):
    """Method for the VALVE 'sub' function.

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
    return validate_condition(config, subfunc, table, column, row_idx, value)


def validate_under(config, args, table, column, row_idx, value):
    """Method for VALVE 'under' function.

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


# ---- TREE BUILDING ----


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
    if not isinstance(child_column, dict) or child_column["type"] != "string":
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


# ---------- OUTPUTS ----------


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


# ---------- HELPERS ----------


def error(config, table, column, row_idx, message, level="ERROR", suggestion=None):
    """Build an error message dict.

    :param config: valve config dict
    :param table: table name
    :param column: column name
    :param row_idx: row number
    :param message: error message
    :param level: error level (error, warn, info)
    :param suggestion: suggested replacement
    :return:
    """
    row_start = config["row_start"]
    col_idx = config["table_details"][table]["fields"].index(column)
    if table in ["datatype", "field", "rule"]:
        row_num = row_idx
    else:
        row_num = row_idx + row_start
    d = {
        "table": table,
        "cell": idx_to_a1(row_num, col_idx + 1),
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
    raise ValueError("Unknown condition type: " + cond_type)


# ---------- GLOBALS ----------

# Required headers for 'datatype' table
datatype_headers = [
    "datatype",
    "parent",
    "match",
    "level",
]

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

# Builtin datatypes
default_datatypes = {
    "blank": {
        "datatype": "blank",
        "parent": "",
        "match": re.compile(r"^$"),
        "level": "ERROR",
        "description": "an empty string",
    },
    "datatype_label": {
        "datatype": "datatype_label",
        "parent": "",
        "match": re.compile(r"[A-Za-z][A-Za-z0-9_-]+"),
        "level": "ERROR",
        "description": "a word that starts with a letter and may contain dashes and underscores",
    },
    "regex": {
        "datatype": "regex",
        "parent": "",
        "match": re.compile(r"^/.+/$"),
        "level": "ERROR",
        "description": "A regex match",
    },
    "regex_sub": {
        "datatype": "regex_sub",
        "parent": "",
        "match": re.compile(r"^s/.+[^\\]|.*(?<!/)/.*[^\\]/.+[^\\]|.*(?<!/)/.*[^\\]/.*$"),
        "level": "ERROR",
        "description": "A regex substitution",
    },
}

# Builtin functions
default_functions = {
    "any": {
        "usage": "any(expression+)",
        "check": ["expression+"],
        "validate": validate_any,
    },
    "concat": {
        "usage": "concat(value+)",
        "check": ["(expression or string)+"],
        "validate": validate_concat,
    },
    "distinct": {
        "usage": "distinct(expression)",
        "check": ["expression", "field*"],
        "validate": validate_distinct,
    },
    "in": {"usage": "in(value+)", "check": ["(string or field)+"], "validate": validate_in},
    "list": {
        "usage": "list(str, expression)",
        "check": ["string", "expression"],
        "validate": validate_list,
    },
    "lookup": {
        "usage": "lookup(table, column, column)",
        "check": check_lookup,
        "validate": validate_lookup,
    },
    "not": {
        "usage": "not(expression)",
        "check": ["expression"],
        "validate": validate_not,
    },
    "sub": {
        "usage": "sub(regex, expression)",
        "check": ["regex", "expression"],
        "validate": validate_sub,
    },
    "tree": {"usage": "tree()", "check": ["string", "field?", "named:split?"], "validate": None},
    "under": {
        "usage": "under(table.column, str, [direct=bool])",
        "check": ["field", "string", "named:direct?"],
        "validate": validate_under,
    },
}

datatype_conditions = [
    ["datatype", "datatype_label"],
    ["parent", "any(blank, in(datatype.datatype))"],
    ["match", "any(blank, regex)"],
    ["level", 'any(blank, in("ERROR", "error", "WARN", "warn", "INFO", "info"))'],
    ["replace", "any(blank, regex_sub)"],
]

field_conditions = [["table", "not(blank)"], ["column", "not(blank)"], ["condition", "not(blank)"]]

rule_conditions = [
    ["table", "not(blank)"],
    ["when column", "not(blank)"],
    ["when condition", "not(blank)"],
    ["then column", "not(blank)"],
    ["then condition", "not(blank)"],
    ["level", 'any(blank, in("ERROR", "error", "WARN", "warn", "INFO", "info"))'],
]


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
