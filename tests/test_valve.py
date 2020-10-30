import difflib
import os

from valve import valve


def get_diff(actual, expected):
    with open(actual) as f2:
        actual_text = f2.readlines()
    with open(expected) as f1:
        expected_text = f1.readlines()
    return list(difflib.unified_diff(actual_text, expected_text))[2:]


def run_valve(output_name, distinct):
    if not os.path.exists("build"):
        os.mkdir("build")
    actual_output = f"build/{output_name}.tsv"
    expected_output = f"tests/resources/{output_name}.tsv"
    valve.valve("tests/resources", actual_output, distinct=distinct)
    diff = get_diff(actual_output, expected_output)
    if diff:
        print("The actual and expected outputs differ")
        print()
        for line in diff:
            print(line)
    assert not diff


def test_valve():
    run_valve("errors", False)


def test_valve_distinct():
    run_valve("errors_distinct", True)

