import os

from valve import valve


def get_diff(actual, expected):
    actual_lines = []
    with open(actual) as f:
        for line in f:
            actual_lines.append(line.strip())
    expected_lines = []
    with open(expected) as f:
        for line in f:
            expected_lines.append(line.strip())
    removed = list(set(expected_lines) - set(actual_lines))
    added = list(set(actual_lines) - set(expected_lines))
    removed = [f"---\t{x}" for x in removed]
    added = [f"+++\t{x}" for x in added]
    return removed + added


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
