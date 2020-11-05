import os
import shutil

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
    try:
        if not os.path.exists("build"):
            os.mkdir("build")
        if not os.path.exists("build/inputs"):
            os.mkdir("build/inputs")
        actual_output = f"build/{output_name}.tsv"
        expected_output = f"tests/resources/{output_name}.tsv"
        src_files = os.listdir("tests/resources/inputs")
        for file_name in src_files:
            full_file_name = os.path.join("tests/resources/inputs", file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, "build/inputs/")
        errors = valve.validate(["build/inputs"], distinct=distinct)
        valve.write_messages(actual_output, errors)
        diff = get_diff(actual_output, expected_output)
        if diff:
            print("The actual and expected outputs differ for " + output_name)
            print()
            for line in diff:
                print(line)
        assert not diff
    finally:
        if os.path.exists("build"):
            shutil.rmtree("build")


def test_valve():
    run_valve("errors", False)


def test_valve_distinct():
    run_valve("errors_distinct", True)
