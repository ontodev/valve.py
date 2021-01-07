import valve


def test_get_indexes():
    assert valve.get_indexes([0, 1, 0, 2, 3, 0], 0) == [0, 2, 5]


def test_has_ancestors():
    tree = {"foo": ["bar", "baz"], "baz": [], "bar": ["quax"], "quax": []}
    assert valve.has_ancestor(tree, "quax", "foo")
    assert valve.has_ancestor(tree, "baz", "foo", direct=True)
    assert valve.has_ancestor(tree, "foo", "foo")
    assert not valve.has_ancestor(tree, "quax", "foo", direct=True)
    assert not valve.has_ancestor(tree, "baz", "quax")
    assert not valve.has_ancestor(tree, "foo", "foo", direct=True)


def test_parsed_to_str():
    text = "tree(Label, external.Label, split=\", \")"
    assert valve.parsed_to_str(valve.parse(text)) == text


def test_idx_to_a1():
    assert valve.idx_to_a1(10, 4) == "D10"
    assert valve.idx_to_a1(10, 40) == "AN10"
    assert valve.idx_to_a1(10, 400) == "OJ10"
