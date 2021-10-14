from lark import Lark, Transformer

grammar = """!n_dqstring: xrule_0 xrule_1 xrule_0-> alias_0
!n_sqstring: xrule_2 xrule_3 xrule_2-> alias_1
!n_btstring: xrule_4 xrule_5 xrule_4-> alias_2
!n_dstrchar: /[^\\"\n]/x-> alias_3
    |xrule_6 n_strescape-> alias_4
!n_sstrchar: /[^\\'\n]/x-> alias_5
    |xrule_6 n_strescape-> alias_6
    |xrule_7-> alias_7
!n_strescape: /["\\/bfnrt]/-> alias_8
    |xrule_8 /[a-fA-F0-9]/ /[a-fA-F0-9]/ /[a-fA-F0-9]/ /[a-fA-F0-9]/-> alias_9
!n_start: n__ws_maybe n_expression n__ws_maybe-> alias_10
!n_expression: n_string-> alias_11
    |n_function-> alias_12
!n_function: n_function_name xrule_9 n_arguments xrule_10-> alias_13
!n_function_name: n_alphanum-> alias_14
!n_arguments: n__ws_maybe n_argument xrule_12 n__ws_maybe-> alias_15
!n_argument: n_string
    |n_field
    |n_function
    |n_named_arg
    |n_regex
!n_field: n_label xrule_13 n_label-> alias_16
!n_named_arg: n_label xrule_14 n_label-> alias_17
!n_string: n_label-> alias_18
!n_label: n_alphanum-> alias_19
    |n_dqstring-> alias_20
!n_alphanum: xrule_15-> alias_21
!n__ws_maybe: xrule_16-> alias_22
!n_regex: n_regex_sub
    |n_regex_match
!n_regex_match: xrule_17 n_regex_pattern xrule_17 n_regex_flag-> alias_23
!n_regex_sub: xrule_18 n_regex_pattern xrule_17 n_regex_pattern xrule_17 n_regex_flag-> alias_24
!n_regex_pattern: n_regex_escaped
    |n_regex_unescaped
!n_regex_escaped: n_regex_unescaped xrule_19 n_regex_unescaped-> alias_25
!n_regex_unescaped: xrule_20-> alias_26
!n_regex_flag: xrule_21-> alias_27
!xrule_0: "\\""
!xrule_1: (n_dstrchar)*
!xrule_2: "'"
!xrule_3: (n_sstrchar)*
!xrule_4: "`"
!xrule_5: (/[^`]/)*
!xrule_6: "\\\\"
!xrule_7: "\\'"
!xrule_8: "u"
!xrule_9: "("
!xrule_10: ")"
!xrule_11: ","
!xrule_12: (n__ws_maybe xrule_11 n__ws_maybe n_argument)*
!xrule_13: "."
!xrule_14: "="
!xrule_15: (/[a-zA-Z0-9-_]/)+
!xrule_16: (/[\s]/)*
!xrule_17: "/"
!xrule_18: "s/"
!xrule_19: "\\/"
!xrule_20: (/[^\/]/)*
!xrule_21: (/[a-z]/)*"""

from js2py.pyjs import *

# setting scope
var = Scope(JS_BUILTINS)
set_global_object(var)

# Code follows:
var.registers(["nonspace", "id", "flatten", "join"])


@Js
def PyJsHoisted_id_(x, this, arguments, var=var):
    var = Scope({"x": x, "this": this, "arguments": arguments}, var)
    var.registers(["x"])
    return var.get("x").get("0")


PyJsHoisted_id_.func_name = "id"
var.put("id", PyJsHoisted_id_)
Js("use strict")
pass


@Js
def PyJs_flatten_0_(list, this, arguments, var=var):
    var = Scope({"list": list, "this": this, "arguments": arguments, "flatten": PyJs_flatten_0_}, var)
    var.registers(["list"])

    @Js
    def PyJs_anonymous_1_(a, b, this, arguments, var=var):
        var = Scope({"a": a, "b": b, "this": this, "arguments": arguments}, var)
        var.registers(["a", "b"])
        return var.get("a").callprop(
            "concat",
            (var.get("flatten")(var.get("b")) if var.get("Array").callprop("isArray", var.get("b")) else var.get("b")),
        )

    PyJs_anonymous_1_._set_name("anonymous")
    return var.get("list").callprop("reduce", PyJs_anonymous_1_, Js([]))


PyJs_flatten_0_._set_name("flatten")
var.put("flatten", PyJs_flatten_0_)


@Js
def PyJs_nonspace_2_(list, this, arguments, var=var):
    var = Scope({"list": list, "this": this, "arguments": arguments, "nonspace": PyJs_nonspace_2_}, var)
    var.registers(["list"])

    @Js
    def PyJs_anonymous_3_(item, this, arguments, var=var):
        var = Scope({"item": item, "this": this, "arguments": arguments}, var)
        var.registers(["item"])
        return (var.get("item") and var.get("item").get("type")) and (var.get("item").get("type") != Js("space"))

    PyJs_anonymous_3_._set_name("anonymous")
    return var.get("flatten")(var.get("list")).callprop("filter", PyJs_anonymous_3_)


PyJs_nonspace_2_._set_name("nonspace")
var.put("nonspace", PyJs_nonspace_2_)


@Js
def PyJs_join_4_(list, this, arguments, var=var):
    var = Scope({"list": list, "this": this, "arguments": arguments, "join": PyJs_join_4_}, var)
    var.registers(["list"])
    return var.get("flatten")(var.get("list")).callprop("join", Js(""))


PyJs_join_4_._set_name("join")
var.put("join", PyJs_join_4_)


@Js
def PyJs_alias_0_5_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_0": PyJs_alias_0_5_}, var)
    var.registers(["d"])
    return var.get("d").get("1").callprop("join", Js(""))


PyJs_alias_0_5_._set_name("alias_0")
var.put("alias_0", PyJs_alias_0_5_)


@Js
def PyJs_alias_1_6_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_1": PyJs_alias_1_6_}, var)
    var.registers(["d"])
    return var.get("d").get("1").callprop("join", Js(""))


PyJs_alias_1_6_._set_name("alias_1")
var.put("alias_1", PyJs_alias_1_6_)


@Js
def PyJs_alias_2_7_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_2": PyJs_alias_2_7_}, var)
    var.registers(["d"])
    return var.get("d").get("1").callprop("join", Js(""))


PyJs_alias_2_7_._set_name("alias_2")
var.put("alias_2", PyJs_alias_2_7_)
var.put("alias_3", var.get("id"))


@Js
def PyJs_alias_4_8_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_4": PyJs_alias_4_8_}, var)
    var.registers(["d"])
    return var.get("JSON").callprop("parse", ((Js('"') + var.get("d").callprop("join", Js(""))) + Js('"')))


PyJs_alias_4_8_._set_name("alias_4")
var.put("alias_4", PyJs_alias_4_8_)
var.put("alias_5", var.get("id"))


@Js
def PyJs_alias_6_9_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_6": PyJs_alias_6_9_}, var)
    var.registers(["d"])
    return var.get("JSON").callprop("parse", ((Js('"') + var.get("d").callprop("join", Js(""))) + Js('"')))


PyJs_alias_6_9_._set_name("alias_6")
var.put("alias_6", PyJs_alias_6_9_)


@Js
def PyJs_alias_7_10_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_7": PyJs_alias_7_10_}, var)
    var.registers(["d"])
    return Js("'")


PyJs_alias_7_10_._set_name("alias_7")
var.put("alias_7", PyJs_alias_7_10_)
var.put("alias_8", var.get("id"))


@Js
def PyJs_alias_9_11_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_9": PyJs_alias_9_11_}, var)
    var.registers(["d"])
    return var.get("d").callprop("join", Js(""))


PyJs_alias_9_11_._set_name("alias_9")
var.put("alias_9", PyJs_alias_9_11_)


@Js
def PyJs_alias_10_12_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_10": PyJs_alias_10_12_}, var)
    var.registers(["d"])
    return var.get("d").get("1")


PyJs_alias_10_12_._set_name("alias_10")
var.put("alias_10", PyJs_alias_10_12_)
var.put("alias_11", var.get("id"))
var.put("alias_12", var.get("id"))


@Js
def PyJs_alias_13_13_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_13": PyJs_alias_13_13_}, var)
    var.registers(["d"])
    return Js({"type": Js("function"), "name": var.get("d").get("0"), "args": var.get("d").get("2")})


PyJs_alias_13_13_._set_name("alias_13")
var.put("alias_13", PyJs_alias_13_13_)
var.put("alias_14", var.get("id"))
var.put("alias_15", var.get("nonspace"))


@Js
def PyJs_alias_16_14_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_16": PyJs_alias_16_14_}, var)
    var.registers(["d"])
    return Js({"type": Js("field"), "table": var.get("d").get("0"), "column": var.get("d").get("2")})


PyJs_alias_16_14_._set_name("alias_16")
var.put("alias_16", PyJs_alias_16_14_)


@Js
def PyJs_alias_17_15_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_17": PyJs_alias_17_15_}, var)
    var.registers(["d"])
    return Js({"type": Js("named_arg"), "key": var.get("d").get("0"), "value": var.get("d").get("2")})


PyJs_alias_17_15_._set_name("alias_17")
var.put("alias_17", PyJs_alias_17_15_)


@Js
def PyJs_alias_18_16_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_18": PyJs_alias_18_16_}, var)
    var.registers(["d"])
    return Js({"type": Js("string"), "value": var.get("d").get("0")})


PyJs_alias_18_16_._set_name("alias_18")
var.put("alias_18", PyJs_alias_18_16_)
var.put("alias_19", var.get("id"))
var.put("alias_20", var.get("id"))
var.put("alias_21", var.get("join"))


@Js
def PyJs_alias_22_17_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_22": PyJs_alias_22_17_}, var)
    var.registers(["d"])
    return Js({"type": Js("space"), "value": var.get("join")(var.get("d"))})


PyJs_alias_22_17_._set_name("alias_22")
var.put("alias_22", PyJs_alias_22_17_)


@Js
def PyJs_alias_23_18_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_23": PyJs_alias_23_18_}, var)
    var.registers(["d"])
    return Js({"type": Js("regex"), "pattern": var.get("d").get("1").get("0"), "flags": var.get("d").get("3"),})


PyJs_alias_23_18_._set_name("alias_23")
var.put("alias_23", PyJs_alias_23_18_)


@Js
def PyJs_alias_24_19_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_24": PyJs_alias_24_19_}, var)
    var.registers(["d"])
    return Js(
        {
            "type": Js("regex"),
            "pattern": var.get("d").get("1").get("0"),
            "replace": var.get("d").get("3").get("0").callprop("replace", Js(""), Js("")),
            "flags": var.get("d").get("5"),
        }
    )


PyJs_alias_24_19_._set_name("alias_24")
var.put("alias_24", PyJs_alias_24_19_)


@Js
def PyJs_alias_25_20_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_25": PyJs_alias_25_20_}, var)
    var.registers(["d"])
    return var.get("flatten")(var.get("d")).callprop("join", Js(""))


PyJs_alias_25_20_._set_name("alias_25")
var.put("alias_25", PyJs_alias_25_20_)
var.put("alias_26", var.get("join"))
var.put("alias_27", var.get("join"))
pass


class TransformNearley(Transformer):
    alias_0 = var.get("alias_0").to_python()
    alias_1 = var.get("alias_1").to_python()
    alias_2 = var.get("alias_2").to_python()
    alias_3 = var.get("alias_3").to_python()
    alias_4 = var.get("alias_4").to_python()
    alias_5 = var.get("alias_5").to_python()
    alias_6 = var.get("alias_6").to_python()
    alias_7 = var.get("alias_7").to_python()
    alias_8 = var.get("alias_8").to_python()
    alias_9 = var.get("alias_9").to_python()
    alias_10 = var.get("alias_10").to_python()
    alias_11 = var.get("alias_11").to_python()
    alias_12 = var.get("alias_12").to_python()
    alias_13 = var.get("alias_13").to_python()
    alias_14 = var.get("alias_14").to_python()
    alias_15 = var.get("alias_15").to_python()
    alias_16 = var.get("alias_16").to_python()
    alias_17 = var.get("alias_17").to_python()
    alias_18 = var.get("alias_18").to_python()
    alias_19 = var.get("alias_19").to_python()
    alias_20 = var.get("alias_20").to_python()
    alias_21 = var.get("alias_21").to_python()
    alias_22 = var.get("alias_22").to_python()
    alias_23 = var.get("alias_23").to_python()
    alias_24 = var.get("alias_24").to_python()
    alias_25 = var.get("alias_25").to_python()
    alias_26 = var.get("alias_26").to_python()
    alias_27 = var.get("alias_27").to_python()
    __default__ = lambda self, n, c, m: c if c else None


parser = Lark(grammar, start="n_start", maybe_placeholders=False)


def parse(text):
    return TransformNearley().transform(parser.parse(text)).to_dict()
