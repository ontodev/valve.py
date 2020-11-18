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
!n__ws_maybe: xrule_9-> alias_10
!n__ws: xrule_10-> alias_11
!n_wschar: /[ \t\n\v\f]/x-> alias_12
!n_expression: n_ws n_expression_trimmed n_ws-> alias_13
!n_expression_trimmed: n_string-> alias_14
    |n_function-> alias_15
!n_function: n_function_name xrule_11 n_arguments xrule_12-> alias_16
!n_function_name: n_alphanum-> alias_17
!n_arguments: n__ws_maybe n_argument xrule_14 n__ws_maybe-> alias_18
!n_argument: n_string
    |n_field
    |n_function
    |n_named_arg
    |n_regex
!n_field: n_label xrule_15 n_label-> alias_19
!n_named_arg: n_label xrule_16 n_label-> alias_20
!n_string: n_label-> alias_21
!n_label: n_alphanum-> alias_22
    |n_dqstring-> alias_23
!n_alphanum: xrule_17-> alias_24
!n_ws: xrule_18-> alias_25
!n_regex: n_regex_sub
    |n_regex_match
!n_regex_match: xrule_19 n_regex_pattern xrule_19 n_regex_flag-> alias_26
!n_regex_sub: xrule_20 n_regex_pattern xrule_19 n_regex_pattern xrule_19 n_regex_flag-> alias_27
!n_regex_pattern: n_regex_escaped
    |n_regex_unescaped
!n_regex_escaped: n_regex_unescaped xrule_21 n_regex_unescaped-> alias_28
!n_regex_unescaped: xrule_22-> alias_29
!n_regex_flag: xrule_23-> alias_30
!xrule_0: "\\""
!xrule_1: (n_dstrchar)*
!xrule_2: "'"
!xrule_3: (n_sstrchar)*
!xrule_4: "`"
!xrule_5: (/[^`]/)*
!xrule_6: "\\\\"
!xrule_7: "\\'"
!xrule_8: "u"
!xrule_9: (n_wschar)*
!xrule_10: (n_wschar)+
!xrule_11: "("
!xrule_12: ")"
!xrule_13: ","
!xrule_14: (n__ws_maybe xrule_13 n__ws_maybe n_argument)*
!xrule_15: "."
!xrule_16: "="
!xrule_17: (/[a-zA-Z0-9-_]/)+
!xrule_18: (/[ \t\n\v\f]/x)*
!xrule_19: "/"
!xrule_20: "s/"
!xrule_21: "\\/"
!xrule_22: (/[^\/]/)*
!xrule_23: (/[a-z]/)*"""

from js2py.pyjs import *

# setting scope
var = Scope(JS_BUILTINS)
set_global_object(var)

# Code follows:
var.registers(["flatten", "join", "id", "object", "_typeof"])


@Js
def PyJsHoisted_id_(x, this, arguments, var=var):
    var = Scope({"x": x, "this": this, "arguments": arguments}, var)
    var.registers(["x"])
    return var.get("x").get("0")


PyJsHoisted_id_.func_name = "id"
var.put("id", PyJsHoisted_id_)
Js("use strict")


@Js
def PyJs_anonymous_0_(obj, this, arguments, var=var):
    var = Scope({"obj": obj, "this": this, "arguments": arguments}, var)
    var.registers(["obj"])
    return var.get("obj", throw=False).typeof()


PyJs_anonymous_0_._set_name("anonymous")


@Js
def PyJs_anonymous_1_(obj, this, arguments, var=var):
    var = Scope({"obj": obj, "this": this, "arguments": arguments}, var)
    var.registers(["obj"])
    return (
        Js("symbol")
        if (
            (
                (
                    var.get("obj")
                    and PyJsStrictEq(var.get("Symbol", throw=False).typeof(), Js("function"))
                )
                and PyJsStrictEq(var.get("obj").get("constructor"), var.get("Symbol"))
            )
            and PyJsStrictNeq(var.get("obj"), var.get("Symbol").get("prototype"))
        )
        else var.get("obj", throw=False).typeof()
    )


PyJs_anonymous_1_._set_name("anonymous")
var.put(
    "_typeof",
    (
        PyJs_anonymous_0_
        if (
            PyJsStrictEq(var.get("Symbol", throw=False).typeof(), Js("function"))
            and PyJsStrictEq(var.get("Symbol").get("iterator").typeof(), Js("symbol"))
        )
        else PyJs_anonymous_1_
    ),
)
pass


@Js
def PyJs_flatten_2_(list, this, arguments, var=var):
    var = Scope(
        {"list": list, "this": this, "arguments": arguments, "flatten": PyJs_flatten_2_}, var
    )
    var.registers(["list"])

    @Js
    def PyJs_anonymous_3_(a, b, this, arguments, var=var):
        var = Scope({"a": a, "b": b, "this": this, "arguments": arguments}, var)
        var.registers(["b", "a"])
        return var.get("a").callprop(
            "concat",
            (
                var.get("flatten")(var.get("b"))
                if var.get("Array").callprop("isArray", var.get("b"))
                else var.get("b")
            ),
        )

    PyJs_anonymous_3_._set_name("anonymous")
    return var.get("list").callprop("reduce", PyJs_anonymous_3_, Js([]))


PyJs_flatten_2_._set_name("flatten")
var.put("flatten", PyJs_flatten_2_)


@Js
def PyJs_object_4_(list, this, arguments, var=var):
    var = Scope({"list": list, "this": this, "arguments": arguments, "object": PyJs_object_4_}, var)
    var.registers(["list"])

    @Js
    def PyJs_anonymous_5_(item, this, arguments, var=var):
        var = Scope({"item": item, "this": this, "arguments": arguments}, var)
        var.registers(["item"])
        return var.get("item") and (
            (
                Js("undefined")
                if PyJsStrictEq(var.get("item", throw=False).typeof(), Js("undefined"))
                else var.get("_typeof")(var.get("item"))
            )
            == Js("object")
        )

    PyJs_anonymous_5_._set_name("anonymous")
    return var.get("list").callprop("filter", PyJs_anonymous_5_).get("0")


PyJs_object_4_._set_name("object")
var.put("object", PyJs_object_4_)


@Js
def PyJs_join_6_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "join": PyJs_join_6_}, var)
    var.registers(["d"])
    return var.get("flatten")(var.get("d")).callprop("join", Js(""))


PyJs_join_6_._set_name("join")
var.put("join", PyJs_join_6_)


@Js
def PyJs_alias_0_7_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_0": PyJs_alias_0_7_}, var)
    var.registers(["d"])
    return var.get("d").get("1").callprop("join", Js(""))


PyJs_alias_0_7_._set_name("alias_0")
var.put("alias_0", PyJs_alias_0_7_)


@Js
def PyJs_alias_1_8_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_1": PyJs_alias_1_8_}, var)
    var.registers(["d"])
    return var.get("d").get("1").callprop("join", Js(""))


PyJs_alias_1_8_._set_name("alias_1")
var.put("alias_1", PyJs_alias_1_8_)


@Js
def PyJs_alias_2_9_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_2": PyJs_alias_2_9_}, var)
    var.registers(["d"])
    return var.get("d").get("1").callprop("join", Js(""))


PyJs_alias_2_9_._set_name("alias_2")
var.put("alias_2", PyJs_alias_2_9_)
var.put("alias_3", var.get("id"))


@Js
def PyJs_alias_4_10_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_4": PyJs_alias_4_10_}, var)
    var.registers(["d"])
    return var.get("JSON").callprop(
        "parse", ((Js('"') + var.get("d").callprop("join", Js(""))) + Js('"'))
    )


PyJs_alias_4_10_._set_name("alias_4")
var.put("alias_4", PyJs_alias_4_10_)
var.put("alias_5", var.get("id"))


@Js
def PyJs_alias_6_11_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_6": PyJs_alias_6_11_}, var)
    var.registers(["d"])
    return var.get("JSON").callprop(
        "parse", ((Js('"') + var.get("d").callprop("join", Js(""))) + Js('"'))
    )


PyJs_alias_6_11_._set_name("alias_6")
var.put("alias_6", PyJs_alias_6_11_)


@Js
def PyJs_alias_7_12_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_7": PyJs_alias_7_12_}, var)
    var.registers(["d"])
    return Js("'")


PyJs_alias_7_12_._set_name("alias_7")
var.put("alias_7", PyJs_alias_7_12_)
var.put("alias_8", var.get("id"))


@Js
def PyJs_alias_9_13_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_9": PyJs_alias_9_13_}, var)
    var.registers(["d"])
    return var.get("d").callprop("join", Js(""))


PyJs_alias_9_13_._set_name("alias_9")
var.put("alias_9", PyJs_alias_9_13_)


@Js
def PyJs_alias_10_14_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_10": PyJs_alias_10_14_}, var)
    var.registers(["d"])
    return var.get(u"null")


PyJs_alias_10_14_._set_name("alias_10")
var.put("alias_10", PyJs_alias_10_14_)


@Js
def PyJs_alias_11_15_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_11": PyJs_alias_11_15_}, var)
    var.registers(["d"])
    return var.get(u"null")


PyJs_alias_11_15_._set_name("alias_11")
var.put("alias_11", PyJs_alias_11_15_)
var.put("alias_12", var.get("id"))


@Js
def PyJs_alias_13_16_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_13": PyJs_alias_13_16_}, var)
    var.registers(["d"])
    return var.get("d").get("1")


PyJs_alias_13_16_._set_name("alias_13")
var.put("alias_13", PyJs_alias_13_16_)
var.put("alias_14", var.get("id"))
var.put("alias_15", var.get("id"))


@Js
def PyJs_alias_16_17_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_16": PyJs_alias_16_17_}, var)
    var.registers(["d"])
    return Js(
        {"type": Js("function"), "name": var.get("d").get("0"), "args": var.get("d").get("2")}
    )


PyJs_alias_16_17_._set_name("alias_16")
var.put("alias_16", PyJs_alias_16_17_)
var.put("alias_17", var.get("id"))


@Js
def PyJs_alias_18_18_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_18": PyJs_alias_18_18_}, var)
    var.registers(["d"])

    @Js
    def PyJs_anonymous_19_(item, this, arguments, var=var):
        var = Scope({"item": item, "this": this, "arguments": arguments}, var)
        var.registers(["item"])
        return var.get("item") and (var.get("item") != Js(","))

    PyJs_anonymous_19_._set_name("anonymous")
    return var.get("flatten")(var.get("d")).callprop("filter", PyJs_anonymous_19_)


PyJs_alias_18_18_._set_name("alias_18")
var.put("alias_18", PyJs_alias_18_18_)


@Js
def PyJs_alias_19_20_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_19": PyJs_alias_19_20_}, var)
    var.registers(["d"])
    return Js(
        {"type": Js("field"), "table": var.get("d").get("0"), "column": var.get("d").get("2")}
    )


PyJs_alias_19_20_._set_name("alias_19")
var.put("alias_19", PyJs_alias_19_20_)


@Js
def PyJs_alias_20_21_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_20": PyJs_alias_20_21_}, var)
    var.registers(["d"])
    return Js(
        {"type": Js("named_arg"), "key": var.get("d").get("0"), "value": var.get("d").get("2")}
    )


PyJs_alias_20_21_._set_name("alias_20")
var.put("alias_20", PyJs_alias_20_21_)


@Js
def PyJs_alias_21_22_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_21": PyJs_alias_21_22_}, var)
    var.registers(["d"])
    return Js({"type": Js("string"), "value": var.get("d").get("0")})


PyJs_alias_21_22_._set_name("alias_21")
var.put("alias_21", PyJs_alias_21_22_)
var.put("alias_22", var.get("id"))
var.put("alias_23", var.get("id"))
var.put("alias_24", var.get("join"))
var.put("alias_25", var.get("join"))


@Js
def PyJs_alias_26_23_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_26": PyJs_alias_26_23_}, var)
    var.registers(["d"])
    return Js(
        {
            "type": Js("regex"),
            "pattern": var.get("d").get("1").get("0"),
            "flags": var.get("d").get("3"),
        }
    )


PyJs_alias_26_23_._set_name("alias_26")
var.put("alias_26", PyJs_alias_26_23_)


@Js
def PyJs_alias_27_24_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_27": PyJs_alias_27_24_}, var)
    var.registers(["d"])
    return Js(
        {
            "type": Js("regex"),
            "pattern": var.get("d").get("1").get("0"),
            "replace": var.get("d").get("3").get("0").callprop("replace", Js(""), Js("")),
            "flags": var.get("d").get("5"),
        }
    )


PyJs_alias_27_24_._set_name("alias_27")
var.put("alias_27", PyJs_alias_27_24_)


@Js
def PyJs_alias_28_25_(d, this, arguments, var=var):
    var = Scope({"d": d, "this": this, "arguments": arguments, "alias_28": PyJs_alias_28_25_}, var)
    var.registers(["d"])
    return var.get("flatten")(var.get("d")).callprop("join", Js(""))


PyJs_alias_28_25_._set_name("alias_28")
var.put("alias_28", PyJs_alias_28_25_)
var.put("alias_29", var.get("join"))
var.put("alias_30", var.get("join"))
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
    alias_28 = var.get("alias_28").to_python()
    alias_29 = var.get("alias_29").to_python()
    alias_30 = var.get("alias_30").to_python()
    __default__ = lambda self, n, c, m: c if c else None


parser = Lark(grammar, start="n_expression", maybe_placeholders=False)


def parse(text):
    return TransformNearley().transform(parser.parse(text)).to_dict()
