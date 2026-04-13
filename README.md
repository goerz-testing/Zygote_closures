# Test of Zygote differentiation around closures

This is a MWE of an issue discussed in https://discourse.julialang.org/t/zygote-access-to-undefined-reference/136704.

Initially, the question was Zygote's behavior around certain closures.

Ultimately, it turned into a question of differentiating w.r.t. structs that have custom `getproperty` routines.
