# WORKING

using Zygote: Zygote
using LinearAlgebra: dot, norm
using Random: rand
using Test

struct Trajectory{ST,GT}
    initial_state::ST
    generator::GT
    target_state::Union{Nothing,ST}
    weight::Float64
    kwargs::Dict{Symbol,Any}

    function Trajectory(
        initial_state::ST,
        generator::GT;
        target_state::Union{Nothing,ST} = nothing,
        weight = 1.0,
        kwargs...
    ) where {ST,GT}
        new{ST,GT}(initial_state, generator, target_state, weight, kwargs)
    end

end


function Base.getproperty(traj::Trajectory, name::Symbol)
    if name in (:initial_state, :generator, :target_state, :weight)
        return getfield(traj, name)
    else
        kwargs = getfield(traj, :kwargs)
        return get(kwargs, name) do
            error("type Trajectory has no property $name")
        end
    end
end


function make_automatic_xi(g_b)
    function automatic_xi(Ψ, trajectory, tlist, n)
        grad = Zygote.gradient(psi -> g_b(psi, trajectory, tlist, n), Ψ)[1]
        if isnothing(grad)
            # g_b does not depend on Ψ
            return zero(Ψ)
        end
        return -0.5 * grad
    end
    return automatic_xi
end

function g_b(Ψ, traj, tlist, n)
    return real(dot(Ψ, traj.D * Ψ))
end


function test()
    N = 4
    A = rand(ComplexF64, N, N)
    D = A * A' / N
    H = nothing
    tlist = [0.0, 1.0]
    Ψ = rand(ComplexF64, N)
    Ψ ./ norm(Ψ)
    traj = Trajectory(Ψ, H; D)
    xi = make_automatic_xi(g_b)
    ξ = xi(Ψ, traj, tlist, 1)
    @show ξ
    ξ_expected = -D * Ψ
    @test norm(ξ - ξ_expected) < 1e-14
end


function test2()
    N = 4
    A1 = rand(ComplexF64, N, N)
    D1 = A1 * A1' / N
    A2 = rand(ComplexF64, N, N)
    D2 = A2 * A2' / N
    H = nothing
    Ψ = rand(ComplexF64, N)
    Ψ ./ norm(Ψ)
    tlist = [0.0, 1.0]
    traj1 = Trajectory(Ψ, H; D=D1)
    traj2 = Trajectory(Ψ, H; D=D2)
    xi = make_automatic_xi(g_b)
    ξ1 = xi(Ψ, traj1, tlist, 1)
    @show ξ1
    ξ1_expected = -D1 * Ψ
    @test norm(ξ1 - ξ1_expected) < 1e-14
    ξ2 = xi(Ψ, traj2, tlist, 1)
    @show ξ2
    ξ2_expected = -D2 * Ψ
    @test norm(ξ2 - ξ2_expected) < 1e-14
end


# The following definitions are required to fix the problem:

using ChainRulesCore: ChainRulesCore, NoTangent

#=
function ChainRulesCore.rrule(::typeof(getproperty), traj::Trajectory, name::Symbol)
    val = getproperty(traj, name)
    function getproperty_pullback(Δ)
        return NoTangent(), NoTangent(), NoTangent()
    end
    return val, getproperty_pullback
end
=#


function ChainRulesCore.rrule(::typeof(getproperty), traj::Trajectory, name::Symbol)
    val = getproperty(traj, name)
    if name in (:initial_state, :generator, :target_state, :weight)
        function field_pullback(Δ)
            dt = ChainRulesCore.Tangent{typeof(traj)}(; (name => Δ,)...)
            return NoTangent(), dt, NoTangent()
        end
        return val, field_pullback
    else
        # kwargs-stored property: route gradient back into the kwargs Dict
        function kwargs_pullback(Δ)
            dkwargs = Dict{Symbol,Any}(name => Δ)
            dt = ChainRulesCore.Tangent{typeof(traj)}(; kwargs=dkwargs)
            return NoTangent(), dt, NoTangent()
        end
        return val, kwargs_pullback
    end
end


test()
test2()
