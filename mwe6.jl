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

function J_T(Ψ; Ψtgt, N)
    return 1 - (abs2(dot(Ψ, Ψtgt)) / N)
end

function f(traj; Ψtgt, N)
    return J_T(traj.initial_state; Ψtgt, N)
end


using ChainRulesCore: ChainRulesCore, NoTangent


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



function test()
    N = 4
    H = nothing
    Ψ = rand(ComplexF64, N)
    Ψ ./ norm(Ψ)
    Ψtgt = zeros(ComplexF64, N)
    Ψtgt[1] = 1.0
    traj = Trajectory(Ψ, H)
    @test f(traj; Ψtgt, N) > 0.0
    grad = Zygote.gradient(traj -> f(traj; Ψtgt, N), traj)[1]
    @show grad
    @test grad isa NamedTuple
    @test grad.initial_state isa Vector
end

test()
