# BROKEN

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
    ξ_expected = -D * Ψ
    @test norm(ξ - ξ_expected) < 1e-14
end

test()
