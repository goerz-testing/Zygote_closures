# WORKS

using Zygote: Zygote
using LinearAlgebra: dot, norm
using Random: rand
using Test

struct Trajectory
    D :: Matrix{ComplexF64}
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


function test1()
    N = 4
    A = rand(ComplexF64, N, N)
    D = A * A' / N
    tlist = [0.0, 1.0]
    traj = Trajectory(D)
    Ψ = rand(ComplexF64, N)
    Ψ ./ norm(Ψ)
    xi = make_automatic_xi(g_b)
    ξ = xi(Ψ, traj, tlist, 1)
    ξ_expected = -D * Ψ
    @test norm(ξ - ξ_expected) < 1e-14
end


function test2()
    N = 4
    A1 = rand(ComplexF64, N, N)
    D1 = A1 * A1' / N
    A2 = rand(ComplexF64, N, N)
    D2 = A2 * A2' / N
    tlist = [0.0, 1.0]
    traj1 = Trajectory(D1)
    traj2 = Trajectory(D2)
    Ψ = rand(ComplexF64, N)
    Ψ ./ norm(Ψ)
    xi = make_automatic_xi(g_b)
    ξ1 = xi(Ψ, traj1, tlist, 1)
    ξ1_expected = -D1 * Ψ
    @test norm(ξ1 - ξ1_expected) < 1e-14
    ξ2 = xi(Ψ, traj2, tlist, 1)
    ξ2_expected = -D2 * Ψ
    @test norm(ξ2 - ξ2_expected) < 1e-14
end


test1()
test2()
