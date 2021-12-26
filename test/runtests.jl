using KroneckerFit

using Test
using Random

import ForwardDiff
import Kronecker

# REMOVE
using BenchmarkTools

include("graphs.jl")

@testset "multiplicity" begin
    multiplicities(M, n, factor) = [
        KroneckerFit.multiplicity(M, entry, n, factor)
        for entry in CartesianIndices(M)
    ]

    @test multiplicities(
        ["a" "b"; "c" "d"; "e" "f"],
        1,
        CartesianIndex(2, 2)
    ) == [
        0 0
        0 1
        0 0
    ]

    @test multiplicities(
        Kronecker.kronecker(["a" "b" "c"; "d" "e" "f"], ["A" "B"; "C" "D"; "E" "F"]),
        1,
        CartesianIndex(2, 2)
    ) == [
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 1 1 0 0
        0 0 1 1 0 0
        0 0 1 1 0 0
    ]

    @test multiplicities(
        Kronecker.kronecker(["a" "b" "c"; "d" "e" "f"], ["A" "B"; "C" "D"; "E" "F"]),
        2,
        CartesianIndex(2, 2)
    ) == [
        0 0 0 0 0 0
        0 1 0 1 0 1
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 1 0 1 0 1
        0 0 0 0 0 0
    ]

    @test multiplicities(
        Kronecker.kronecker(["a" "b"; "c" "d"], 3),
        2,
        CartesianIndex(2, 2)
    ) == [
        0 0 0 0 0 0 0 0
        0 1 0 1 0 1 0 1
        0 0 1 1 0 0 1 1
        0 1 1 2 0 1 1 2
        0 0 0 0 1 1 1 1
        0 1 0 1 1 2 1 2
        0 0 1 1 1 1 2 2
        0 1 1 2 1 2 2 3
    ]

    @test multiplicities(
        Kronecker.kronecker(["a" "b"; "c" "d"; "e" "f"], 2),
        2,
        CartesianIndex(2, 2)
    ) == [
        0 0 0 0
        0 1 0 1
        0 0 0 0
        0 0 1 1
        0 1 1 2
        0 0 1 1
        0 0 0 0
        0 1 0 1
        0 0 0 0
    ]
end

graph(A::AbstractMatrix) = KroneckerFit.Graph(A)
adjacency(G::KroneckerFit.Graph) = collect(G.A)

function logℒ(G, P)
    A = adjacency(G)
    sum(log.(P .* A .+ (1.0 .- P) .* (1.0 .- A)))
end
logℒ(G, P, σ) = logℒ(G, P[σ.σ, σ.σ])

function ∇logℒ(G, P::AbstractMatrix, σ)
    A = adjacency(G)
    Pσ = P[σ.σ, σ.σ]
    σ′ = invperm(σ.σ)
    (((2.0A .- 1.0) ./ (Pσ .* A .+ (1.0 .- Pσ) .* (1.0 .- A)))[σ′, σ′],)
end

function ∇logℒ(G, P::Kronecker.AbstractKroneckerProduct, σ)
    A = adjacency(G)
    Θs = Kronecker.getallfactors(P)
    Pσ = P[σ.σ, σ.σ]

    selector(n, factor) = Kronecker.kronecker((
        [n != n′ || factor == factor′ for factor′ in factors′]
        for (n′, factors′) in enumerate(CartesianIndices.(Θs))
    )...)

    Tuple(
        [
            sum(
                selector(n, factor)[σ.σ, σ.σ] .*
                    (2.0A .- 1.0) .* Pσ ./
                    (Θ[factor] .* (Pσ .* A .+ (1.0 .- Pσ) .* (1.0 .- A)))
            )
            for factor in CartesianIndices(Θ)
        ]
        for (n, Θ) in enumerate(Θs)
    )
end

∇logℒ(G, P::Kronecker.KroneckerPower, σ) =
    (reduce(.+, ∇logℒ(G, Kronecker.kronecker(Kronecker.getallfactors(P)...), σ)),)

∇logℒ(G, P) = ∇logℒ(G, P, Permutation(1:size(P, 1)))

@testset "meta test (∇)logℒ" begin
    P = [
        0.7 0.4
        0.3 0.2
    ]
    P2 = Kronecker.kronecker(P, P)

    A = Bool[
        1 1
        0 1
    ]
    G = graph(A)
    G2 = graph(Kronecker.kronecker(A, 2))
    G3 = graph(Kronecker.kronecker(A, 3))

    @test logℒ(G, P) ≈ logℒ(G, P, Permutation(1:2)) ≈ log(0.7 * 0.4 * (1.0 - 0.3) * 0.2)
    @test logℒ(G2, P2) ≈ logℒ(G2, P2, Permutation(1:4)) ≈ sum(log.([
        (      0.7 * 0.7)   (      0.7 * 0.4)   (      0.4 * 0.7)   (0.4 * 0.4)
        (1.0 - 0.7 * 0.3)   (      0.7 * 0.2)   (1.0 - 0.4 * 0.3)   (0.4 * 0.2)
        (1.0 - 0.3 * 0.7)   (1.0 - 0.3 * 0.4)   (      0.2 * 0.7)   (0.2 * 0.4)
        (1.0 - 0.3 * 0.3)   (1.0 - 0.3 * 0.2)   (1.0 - 0.2 * 0.3)   (0.2 * 0.2)
    ]))
    @test logℒ(G2, P2, Permutation([1, 4, 2, 3])) ≈ sum(log.([
        (      0.7 * 0.7)   (      0.4 * 0.4)   (      0.7 * 0.4)   (0.4 * 0.7)
        (1.0 - 0.3 * 0.3)   (      0.2 * 0.2)   (1.0 - 0.3 * 0.2)   (0.2 * 0.3)
        (1.0 - 0.7 * 0.3)   (1.0 - 0.4 * 0.2)   (      0.7 * 0.2)   (0.4 * 0.3)
        (1.0 - 0.3 * 0.7)   (1.0 - 0.2 * 0.4)   (1.0 - 0.3 * 0.4)   (0.2 * 0.7)
    ]))
    
    ∇ = ForwardDiff.gradient

    @test all(∇logℒ(G, P) .≈ (∇(X -> logℒ(G, X), P),))
    let σ = Permutation([1, 2])
        @test all(∇logℒ(G, P, σ) .≈ (∇(X -> logℒ(G, X, σ), P),))
    end

    let σ = Permutation([2, 1])
        @test all(∇logℒ(G, P, σ) .≈ (∇(X -> logℒ(G, X, σ), P),))
    end

    @test all(
        ∇logℒ(G2, Kronecker.kronecker(P, P)) .≈ (
            ∇(X -> logℒ(G2, Kronecker.kronecker(X, P)), P),
            ∇(X -> logℒ(G2, Kronecker.kronecker(P, X)), P)
        )
    )

    let σ = Permutation([1, 4, 2, 3])
        @test all(
            ∇logℒ(G2, collect(Kronecker.kronecker(P, P)), σ) .≈ (
                ∇(X -> logℒ(G2, X, σ), collect(Kronecker.kronecker(P, P))),
            )
        )
        @test all(
            ∇logℒ(G2, Kronecker.kronecker(P, P), σ) .≈ (
                ∇(X -> logℒ(G2, Kronecker.kronecker(X, P), σ), P),
                ∇(X -> logℒ(G2, Kronecker.kronecker(P, X), σ), P)
            )
        )
    end

    @test all(
        ∇logℒ(G3, Kronecker.kronecker(P, 3)) .≈ ((
            ∇(X -> logℒ(G3, Kronecker.kronecker(X, P, P)), P) .+
            ∇(X -> logℒ(G3, Kronecker.kronecker(P, X, P)), P) .+
            ∇(X -> logℒ(G3, Kronecker.kronecker(P, P, X)), P)
        ),)
    )

    let σ = Permutation([6, 3, 1, 7, 2, 5, 8, 4])
        @test all(
            ∇logℒ(G3, collect(Kronecker.kronecker(P, 3)), σ) .≈ (
                ∇(X -> logℒ(G3, X, σ), collect(Kronecker.kronecker(P, 3))),
            )
        )
        @test all(
            ∇logℒ(G3, Kronecker.kronecker(P, 3), σ) .≈ ((
                ∇(X -> logℒ(G3, Kronecker.kronecker(X, P, P), σ), P) .+
                ∇(X -> logℒ(G3, Kronecker.kronecker(P, X, P), σ), P) .+
                ∇(X -> logℒ(G3, Kronecker.kronecker(P, P, X), σ), P)
            ),)
        )
    end

    let σ = Permutation([1, 4, 2, 3]), P1 = [0.2 0.8], P2 = [0.2; 0.3; 0.4; 0.7], P3 = [0.9 0.3]
        @test all(
            ∇logℒ(G2, Kronecker.kronecker(P1, P2, P3), σ) .≈ (
                ∇(X -> logℒ(G2, Kronecker.kronecker(X, P2, P3), σ), P1),
                ∇(X -> logℒ(G2, Kronecker.kronecker(P1, X, P3), σ), P2),
                ∇(X -> logℒ(G2, Kronecker.kronecker(P1, P2, X), σ), P3),
            )
        )
    end
end

@testset "empty logℒ" begin
    # Test some trivial cases
    @test approximate_empty_logℒ(zeros(Float64, 2, 2), n = 10) == 0.0
    @test approximate_empty_logℒ(ones(Float64, 2, 2), n = 10) == -Inf
    @test approximate_empty_logℒ(Kronecker.kronecker(zeros(Float64, 2, 2), 8), n = 10) == 0.0
    @test approximate_empty_logℒ(Kronecker.kronecker(ones(Float64, 2, 2), 8), ε = 1e-10) == -Inf
    @test approximate_empty_logℒ(Kronecker.kronecker(zeros(Float64, 2, 2), zeros(Float64, 2, 2)), n = 10) == 0.0
    @test approximate_empty_logℒ(Kronecker.kronecker(ones(Float64, 2, 2), ones(Float64, 2, 2)), n = 10) == -Inf

    # Check that the floating point type is consistently used
    @test typeof(approximate_empty_logℒ(zeros(Float16, 2, 2), n = 10)) == Float16
    @test typeof(approximate_empty_logℒ(ones(Float16, 2, 2), n = 10)) == Float16
    @test typeof(approximate_empty_logℒ(Kronecker.kronecker(zeros(Float16, 2, 2), 2), n = 10)) == Float16
    @test typeof(approximate_empty_logℒ(Kronecker.kronecker(ones(Float16, 2, 2), 2), n = 10)) == Float16
    @test typeof(approximate_empty_logℒ(Kronecker.kronecker(zeros(Float16, 2, 2), zeros(Float16, 2, 2)), n = 10)) == Float16
    @test typeof(approximate_empty_logℒ(Kronecker.kronecker(ones(Float16, 2, 2), ones(Float16, 2, 2)), n = 10)) == Float16

    Θ = [
        0.875 0.75
        0.5   0.25
    ]

    # Computation should be exact for a proper Matrix Θ.
    @test approximate_empty_logℒ(Θ) == sum(log.(1.0 .- Θ))
    @test approximate_empty_logℒ(Θ, ε = 1e10) == sum(log.(1.0 .- Θ))
    @test approximate_empty_logℒ(Θ, n = 1) == sum(log.(1.0 .- Θ))
    @test approximate_empty_logℒ(Θ, ε = 1e10, n = 1) == sum(log.(1.0 .- Θ))

    # If Θ is a KroneckerPower we approximate efficiently according to
    # tolerance ε and order n... default is arbitrary tolerance in which case
    # the order does not matter.
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2)) == 0.0
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2), n = 1) == 0.0
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2), n = 2) == 0.0
    # On the other hand, if order is zero, ε does not matter.
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2), n = 0) == 0.0
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2), ε = 0.0, n = 0) == 0.0
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2), ε = 0.1, n = 0) == 0.0
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2), ε = Inf, n = 0) == 0.0
    # First-order approximation is -(Σx)ᵏ.
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2), ε = 0.0, n = 1) == -sum(Θ)^2
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 3), ε = 0.0, n = 1) == -sum(Θ)^3
    # Second-order approximation is -((Σx)ᵏ + ½(Σx²)ᵏ).
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2), ε = 0.0, n = 2) == -sum(Θ)^2 - (1/2) * sum(Θ .* Θ)^2
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 3), ε = 0.0, n = 2) == -sum(Θ)^3 - (1/2) * sum(Θ .* Θ)^3

    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 2), ε = 1e-12) ≈
        sum(log.(1.0 .- Kronecker.kronecker(Θ, 2))) atol = 2e-10
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 3), ε = 1e-12) ≈
        sum(log.(1.0 .- Kronecker.kronecker(Θ, 3))) atol = 2e-10
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 8), ε = 1e-12) ≈
        sum(log.(1.0 .- Kronecker.kronecker(Θ, 8))) atol = 2e-10
    
    # Direct computation is too expensive for k = 15, but ~30k nodes is what we
    # are aiming for, so we just calculate the target value off-line.
    @test approximate_empty_logℒ(Kronecker.kronecker(Θ, 15), ε = 1e-10) ==
        -432321.07554052793

    # Check that the approximation is equivalent, albeit slower, for general
    # AbstractKroneckerProducts that are not KroneckerPowers.
    @test approximate_empty_logℒ(Kronecker.kronecker(Tuple(Θ for _ in 1:8)...), ε = 1e-12) ==
        approximate_empty_logℒ(Kronecker.kronecker(Θ, 8), ε = 1e-12)
end

function approximate_empty_∇logℒ(P; keywords...)
    result = similar.(KroneckerFit.factors(P))
    KroneckerFit.approximate_empty_∇logℒ!(result, P; keywords...)
    result
end

@testset "empty ∇logℒ" begin
    # Test some trivial cases
    @test approximate_empty_∇logℒ(zeros(Float64, 2, 2), n = 10) == (-ones(2, 2),)
    @test approximate_empty_∇logℒ(ones(Float64, 2, 2), n = 10) == (fill(-Inf, 2, 2),)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(zeros(Float64, 2, 2), 8), n = 10) == (zeros(2, 2),)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(ones(Float64, 2, 2), 8), ε = 1e-10) == (fill(-Inf, 2, 2),)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(zeros(Float64, 2, 2), zeros(Float64, 2, 2)), n = 10) == (zeros(2, 2), zeros(2, 2))
    @test approximate_empty_∇logℒ(Kronecker.kronecker(ones(Float64, 2, 2), ones(Float64, 2, 2)), n = 10) == (fill(-Inf, 2, 2), fill(-Inf, 2, 2))


    # Check that the floating point type is consistently used
    @test typeof(approximate_empty_∇logℒ(zeros(Float16, 2, 2), n = 10)) == Tuple{Matrix{Float16}}
    @test typeof(approximate_empty_∇logℒ(ones(Float16, 2, 2), n = 10)) == Tuple{Matrix{Float16}}
    @test typeof(approximate_empty_∇logℒ(Kronecker.kronecker(zeros(Float16, 2, 2), 2), n = 10)) == Tuple{Matrix{Float16}}
    @test typeof(approximate_empty_∇logℒ(Kronecker.kronecker(ones(Float16, 2, 2), 2), n = 10)) == Tuple{Matrix{Float16}}
    @test typeof(approximate_empty_∇logℒ(Kronecker.kronecker(zeros(Float16, 2, 2), zeros(Float16, 2, 2)), n = 10)) == Tuple{Matrix{Float16}, Matrix{Float16}}
    @test typeof(approximate_empty_∇logℒ(Kronecker.kronecker(ones(Float16, 2, 2), ones(Float16, 2, 2)), n = 10)) == Tuple{Matrix{Float16}, Matrix{Float16}}

    Θ = [
        0.875 0.75
        0.5   0.25
    ]

    # Computation should be exact for a proper Matrix Θ.
    @test approximate_empty_∇logℒ(Θ) == (-1.0 ./ (1.0 .- Θ),)
    @test approximate_empty_∇logℒ(Θ, ε = 1e10) == (-1.0 ./ (1.0 .- Θ),)
    @test approximate_empty_∇logℒ(Θ, n = 1) == (-1.0 ./ (1.0 .- Θ),)
    @test approximate_empty_∇logℒ(Θ, ε = 1e10, n = 1) == (-1.0 ./ (1.0 .- Θ),)

    # If Θ is a KroneckerPower we approximate efficiently according to
    # tolerance ε and order n... default is arbitrary tolerance in which case
    # the order does not matter.
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2)) == (zeros(2, 2),)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2), n = 1) == (zeros(2, 2),)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2), n = 2) == (zeros(2, 2),)
    # On the other hand, if order is zero, ε does not matter.
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2), n = 0) == (zeros(2, 2),)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2), ε = 0.0, n = 0) == (zeros(2, 2),)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2), ε = 0.1, n = 0) == (zeros(2, 2),)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2), ε = Inf, n = 0) == (zeros(2, 2),)
    # First-order approximation is -k(Σx)^(k-1).
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2), ε = 0.0, n = 1) == (-2sum(Θ) .* ones(2, 2),)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 3), ε = 0.0, n = 1) == (-3sum(Θ).^2 .* ones(2, 2),)
    # Second-order approximation is -k(Σx)^(k-1) - k(Σx²)^(k-1)x.
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2), ε = 0.0, n = 2) == (-2sum(Θ) .- 2sum(Θ .* Θ) .* Θ,)
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 3), ε = 0.0, n = 2) == (-3sum(Θ).^2 .- 3sum(Θ .* Θ).^2 .* Θ,)

    ∇ = ForwardDiff.gradient
    target(X, k) = ∇(X -> sum(log.(1.0 .- Kronecker.kronecker(X, k))), X)

    @test all(.≈(approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 2), ε = 1e-12), (target(Θ, 2),), atol = 2e-10))
    @test all(.≈(approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 3), ε = 1e-12), (target(Θ, 3),), atol = 2e-10))
    @test all(.≈(approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 8), ε = 1e-12), (target(Θ, 8),), atol = 2e-10))

    # Direct computation is too expensive for k = 15, but ~30k nodes is what we
    # are aiming for, so we just calculate the target value off-line.
    @test approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 15), ε = 1e-10) == ([
        -2738751.9874639893 -2736773.3846443812
        -2732844.4751200434 -2728952.4403680484
    ],)

    # Check that the approximation is equivalent, albeit slower, for general
    # AbstractKroneckerProducts that are not KroneckerPowers.
    @test sum(approximate_empty_∇logℒ(Kronecker.kronecker(Tuple(Θ for _ in 1:8)...), ε = 1e-12)) ≈
        approximate_empty_∇logℒ(Kronecker.kronecker(Θ, 8), ε = 1e-12)[1] atol = 2e-10
end

function approximate_∇logℒ(G, P, σ; keywords...)
    result = similar.(KroneckerFit.factors(P))
    KroneckerFit.approximate_∇logℒ!(result; P, σ, G, keywords...)
    result
end

@testset "(∇)logℒ" begin
    # Test some trivial cases
    @test approximate_logℒ(graph(trues(1, 1)), fill(.5, 1, 1), Permutation([1])) == log(0.5)
    # @test approximate_log_likelihood(graph(zeros(2, 2)), zeros(Float64, 2, 2), [0x1, 0x2]) == ???
    # @test approximate_log_likelihood(graph(zeros(2, 2)), zeros(Float64, 2, 2), [0x1, 0x2]) == ???
    # TODO ^ test on empty graph after switch away from StaticGraphs
    @test approximate_logℒ(graph(trues(2, 2)), zeros(Float64, 2, 2), Permutation([1, 2])) == -Inf
    # @test approximate_log_likelihood(graph(ones(2, 2)), ones(Float64, 2, 2), [0x1, 0x2]) == 0.0
    # TODO ^ think about how to handle 0/1s in the swap change calculation

    Θ = [
        0.875 0.75
        0.5   0.25
    ]

    A = Bool[
        1 0
        1 1
    ]

    @test approximate_logℒ(graph(A), Θ, Permutation([1, 2])) == logℒ(graph(A), Θ)

    for k in 2:9
        G = graph(Kronecker.kronecker(A, k))
        P = Kronecker.kronecker(Θ, k)
        σ = Permutation(collect(1:2^k))
        σ′ = Permutation(collect(reverse(1:2^k)))
        σ′.σ[2], σ′.σ[end - 1] = σ′.σ[end - 1], σ′.σ[2]

        # Note that the naive (test target) calculations get inaccurate for the
        # higher-k iterations, hence the high comparison tolerance.

        @test approximate_logℒ(G, P, σ, ε = 1e-12) ≈ logℒ(G, P) atol = 1e-8
        @test approximate_logℒ(G, Kronecker.kronecker(Tuple(Θ for _ in 1:k)...), σ, ε = 1e-12) ≈ logℒ(G, P) atol = 1e-8
        @test approximate_logℒ(G, P, σ′, ε = 1e-12) ≈ logℒ(G, P, σ′) atol = 1e-8

        @test all(.≈(approximate_∇logℒ(G, P, σ, ε = 1e-12), ∇logℒ(G, P), atol = 1e-7))
        @test all(.≈(
            approximate_∇logℒ(G, Kronecker.kronecker(Tuple(Θ for _ in 1:k)...), σ, ε = 1e-12),
            ∇logℒ(G, Kronecker.kronecker(Tuple(Θ for _ in 1:k)...)),
            atol = 1e-7
        ))
        @test all(.≈(approximate_∇logℒ(G, P, σ′, ε = 1e-12), ∇logℒ(G, P, σ′), atol = 1e-7))
    end
end

function test_Δlogℒ_for_swap(G, P, σ, u, v)
    σ′ = Permutation(copy(σ.σ))
    KroneckerFit.swap!(σ′, u, v)
    @test Δlogℒ_for_swap(G, P, σ, u, v) ≈ logℒ(G, P, σ′) - logℒ(G, P, σ) atol=10^-10

    gradient = zero.(KroneckerFit.factors(P))
    KroneckerFit.add_Δ∇logℒ_for_swap!(gradient, G, P, σ, u, v)
    @test all(gradient .≈ ∇logℒ(G, P, σ′) .- ∇logℒ(G, P, σ))
end

function test_Δlogℒ_for_swap(entropy, n, k)
    test_Δlogℒ_for_swap(
        graph(rand(entropy, Bool, n^k, n^k)),
        Kronecker.kronecker(rand(entropy, n, n), k),
        randperm(entropy, UInt(n^k)),
        rand(entropy, 0x1:UInt(n^k)),
        rand(entropy, 0x1:UInt(n^k))
    )
end

@testset "Δ(∇)logℒ for swap" begin
    # Test a bunch of trivial cases.

    @test Δlogℒ_for_swap(graph(trues(1, 1)), fill(.5, 1, 1), Permutation([1]), 1, 1) == 0.0

    # Having Θ with 0- or 1-entries make likelihoods infinite and therefore the
    # change ratio undefined. Ultimately we need to handle this but for now we
    # just document the behavior.
    @test Δlogℒ_for_swap(graph(trues(1, 1)), zeros(1, 1), Permutation([1]), 1, 1) |> isnan
    @test Δlogℒ_for_swap(graph(trues(1, 1)), ones(1, 1), Permutation([1]), 1, 1) |> isnan

    Θ = [
        0.875 0.75
        0.5   0.25
    ]
    A = Bool[
        1 0
        1 1
    ]

    test_Δlogℒ_for_swap(graph(A), Θ, Permutation([1, 2]), 1, 2)
    test_Δlogℒ_for_swap(graph(A), Θ, Permutation([1, 2]), 2, 1)
    test_Δlogℒ_for_swap(graph(A), Θ, Permutation([2, 1]), 1, 2)
    test_Δlogℒ_for_swap(graph(A), Θ, Permutation([2, 1]), 2, 1)

    test_Δlogℒ_for_swap(graph(A), Kronecker.kronecker(Θ, 2), Permutation([1, 2]), 1, 2)
    test_Δlogℒ_for_swap(graph(Kronecker.kronecker(A, 2)), Kronecker.kronecker(Θ, 2), Permutation(collect(1:2^2)), 3, 2)
    test_Δlogℒ_for_swap(graph(Kronecker.kronecker(A, 3)), Kronecker.kronecker(Θ, 3), Permutation(collect(1:2^3)), 2, 6)
    test_Δlogℒ_for_swap(graph(Kronecker.kronecker(A, 6)), Kronecker.kronecker(Θ, 6), Permutation(collect(1:2^6)), 2, 6)

    test_Δlogℒ_for_swap(graph(Kronecker.kronecker(A, 2)), Kronecker.kronecker(Θ, 2), Permutation([1, 4, 2, 3]), 3, 1)
    test_Δlogℒ_for_swap(graph(Kronecker.kronecker(A, 2)), collect(Kronecker.kronecker(Θ, 2)), Permutation([1, 4, 2, 3]), 3, 1)
    test_Δlogℒ_for_swap(graph(Kronecker.kronecker(A, 2)), Kronecker.kronecker(Θ, Θ), Permutation([1, 4, 2, 3]), 3, 1)


    test_Δlogℒ_for_swap(
        graph(Bool[
            0 1 1 0 1 0 1 0
            0 1 0 1 0 0 0 0
            1 1 1 1 0 1 0 1
            0 0 0 1 1 1 0 0
            1 0 0 1 0 1 1 0
            1 0 1 1 0 0 1 0
            0 0 0 0 0 0 0 0
            1 1 1 1 1 1 1 1
        ]),
        Kronecker.kronecker(Θ, 3),
        Permutation(collect(1:2^3)),
        2,
        6
    )

    #=
    # TODO restore this after migrating away from StaticGraphs
    entropy = MersenneTwister(42)
    test_random_swap_log_likelihood_change(entropy, 2, 5)
    test_random_swap_log_likelihood_change(entropy, 3, 4)
    test_random_swap_log_likelihood_change(entropy, 10, 2)
    test_random_swap_log_likelihood_change(entropy, 2, 10)
    =#
end