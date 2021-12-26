using KroneckerFit
using Random
using SparseArrays
using Kronecker
using BenchmarkTools
using DataFrames
using RollingFunctions
using Plots
using CSV

import Dates
import Arrow

g = CSV.read("p2p-Gnutella25.txt", DataFrame, comment="#") .+ 1
k = g |> eachcol .|> maximum |> maximum |> x -> ceil(Int, log(2, x))
G = KroneckerFit.Graph(sparse(g[!, 1], g[!, 2], [true for _ in g[!, 1]], 2^k, 2^k))
P = Kronecker.kronecker([.7 .6; .4 .3], k)

@warn "PROBLEM" G KroneckerFit.factors(P)[1]

diagnostics = NamedTuple[]

T = Float64
entropy = MersenneTwister(42)
policy = KroneckerFit.GradientDescentPolicy{T}(
    steps = 50,
    expectation = KroneckerFit.ExpectationPolicy{T}(
        ω = 0.6,
        burn = 10000,
        samples = 100000,
        ∇logℒ = KroneckerFit.ApproximationPolicy{T}(
            ε = 1e-8,
            n = 100,
        ),
        trace_burn = (event -> push!(diagnostics, event)),
        trace_sample = (event -> push!(diagnostics, event)),
    ),
    #update = KroneckerFit.GradientUpdatePolicy{T}(λ = 1e-5),#, range = (0.0, 1.0)),
    trace = x -> @warn("applied", x...),
)

KroneckerFit.optimize!(P; G, policy, entropy)

@warn "Θ" KroneckerFit.factors(P)[1]

swaps = DataFrame(diagnostics)
Arrow.write("swaps-$(Dates.now()).arrow", swaps)
swaps = transform(
    swaps,
    #:rejected => (x -> 1 ./ (1 .+ x)) => :acceptance),
    :rejected => (x -> runmean(1 ./ (1 .+ x), 100)) => :acceptance_mean,
    #:acceptance => (x -> runstd(x, 100)) => :acceptance_std,
)

plot(swaps.acceptance_mean)