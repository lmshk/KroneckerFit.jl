module KroneckerFit

using Memoize
using Random

import Kronecker

export Permutation, approximate_empty_logℒ, approximate_logℒ, Δlogℒ_for_swap

include("graphs.jl")
include("permutations.jl")

# Kronecker utilities

factors(P::AbstractMatrix) = (P,)
factors(P::Kronecker.AbstractKroneckerProduct) = Kronecker.getallfactors(P)
factors(P::Kronecker.KroneckerPower) = (P.A,)

sizes(P::AbstractMatrix) = size.(factors(P))
sizes(P::Kronecker.KroneckerPower) = fill(size(P.A), P.pow)

@memoize _factor_repetitions_by_level(P::AbstractMatrix) = (
    [sizes(P)[2:end]..., (1, 1)]
    |> reverse
    |> (xs -> Iterators.accumulate(.*, xs))
    |> collect
    |> reverse
)
# TODO: ^ the cache should probably be of limited size

_factor(
    entry::CartesianIndex{2},
    distincts::Tuple{Int, Int},
    repetitions::Tuple{Int, Int},
) = CartesianIndex(rem.(div.(Tuple(entry) .- 1, repetitions), distincts) .+ 1)


multiplicity(P::AbstractMatrix) = 1
multiplicity(P::Kronecker.KroneckerPower) = P.pow

multiplicity(
    P::AbstractMatrix,
    entry::CartesianIndex{2},
    n::Integer,
    factor::CartesianIndex{2},
) = _factor(entry, sizes(P)[n], _factor_repetitions_by_level(P)[n]) == factor

multiplicity(
    P::Kronecker.KroneckerPower,
    entry::CartesianIndex{2},
    _::Integer,
    factor::CartesianIndex{2},
) = sum(
    _factor(entry, size(P.A), repetitions) == factor
    for repetitions in _factor_repetitions_by_level(P)
)


# likelihood

empty_logℒ(P::AbstractMatrix{T}) where T <: Real =
    sum(log.(one(T) .- P))

function approximate_empty_logℒ(
    P::AbstractMatrix{T};
    ε::T = T(NaN),
    n::Int = typemax(Int),
) where T <: Real
    empty_logℒ(P)
end

function approximate_empty_logℒ(
    P::Kronecker.AbstractKroneckerProduct{T};
    ε::T = T(NaN),
    n::Int = typemax(Int),
) where T <: Real
    # TODO: Test behavior with bad values, especially <0, >1 and =1. -- 2022-01-16

    Θs = factors(P)
    
    # For non-probability inputs, logℒ is not defined.
    all(all(zero(T) .<= Θ .<= one(T)) for Θ in Θs) || throw(DomainError(Θs))

    result = zero(T)
    n ≤ 0 && return result
    
    # If all factors contain at least one 1, the series diverges to -∞, but
    # extremely slowly; no need to take the scenic route.
    all(any(isone.(Θ)) for Θ in Θs) && return typemin(T)
    
    k = multiplicity(P)
    i = 0
    Θs′ = copy.(Θs)
    while true
        i += 1
        Δ = prod(sum.(Θs′))^k / i
        Δ ≥ ε || break
        result -= Δ
        i < n || break
        for (Θ′, Θ) in zip(Θs′, Θs)
            Θ′ .*= Θ
        end
    end

    return result
end

Δlogℒ_for_link(
    P::AbstractMatrix{T},
    entry::CartesianIndex{2},
) where T <: Real =
    log(P[entry]) - log(one(T) - P[entry])

function approximate_logℒ(
    G::Graph{Index},
    P::AbstractMatrix{T},
    σ::Permutation;
    ε::T = T(NaN),
    n::Int = typemax(Int),
) where {Index <: Integer, T <: Real}
    approximate_empty_logℒ(P; ε, n) + sum(
        Δlogℒ_for_link(P, σ[edge])
        for edge in Tuple.(edges(G))
    )
end

function Δlogℒ_for_swap(
    G::Graph{Index},
    P::AbstractMatrix{T},
    σ::Permutation,
    u::Index,
    v::Index,
) where {Index <: Integer, T <: Real}
    # TODO: handle 0s and 1s in P?

    result = zero(T)

    new_index(x) = x == u ? v : x == v ? u : x
    foreach_incident_edge(G, u, v) do (x, y)
        x′ = new_index(x)
        y′ = new_index(y)
        result +=
            Δlogℒ_for_link(P, σ[(x′, y′)]) -
            Δlogℒ_for_link(P, σ[(x, y)])
    end

    return result
end


# gradient

function empty_∇logℒ!(
    ∇logℒ::Tuple{AbstractMatrix{T}},
    P::AbstractMatrix{T},
) where T <: Real
    ∇logℒ[1] .= -one(T) ./ (one(T) .- P)
end

function approximate_empty_∇logℒ!(
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
    P::AbstractMatrix{T};
    ε::T = T(NaN),
    n::Int = typemax(Int),
) where T <: Real
    empty_∇logℒ!(∇logℒ, P)
end

function approximate_empty_∇logℒ!(
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
    P::Kronecker.AbstractKroneckerProduct{T};
    ε::T = T(NaN),
    n::Int = typemax(Int),
) where T <: Real
    Θs = factors(P)
    
    # For non-probability inputs, ∇logℒ is not defined.
    all(all(zero(T) .<= Θ .<= one(T)) for Θ in Θs) || throw(DomainError(Θs))

    for ∇logΘ in ∇logℒ
        ∇logΘ .= zero(T)
    end
    n ≤ 0 && return
    
    # Like for the approximate logℒ, if all factors' entries are 1, the series
    # diverges to -∞, but too slowly; take the shortcut.
    if all(any(isone.(Θ)) for Θ in Θs)
        for ∇logΘ in ∇logℒ
            ∇logΘ .= typemin(T)
        end
        return
    end

    k = multiplicity(P)
    Θs′ = copy.(Θs)
    Δs = fill.(one(T), size.(Θs))
    # TODO: ^ Check if pre-allocating these buffers makes a performance
    # difference. -- 2021-12-21
    i = 0
    while true
        sums = sum.(Θs′)
        changing = false
        for (m, (Δ, ∇logΘ, s)) in enumerate(zip(Δs, ∇logℒ, sums))
            c = prod((s′ for (m′, s′) in enumerate(sums) if m != m′), init = 1.0)
            # TODO: ^ Check if we can get this "reducing over empty collection"
            # behavior fixed in Julia. -- 2021-12-21
            Δ .*= k * s^(k - 1) * c^k
            # It would be simpler (and faster) to do this instead:
            #
            #     Δ .*= k * prod(sums)^k / s
            #
            # Additionally, this would make the sums buffer above unnecessary.
            # However, s might be tiny so we may run into numerical issues and
            # I would like to avoid that unless profiling shows that we really
            # have a performance issue here.
            # TODO: ^ Check if we should use the log-sum-exp trick here (and
            # maybe also for approximating logℒ). -- 2021-12-21
            for x in CartesianIndices(Δ)
                δ = Δ[x]
                if δ ≥ ε
                    ∇logΘ[x] -= δ
                    changing = true
                end
            end
        end
        changing || break
        i += 1
        # i < n || @warn("ill-conditioned", i, n, Δs, ∇logℒ)
        # TODO: Do we need to warn here^? -- 2022-01-17
        i < n || break
        Δs, Θs′ = Θs′, Δs
        for (Θ′, Δ, Θ) in zip(Θs′, Δs, Θs)
            Θ′ .= Δ .* Θ
        end
    end
end

function Δ∂logℒ_for_link(
    P::AbstractMatrix{T},
    entry::CartesianIndex{2},
    n::Integer,
    factor::CartesianIndex{2},
    x::T,
) where T <: Real
    # For KroneckerPower Θs, let y := non-x factors and we have:
    #
    #   ∂/∂x (log(P[σᵤ, σᵥ]) - log(1 - P[σᵤ, σᵥ]))
    # = ∂/∂x (log(x^m * y) - log(1 - x^m * y))
    # = m / x - (-m * x^(m-1) * y) / (1 - x^m * y)
    # = m / x - ((-m / x) * x^m * y) / (1 - x^m * y)
    # = (m / x) * (1 + P[σᵤ, σᵥ] / (1 - P[σᵤ, σᵥ]))
    # = (m / x) / (1 - P[σᵤ, σᵥ]))
    # = m / (x * (1 - P[σᵤ, σᵥ]))
    #
    # For other Θs (where factors are not repeated across the entries), m
    # (the "multiplicity") becomes an indicator deciding whether the factor
    # (i.e. parameter) contributes to the (permuted) entry in P.

    multiplicity(P, entry, n, factor) / (x * (one(T) - P[entry]))
end

function approximate_∇logℒ!(
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}};
    P::AbstractMatrix{T},
    σ::Permutation,
    G::Graph{Index},
    ε::T = T(NaN),
    n::Int = typemax(Int),
) where {Index <: Integer, T <: Real}
    approximate_empty_∇logℒ!(∇logℒ, P; ε, n)
    Θs = factors(P)
    for edge in Tuple.(edges(G))
        entry = σ[edge]
        for (n, (Θ, ∇logΘ)) in enumerate(zip(Θs, ∇logℒ))
            for factor in CartesianIndices(∇logΘ)
                x = Θ[factor]
                ∇logΘ[factor] += Δ∂logℒ_for_link(P, entry, n, factor, x)
                # TODO: Check if we can make this more efficient as described
                # below. -- 2021-12-20
            end
        end
    end
end

function add_Δ∇logℒ_for_swap!(
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
    G::Graph{Index},
    P::AbstractMatrix{T},
    σ::Permutation,
    u::Index,
    v::Index,
) where {Index <: Integer, T <: Real}
    Θs = factors(P)
    new_index(x) = x == u ? v : x == v ? u : x
    foreach_incident_edge(G, u, v) do (x, y)
        x′ = new_index(x)
        y′ = new_index(y)
        entry = σ[(x, y)]
        entry′ = σ[(x′, y′)]
        for (n, (Θ, ∇logΘ)) in enumerate(zip(Θs, ∇logℒ))
            for factor in CartesianIndices(∇logΘ)
                x = Θ[factor]
                ∇logΘ[factor] +=
                    Δ∂logℒ_for_link(P, entry′, n, factor, x) -
                    Δ∂logℒ_for_link(P, entry, n, factor, x)
                # TODO: Check if we can save time by enumerating and adding the
                # factors (in multiplicity) contributing to the derivative one
                # by one (instead of calculating the potentially zero
                # multiplicity) for each parameter. I.e. for each incident
                # edge, list its factors (i.e. parameter occurences) and add
                # derivative entries for them to the ∇logΘs individually, doing
                # `multiplicity` many adds instead of multiplying by it but
                # saving the loop in `multiplicity`. This may be less accurate,
                # though.
            end
        end
    end
end


struct ApproximationPolicy{T <: Real}
    ε::T
    n::Int

    ApproximationPolicy{T}(;
        ε,
        n = typemax(Int),
    ) where T <: Real = new(ε, n)
end

struct ExpectationPolicy{T}
    ω::Real
    burn::Int
    samples::Int
    ∇logℒ::ApproximationPolicy{T}
    trace_burn::Function
    trace_sample::Function

    ExpectationPolicy{T}(;
        ω,
        burn,
        samples,
        ∇logℒ,
        trace_burn = identity,
        trace_sample = identity,
    ) where T <: Real = new(ω, burn, samples, ∇logℒ, trace_burn, trace_sample)
end

struct GradientUpdatePolicy{T}
    α::T
    β₁::T
    β₂::T
    range::Tuple{T, T}

    GradientUpdatePolicy{T}(;
        α  = T(0.001),
        β₁ = T(0.9),
        β₂ = T(0.999),
        range = (nextfloat(zero(T)), prevfloat(one(T))),
    ) where T <: Real = new(α, β₁, β₂, range)
end

struct GradientDescentPolicy{T}
    steps::Int
    expectation::ExpectationPolicy{T}
    update::GradientUpdatePolicy{T}
    trace::Function

    GradientDescentPolicy{T}(;
        steps,
        expectation,
        update = GradientUpdatePolicy{T}(),
        trace = identity,
    ) where T <: Real = new(steps, expectation, update, trace)
end

function next_swap(
    σ::Permutation;
    P::AbstractMatrix{T},
    G::Graph{Index},
    ω::Real,
    trace::Function,
    entropy::AbstractRNG,
) where {T <: Real, Index <: Integer}
    for rejected in Iterators.countfrom(0)
        (u, v) = random_pair(entropy, G, ω)
        Δlogℒ = Δlogℒ_for_swap(G, P, σ, u, v)
        # TODO: maybe trace trace Δlogℒ even for rejected samples -- 2022-01-10
        if log(rand(entropy, T)) ≤ Δlogℒ
            trace((; u, v, rejected, Δlogℒ))
            return (u, v)
        end
    end
end

function approximate_𝔼σ_∇logℒ!(
    𝔼σ_∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
    σ::Permutation;
    P::AbstractMatrix{T},
    G::Graph{Index},
    policy::ExpectationPolicy{T},
    entropy::AbstractRNG,
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
) where {T <: Real, Index <: Integer}
    ω = policy.ω

    # Since P has likely changed, we need to re-burn σ and then re-estimate
    # ∇logℒ.
    for i in 1:policy.burn
        (u, v) = next_swap(σ; P, G, ω, trace = policy.trace_burn, entropy)
        swap!(σ, u, v)
        # TODO: Check for convergence. -- 2022-01-14
    end
    approximate_∇logℒ!(∇logℒ; P, σ, G, ε = policy.∇logℒ.ε, n = policy.∇logℒ.n)

    copy!.(𝔼σ_∇logℒ, ∇logℒ)
    for _ in 1:policy.samples
        (u, v) = next_swap(σ; P, G, ω, trace = policy.trace_sample, entropy)
        add_Δ∇logℒ_for_swap!(∇logℒ, G, P, σ, u, v)
        for (𝔼σ_∇logΘ, ∇logΘ) in zip(𝔼σ_∇logℒ, ∇logℒ)
            𝔼σ_∇logΘ .+= ∇logΘ
        end
        swap!(σ, u, v)
        # TODO: Trace swap. -- 2022-01-13
        # TODO: Check for convergence. -- 2022-01-14
    end

    for 𝔼σ_∇logΘ in 𝔼σ_∇logℒ
        𝔼σ_∇logΘ ./= policy.samples
    end
end

function ascend!(
    P::AbstractMatrix{T},
    σ::Permutation;
    G::Graph{Index},
    policy::GradientDescentPolicy{T},
    entropy::AbstractRNG,
) where {T <: Real, Index <: Integer}
    Θs = factors(P)  # aliased: Θs' and P's values are linked

    𝔼σ_∇logℒ = similar.(Θs)
    ∇logℒ = similar.(Θs)
    ΔΘs = similar.(Θs)
    ms = zero.(Θs)
    vs = zero.(Θs)

    α = policy.update.α
    β₁ = policy.update.β₁
    β₂ = policy.update.β₂
    for i in 1:policy.steps
        approximate_𝔼σ_∇logℒ!(
            𝔼σ_∇logℒ,
            σ;
            P,
            G,
            policy = policy.expectation,
            entropy,
            ∇logℒ,
        )
        for (Θ, ∇logΘ, m, v, ΔΘ) in zip(Θs, 𝔼σ_∇logℒ, ms, vs, ΔΘs)
            m .= β₁ .* m .+ (one(T) - β₁) .* ∇logΘ
            v .= β₂ .* v .+ (one(T) - β₂) .* ∇logΘ.^2
            debias = sqrt(one(T) - β₂^i) / (one(T) - β₁^i)
            ΔΘ .= α * debias .* m ./ (sqrt.(v) .+ 1e-8)
            Θ .+= ΔΘ  # updates P
            clamp!(Θ, policy.update.range...)
        end
        logℒ = approximate_logℒ(G, P, σ; ε = 1e-12, n = 10)
        policy.trace((; i, 𝔼σ_∇logℒ, ms, vs, ΔΘs, Θs, logℒ))
        # TODO: Trace logℒ (approximating it here) and potentially other
        #   convergence statistics. -- 2022-01-14
        # TODO: Check for convergence. -- 2022-01-14
        #   ^ e.g. Adam "SNR" m̂s ./ √v̂s
    end
end

function optimize!(
    P::AbstractMatrix{T};
    G::Graph{Index},
    policy::GradientDescentPolicy{T},
    entropy::AbstractRNG,
) where {T <: Real, Index <: Integer}
    σ = Permutation(shuffle(entropy, nodes(G)))  # TODO: Sort by degree.
    ascend!(P, σ; G, policy, entropy)
end

#=
TODO: implement public interface
TODO: split up files
TODO: test sampling permutations
TODO: clean up tests
TODO: documentation

TODO: check if Adam, RMSprop or similar are faster
TODO: check if we should use ComponentArrays
TODO: parallelization?
TODO: GPU?
=#

end # module
