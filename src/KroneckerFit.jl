module KroneckerFit

using Memoize

import Base.getindex
import Graphs
import Kronecker
import StaticGraphs

export Permutation, approximate_empty_logℒ, approximate_logℒ, Δlogℒ_for_swap


# Graph definition

const Graph = StaticGraphs.StaticDiGraph  # TODO get rid of StaticGraphs

outneighbors(G::Graph{Index, EdgeIndex}, u::Index) where {
    Index <: Integer, EdgeIndex <: Integer
} = Graphs.outneighbors(G, u)

inneighbors(G::Graph{Index, EdgeIndex}, u::Index) where {
    Index <: Integer, EdgeIndex <: Integer
} = Graphs.inneighbors(G, u)

edges(G::Graph{Index, EdgeIndex}) where {
    Index <: Integer, EdgeIndex <: Integer
} = Graphs.edges(G)

# Iterate indices of 2 rows and 2 columns of G's adjacency matrix, taking care
# not to double-enumerate 4 points of intersection.
function foreach_incident_edge(
    f::Function,
    G::Graph{Index},
    u::Index,
    v::Index,
) where Index <: Integer
    foreach(f, (u, w) for w in outneighbors(G, u))
    foreach(f, (w, u) for w in inneighbors(G, u) if w != u)
    foreach(f, (v, w) for w in outneighbors(G, v) if w != u)
    foreach(f, (w, v) for w in inneighbors(G, v) if w != v && w != u)
end


# Permutation definition

struct Permutation
    σ::AbstractVector{Int}
end

Base.getindex(σ::Permutation, (u, v)::Tuple{Index, Index}) where {
    Index <: Integer
} = CartesianIndex(σ.σ[u], σ.σ[v])


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
    result = zero(T)

    n ≤ 0 && return result

    Θs = factors(P)

    # If all factors' entries are 1, the series diverges to -∞, but extremely
    # slowly; no need to take the scenic route.
    all(all.(isone, Θs)) && return typemin(T)

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

function set_empty_∇logℒ!(
    ∇logℒ::Tuple{AbstractMatrix{T}},
    P::AbstractMatrix{T}
) where T <: Real
    ∇logℒ[1] .= -one(T) ./ (one(T) .- P)
end

function set_approximate_empty_∇logℒ!(
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
    P::AbstractMatrix{T};
    ε::T = T(NaN),
    n::Int = typemax(Int),
) where T <: Real
    set_empty_∇logℒ!(∇logℒ, P)
end

function set_approximate_empty_∇logℒ!(
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
    P::Kronecker.AbstractKroneckerProduct{T};
    ε::T = T(NaN),
    n::Int = typemax(Int),
) where T <: Real
    for ∇logΘ in ∇logℒ
        ∇logΘ .= zero(T)
    end

    n ≤ 0 && return

    Θs = factors(P)

    # Like for the approximate logℒ, if all factors' entries are 1, the series
    # diverges to -∞, but too slowly; take the shortcut.
    if all(all.(isone, Θs))
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

function set_approximate_∇logℒ!(
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
    G::Graph{Index},
    P::AbstractMatrix{T},
    σ::Permutation;
    ε::T = T(NaN),
    n::Int = typemax(Int),
) where {Index <: Integer, T <: Real}
    set_approximate_empty_∇logℒ!(∇logℒ, P; ε, n)
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

#=
TODO: Implement sampling permutations
TODO: Implement gradient descent
TODO: Implement public interface
TODO: Replace StaticGraphs
TODO: Split up files
TODO: Clean up tests
TODO: documentation

σ = initial permutation
logΘs = log.(Θs)
log_likelihood = approximate_empty_log_likelihood(...)
until logΘs convergence:
    k times:
        (u, v) = repeat
            (u, v) = sample swap proposal
            if accept (u, v): yield it
        add_∇logΘs_change_for_swap!(Δ∇logΘs, Θs)
        ∇logΘs += Δ∇logΘs ./ k
    σ = swap last (u, v) in σ
    logΘs += λ .* ∇logΘs
    Θs = exp.(logΘs)

=#

end # module
