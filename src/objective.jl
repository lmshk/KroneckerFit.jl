Base.@kwdef struct TaylorApproximationPolicy{T <: Real}
    ε::T = T(NaN)
    n::Int = typemax(Int)
end

empty_logℒ(P::AbstractMatrix{T}) where T <: Real =
    sum(log.(one(T) .- P))

function approximate_empty_logℒ(
    P::AbstractMatrix{T};
    policy::TaylorApproximationPolicy{T},
) where T <: Real
    empty_logℒ(P)
end

function approximate_empty_logℒ(
    P::Kronecker.AbstractKroneckerProduct{T};
    policy::TaylorApproximationPolicy{T},
) where T <: Real
    # TODO: Test behavior with bad values, especially <0, >1 and =1. -- 2022-01-16

    Θs = factors(P)
    
    # For non-probability inputs, logℒ is not defined.
    all(all(zero(T) .<= Θ .<= one(T)) for Θ in Θs) || throw(DomainError(Θs))

    result = zero(T)
    policy.n ≤ 0 && return result
    
    # If all factors contain at least one 1, the series diverges to -∞, but
    # extremely slowly; no need to take the scenic route.
    all(any(isone.(Θ)) for Θ in Θs) && return typemin(T)
    
    k = multiplicity(P)
    i = 0
    Θs′ = copy.(Θs)
    while true
        i += 1
        Δ = prod(sum.(Θs′))^k / i
        Δ ≥ policy.ε || break
        result -= Δ
        i < policy.n || break
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
    policy::TaylorApproximationPolicy{T},
) where {Index <: Integer, T <: Real}
    approximate_empty_logℒ(P; policy) + sum(
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

function empty_∇logℒ!(
    ∇logℒ::Tuple{AbstractMatrix{T}},
    P::AbstractMatrix{T},
) where T <: Real
    ∇logℒ[1] .= -one(T) ./ (one(T) .- P)
end

function approximate_empty_∇logℒ!(
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
    P::AbstractMatrix{T};
    policy::TaylorApproximationPolicy{T},
) where T <: Real
    empty_∇logℒ!(∇logℒ, P)
end

function approximate_empty_∇logℒ!(
    ∇logℒ::Tuple{Vararg{AbstractMatrix{T}}},
    P::Kronecker.AbstractKroneckerProduct{T};
    policy::TaylorApproximationPolicy{T},
) where T <: Real
    Θs = factors(P)
    
    # For non-probability inputs, ∇logℒ is not defined.
    all(all(zero(T) .<= Θ .<= one(T)) for Θ in Θs) || throw(DomainError(Θs))

    for ∇logΘ in ∇logℒ
        ∇logΘ .= zero(T)
    end
    policy.n ≤ 0 && return
    
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
                if δ ≥ policy.ε
                    ∇logΘ[x] -= δ
                    changing = true
                end
            end
        end
        changing || break
        i += 1
        # i < n || @warn("ill-conditioned", i, n, Δs, ∇logℒ)
        # TODO: Do we need to warn here^? -- 2022-01-17
        i < policy.n || break
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
    policy::TaylorApproximationPolicy{T},
) where {Index <: Integer, T <: Real}
    approximate_empty_∇logℒ!(∇logℒ, P; policy)
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