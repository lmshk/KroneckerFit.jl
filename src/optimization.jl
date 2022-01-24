using Random

struct TaylorApproximationPolicy{T <: Real}
    ε::T
    n::Int

    TaylorApproximationPolicy{T}(;
        ε = T(NaN),
        n = typemax(Int),
    ) where T <: Real = new(ε, n)
end

Base.@kwdef struct ConvergencePolicy{T <: Real}
    ω::Float64 = 0.6
    φ::Float64 = (1 + sqrt(5)) / 2
    initial_target::Float64 = φ^10 / sqrt(5)
    α::T = 0.001
    maximal_burn::Int = 2178220
end

struct GradientExpectationPolicy{T}
    ω::Real
    logℒ::TaylorApproximationPolicy{T}
    ∇logℒ::TaylorApproximationPolicy{T}
    # TODO: Maybe add convergence criteria. -- 2022-01-22

    GradientExpectationPolicy{T}(;
        ω = 0.6,
        logℒ = TaylorApproximationPolicy{T}(ε = 1e-12, n = 100),
        ∇logℒ = TaylorApproximationPolicy{T}(ε = 1e-12, n = 100),
    ) where T <: Real = new(ω, logℒ, ∇logℒ)
end

struct GradientUpdatePolicy{T}
    α::T
    β₁::T
    β₂::T
    range::Tuple{T, T}

    GradientUpdatePolicy{T}(;
        α  = T(0.01),
        β₁ = T(0.9),
        β₂ = T(0.999),
        range = (nextfloat(zero(T)), prevfloat(one(T))),
    ) where T <: Real = new(α, β₁, β₂, range)
end

struct GradientAscentPolicy{T}
    maximum_steps::Int
    σ_convergence::ConvergencePolicy{T}
    ∇logℒ_expectation::GradientExpectationPolicy{T}
    ∇logℒ_update::GradientUpdatePolicy{T}
    ε::T
    window::Int
    logℒ::TaylorApproximationPolicy{T}

    GradientAscentPolicy{T}(;
        maximum_steps = typemax(Int),
        σ_convergence = ConvergencePolicy{T}(),
        ∇logℒ_expectation = GradientExpectationPolicy{T}(),
        ∇logℒ_update = GradientUpdatePolicy{T}(),
        ε = 1e-5,
        window = 5,
        logℒ = TaylorApproximationPolicy{T}(ε = 1e-12, n = 100),
    ) where T <: Real = new(
        maximum_steps,
        σ_convergence,
        ∇logℒ_expectation,
        ∇logℒ_update,
        ε,
        window,
        logℒ,
    )
end

Base.@kwdef mutable struct Convergence{T <: Real}
    target::Float64
    burned::Int = 0
    estimates::Int = 0
    samples::Int = 0
    μ::T = zero(T)
    Σ𝕍::T = zero(T)
end

function Convergence(policy::ConvergencePolicy{T}) where T <: Real
    policy.initial_target ≥ 2.0 || throw(DomainError(policy.initial_target))
    Convergence{T}(target = policy.initial_target)
end

is_estimate_complete(convergence::Convergence{T}) where T <: Real =
    convergence.samples ≥ round(Int, convergence.target)

function update!(
    convergence::Convergence{T},
    x::T;
    policy::ConvergencePolicy{T},
) where T <: Real
    if is_estimate_complete(convergence)
        convergence.target *= policy.φ
        convergence.burned += convergence.samples
        convergence.samples = 0
        convergence.estimates += 1
        convergence.μ = zero(T)
        convergence.Σ𝕍 = zero(T)
    end

    convergence.samples += 1
    δ = x - convergence.μ
    convergence.μ += δ / convergence.samples
    convergence.Σ𝕍 += δ * (x - convergence.μ)

    convergence
end

function is_converged(
    convergence::Convergence{T};
    policy::ConvergencePolicy{T},
) where T <: Real
    α = policy.α
    n = convergence.samples
    μ = convergence.μ
    Σ𝕍 = convergence.Σ𝕍

    is_estimate_complete(convergence) && μ * μ ≤ α * Σ𝕍 / (n - 1)
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

function next_swap(
    σ::Permutation;
    P::AbstractMatrix{T},
    G::Graph{Index},
    ω::Real,
    entropy::AbstractRNG,
) where {T <: Real, Index <: Integer}
    for rejected in Iterators.countfrom(0)
        (u, v) = random_pair(entropy, G, ω)
        Δlogℒ = Δlogℒ_for_swap(G, P, σ, u, v)
        if log(rand(entropy, T)) ≤ Δlogℒ
            return (; u, v, rejected, Δlogℒ)
        end
    end
end

function converge!(
    σ::Permutation;
    P::AbstractMatrix{T},
    G::Graph{Index},
    policy::ConvergencePolicy{T},
    entropy::AbstractRNG,
) where {T <: Real, Index <: Integer}
    convergence = Convergence(policy)

    while convergence.burned ≤ policy.maximal_burn
        swap = next_swap(σ; P, G, policy.ω, entropy)
        @debug("burned swap", _id = :swap, phase = :burn, swap...)
        swap!(σ, swap.u, swap.v)
        update!(convergence, swap.Δlogℒ; policy)

        if is_estimate_complete(convergence)
            @debug("estimated 𝔼σ_Δlogℒ", _id = :estimate,
                μ = convergence.μ,
                𝕍 = convergence.Σ𝕍 / (convergence.samples - 1),
                samples = convergence.samples,
                burned = convergence.burned,
            )
        end

        if is_converged(convergence; policy)
            return (;
                converged = true,
                estimates = convergence.estimates,
                μ = convergence.μ,
                𝕍 = convergence.Σ𝕍 / (convergence.samples - 1),
                samples = convergence.samples,
                burned = convergence.burned,
            )
        end
    end

    return (;
        converged = false,
        estimates = convergence.estimates,
        burned = convergence.burned,
        μ = convergence.μ,
        𝕍 = convergence.Σ𝕍 / (convergence.samples - 1),
        samples = convergence.samples
    )
end

function approximate_𝔼σ_∇logℒ!(
    𝔼σ_∇logℒ_Θs::Tuple{Vararg{AbstractMatrix{T}}},
    σ::Permutation;
    P::AbstractMatrix{T},
    G::Graph{Index},
    samples::Int,
    policy::GradientExpectationPolicy{T},
    entropy::AbstractRNG,
    ∇logℒ_Θs::Tuple{Vararg{AbstractMatrix{T}}},
) where {T <: Real, Index <: Integer}
    logℒ = approximate_logℒ(G, P, σ; policy = policy.logℒ)
    𝔼σ_logℒ = zero(T)
    approximate_∇logℒ!(∇logℒ_Θs; P, σ, G, policy = policy.∇logℒ)
    fill!.(𝔼σ_∇logℒ_Θs, zero(T))
    for _ in 1:samples
        swap = next_swap(σ; P, G, policy.ω, entropy)
        logℒ += Δlogℒ_for_swap(G, P, σ, swap.u, swap.v)
        𝔼σ_logℒ += logℒ
        add_Δ∇logℒ_for_swap!(∇logℒ_Θs, G, P, σ, swap.u, swap.v)
        for (𝔼σ_∇logℒ_Θ, ∇logℒ_Θ) in zip(𝔼σ_∇logℒ_Θs, ∇logℒ_Θs)
            𝔼σ_∇logℒ_Θ .+= ∇logℒ_Θ
        end
        @debug("sampled swap", _id = :swap, phase = :sample, swap...)
        swap!(σ, swap.u, swap.v)
    end
    
    𝔼σ_logℒ /= samples
    for 𝔼σ_∇logℒ_Θ in 𝔼σ_∇logℒ_Θs
        𝔼σ_∇logℒ_Θ ./= samples
    end

    # TODO: Maybe continue sampling if the gradient (as opposed to the logℒ)
    # has not converged yet. -- 2022-01-22

    𝔼σ_logℒ
end

function update_Θs!(
    Θs::Tuple{Vararg{AbstractMatrix{T}}},
    ∇logℒ_Θs::Tuple{Vararg{AbstractMatrix{T}}};
    i::Int,
    policy::GradientUpdatePolicy{T},
    ms::Tuple{Vararg{AbstractMatrix{T}}},
    vs::Tuple{Vararg{AbstractMatrix{T}}},
    ΔΘs::Tuple{Vararg{AbstractMatrix{T}}},
) where T <: Real
    α, β₁, β₂ = policy.α, policy.β₁, policy.β₂
    for (Θ, ∇logΘ, m, v, ΔΘ) in zip(Θs, ∇logℒ_Θs, ms, vs, ΔΘs)
        m .= β₁ .* m .+ (one(T) - β₁) .* ∇logΘ
        v .= β₂ .* v .+ (one(T) - β₂) .* ∇logΘ.^2
        debias = sqrt(one(T) - β₂^i) / (one(T) - β₁^i)
        ΔΘ .= α * debias .* m ./ (sqrt.(v) .+ 1e-8)
        Θ .+= ΔΘ
        clamp!(Θ, policy.range...)
    end
end

function ascend!(
    P::AbstractMatrix{T},
    σ::Permutation;
    G::Graph{Index},
    policy::GradientAscentPolicy{T},
    entropy::AbstractRNG,
) where {T <: Real, Index <: Integer}
    Θs = factors(P)  # aliased: Θs' and P's values are linked

    # Pre-allocate buffers for the gradient updates.
    𝔼σ_∇logℒ_Θs = similar.(Θs)
    ∇logℒ_Θs = similar.(Θs)
    ms = zero.(Θs)
    vs = zero.(Θs)
    ΔΘs = similar.(Θs)

    σ_convergence = converge!(σ; P, G, policy = policy.σ_convergence, entropy)
    @debug("converged", _id = :convergence, σ_convergence...)

    𝔼σ_logℒ = nextfloat(typemin(T))
    converged = 0
    for i in 1:policy.maximum_steps
        σ_convergence = converge!(σ; P, G, policy = policy.σ_convergence, entropy)
        @debug("reconverged", _id = :convergence, σ_convergence...)

        new_𝔼σ_logℒ = approximate_𝔼σ_∇logℒ!(
            𝔼σ_∇logℒ_Θs,
            σ;
            P,
            G,
            σ_convergence.samples,
            policy = policy.∇logℒ_expectation,
            entropy,
            ∇logℒ_Θs,
        )

        # TODO: Improve convergence detection. -- 2022-01-22
        if isapprox(𝔼σ_logℒ, new_𝔼σ_logℒ; rtol = policy.ε)
            converged += 1
            if converged ≥ policy.window
                break
            end
        else
            converged = 0
        end

        𝔼σ_logℒ = new_𝔼σ_logℒ
        update_Θs!(Θs, 𝔼σ_∇logℒ_Θs; i, policy = policy.∇logℒ_update, ms, vs, ΔΘs)
        @debug("gradient updated", _id = :update,
            i, 𝔼σ_logℒ, Θs, 𝔼σ_∇logℒ_Θs, ms, vs, ΔΘs
        )
    end

    # TODO: Post-optimize σ. -- 2022-01-21
    
    logℒ = approximate_logℒ(G, P, σ; policy = policy.logℒ)

    (; converged = converged ≥ policy.window, Θs, σ, logℒ, 𝔼σ_logℒ)
end

function optimize!(
    P::AbstractMatrix{T};
    G::Graph{Index},
    policy::GradientAscentPolicy{T},
    entropy::AbstractRNG,
) where {T <: Real, Index <: Integer}
    σ = Permutation(shuffle(entropy, nodes(G)))
    ascend!(P, σ; G, policy, entropy)
end

function fit!(
    Θ::Matrix{T};
    G::Graph{Index},
    σ::Union{Permutation, Nothing} = nothing,
    policy::GradientAscentPolicy{T} = GradientAscentPolicy{T}(),
    entropy::AbstractRNG = Random.GLOBAL_RNG,
) where {T <: Real, Index <: Integer}
    size(Θ, 1) == size(Θ, 2) || throw(DomainError(Θ))

    n₀ = size(Θ, 1)
    n = length(nodes(G))
    2 ≤ n₀ ≤ n || throw(DomainError(Θ))

    k = ceil(Int, log(n₀, n))
    n₀^k == n || throw(DomainError(G))

    P = k ≤ 1 ? Θ : Kronecker.kronecker(Θ, k)

    if isnothing(σ)
        σ = Permutation(shuffle(entropy, nodes(G)))
    end

    ascend!(P, σ; G, policy, entropy)
end

function fit!(
    Θ::Matrix{Missing} = [missing missing; missing missing];
    arguments...
)
    fit!(
        collect(
            reshape(
                range(1.0, 0.0, length = length(Θ) + 2)[2 : end - 1],
                size(Θ)...
            )'
        );
        arguments...
    )
end