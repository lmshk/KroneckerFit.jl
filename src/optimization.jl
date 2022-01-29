using Random

Base.@kwdef struct GradientExpectationPolicy{T}
    ω::Real = 0.6
    logℒ::TaylorApproximationPolicy{T} =
        TaylorApproximationPolicy{T}(ε = 1e-12, n = 100)
    ∇logℒ::TaylorApproximationPolicy{T} =
        TaylorApproximationPolicy{T}(ε = 1e-12, n = 100)
end

Base.@kwdef struct GradientUpdatePolicy{T}
    α::T = T(0.01)
    β₁::T = T(0.9)
    β₂::T = T(0.999)
    range::Tuple{T, T} = (nextfloat(zero(T)), prevfloat(one(T)))
end

Base.@kwdef struct GradientAscentPolicy{T}
    σ_convergence::ConvergencePolicy{T} = ConvergencePolicy{T}(
        α = 0.05,
        initial_target = φ^10 / sqrt(5),
        maximal_burn = 2178220,
    )
    ∇logℒ_expectation::GradientExpectationPolicy{T} =
        GradientExpectationPolicy{T}()
    ∇logℒ_update::GradientUpdatePolicy{T} = GradientUpdatePolicy{T}()
    Θs_convergence::ConvergencePolicy{T} = ConvergencePolicy{T}(α = 0.05)
    logℒ::TaylorApproximationPolicy{T} =
        TaylorApproximationPolicy{T}(ε = 1e-12, n = 100)
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
    while true
        swap = next_swap(σ; P, G, policy.ω, entropy)
        @debug("burned swap", _id = :swap, phase = :burn, swap...)
        swap!(σ, swap.u, swap.v)
        update!(convergence, swap.Δlogℒ)

        if is_estimate_complete(convergence)
            @debug("estimated 𝔼σ(Δlogℒ)", _id = :chain_estimate,
                summarize(convergence)...
            )
        end

        if is_converged(convergence) || should_give_up(convergence)
            return convergence
        end
    end
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
        ΔΘ .+= Θ
        clamp!(ΔΘ, policy.range...)
        ΔΘ .-= Θ
        Θ .+= ΔΘ
        clamp!(Θ, policy.range...)
        # TODO: check if ^ is necessary. -- 2022-01-29
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

    total = 0

    σ_convergence = converge!(σ; P, G, policy = policy.σ_convergence, entropy)
    @debug("converged σ", _id = :chain_convergence,
        summarize(σ_convergence)...
    )
    total += σ_convergence.samples + σ_convergence.burned

    𝔼σ_logℒ = approximate_logℒ(G, P, σ; policy = policy.logℒ)
    convergence = Tuple(
        [Convergence(policy.Θs_convergence) for _ in Θ]
        for Θ in Θs
    )
    for i in Iterators.countfrom(1)
        σ_convergence = converge!(
            σ;
            P,
            G,
            policy = policy.σ_convergence,
            entropy
        )
        @debug("reconverged σ", _id = :chain_convergence,
            summarize(σ_convergence)...
        )
        total += σ_convergence.samples + σ_convergence.burned

        𝔼σ_logℒ = approximate_𝔼σ_∇logℒ!(
            𝔼σ_∇logℒ_Θs,
            σ;
            P,
            G,
            σ_convergence.samples,
            policy = policy.∇logℒ_expectation,
            entropy,
            ∇logℒ_Θs,
        )
        total += σ_convergence.samples

        update_Θs!(
            Θs,
            𝔼σ_∇logℒ_Θs;
            i,
            policy = policy.∇logℒ_update,
            ms,
            vs,
            ΔΘs
        )
        @debug("updated Θs", _id = :update,
            i, 𝔼σ_logℒ, Θs, 𝔼σ_∇logℒ_Θs, ms, vs, ΔΘs
        )

        update!(convergence, ΔΘs)
        if is_estimate_complete(convergence)
            @debug("estimated 𝔼σ(ΔΘ))", _id = :estimate,
                summarize(convergence)...
            )
        end
        if is_converged(convergence) || should_give_up(convergence)
            break
        end
    end

    # TODO: Post-optimize σ. -- 2022-01-21
    
    logℒ = approximate_logℒ(G, P, σ; policy = policy.logℒ)
    (; summarize(convergence)..., Θs, σ, logℒ, 𝔼σ_logℒ, total)
end