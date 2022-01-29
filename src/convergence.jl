const φ = (1 + sqrt(5)) / 2

Base.@kwdef struct ConvergencePolicy{T <: Real}
    α::T
    ω::Float64 = 0.6
    growth::Float64 = φ
    initial_target::Float64 = φ^7 / sqrt(5)
    maximal_burn::Int = typemax(Int)
end

Base.@kwdef mutable struct Convergence{T <: Real}
    policy::ConvergencePolicy{T}
    target::Float64
    burned::Int = 0
    estimate::Int = 1
    samples::Int = 0
    μ::T = zero(T)
    Σ𝕍::T = zero(T)
end

function Convergence(policy::ConvergencePolicy{T}) where T <: Real
    policy.initial_target ≥ 1.5 || throw(DomainError(policy.initial_target))
    Convergence{T}(; policy, target = policy.initial_target)
end

function summarize(convergence::Convergence{T}) where T <: Real
    𝕍 = convergence.Σ𝕍 / (convergence.samples - 1)
    (;
        converged = is_converged(convergence),
        convergence.estimate,
        convergence.μ,
        𝕍,
        z = convergence.μ / sqrt(𝕍),
        convergence.samples,
        convergence.burned,
    )
end

function summarize(
    convergence::Tuple{Vararg{AbstractMatrix{Convergence{T}}}}
) where T <: Real
    consolidated = Tuple([summarize(c2) for c2 in c1] for c1 in convergence)
    (;
        converged = Tuple([c2.converged for c2 in c1] for c1 in consolidated),
        estimate = last(last(consolidated)).estimate,
        μ = Tuple([c2.μ for c2 in c1] for c1 in consolidated),
        𝕍 = Tuple([c2.𝕍 for c2 in c1] for c1 in consolidated),
        z = Tuple([c2.z for c2 in c1] for c1 in consolidated),
        samples = last(last(consolidated)).samples,
        burned = last(last(consolidated)).burned,
    )
end

is_estimate_complete(convergence::Convergence{T}) where T <: Real =
    convergence.samples ≥ round(Int, convergence.target)

is_estimate_complete(
    convergence::Tuple{Vararg{AbstractMatrix{Convergence{T}}}}
) where T <: Real = convergence |> last |> last |> is_estimate_complete

should_give_up(convergence::Convergence{T}) where T <: Real =
    convergence.burned + convergence.samples ≥ convergence.policy.maximal_burn

should_give_up(
    convergence::Tuple{Vararg{AbstractMatrix{Convergence{T}}}}
) where T <: Real = convergence |> last |> last |> should_give_up

function is_converged(convergence::Convergence{T}) where T <: Real
    α = convergence.policy.α
    n = convergence.samples
    μ = convergence.μ
    Σ𝕍 = convergence.Σ𝕍

    is_estimate_complete(convergence) && μ * μ ≤ α * α * Σ𝕍 / (n - 1)
end

function is_converged(
    convergence::Tuple{Vararg{AbstractMatrix{Convergence{T}}}}
) where T <: Real
    for c1 in convergence
        for c2 in c1
            is_converged(c2) || return false
        end
    end
    
    true
end

function update!(convergence::Convergence{T}, x::T) where T <: Real
    if is_estimate_complete(convergence)
        convergence.target *= convergence.policy.growth
        convergence.burned += convergence.samples
        convergence.samples = 0
        convergence.estimate += 1
        convergence.μ = zero(T)
        convergence.Σ𝕍 = zero(T)
    end

    convergence.samples += 1
    δ = x - convergence.μ
    convergence.μ += δ / convergence.samples
    convergence.Σ𝕍 += δ * (x - convergence.μ)

    convergence
end

function update!(
    convergence::Tuple{Vararg{AbstractMatrix{Convergence{T}}}},
    x::Tuple{Vararg{AbstractMatrix{T}}}
) where T <: Real
    for (convergences, xs) in zip(convergence, x)
        update!.(convergences, xs)
    end
end