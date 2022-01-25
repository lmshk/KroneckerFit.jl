module KroneckerFit

import Random

import Kronecker

export fit!

include("graphs.jl")
include("permutations.jl")
include("convergence.jl")
include("kronecker.jl")
include("objective.jl")
include("optimization.jl")

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

end # module
