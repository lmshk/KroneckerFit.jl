import Base.getindex

struct Permutation
    σ::AbstractVector{Int}
end

Base.getindex(σ::Permutation, (u, v)::Tuple{Index, Index}) where {
    Index <: Integer
} = CartesianIndex(σ.σ[u], σ.σ[v])

function swap!(σ::Permutation, u::Index, v::Index) where Index <: Integer
    σ.σ[u], σ.σ[v] = σ.σ[v], σ.σ[u]
end