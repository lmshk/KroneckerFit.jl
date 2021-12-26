using SparseArrays

import Distributions

struct Graph{Index <: Integer}
    edges::Vector{Tuple{Index, Index}}
    A::SparseMatrixCSC{Bool, Index}
    Aᵀ::SparseMatrixCSC{Bool, Index}
end

Graph(A::SparseMatrixCSC{Bool, Index}) where Index <: Integer = Graph{Index}(
    collect(zip(findnz(A)[1], findnz(A)[2])),
    A,
    sparse(A')
)

Graph(A::AbstractMatrix{Bool}) = Graph(sparse(A))
Graph(edges::AbstractVector{Tuple{Index, Index}}) where Index <: Integer =
    Graph(
        sparse(
            first.(edges),
            last.(edges),
            [true for _ in edges]
        )
    )

nodes(G::Graph{Index}) where Index <: Integer = Base.OneTo{Index}(size(G.A)[1])
edges(G::Graph{Index}) where Index <: Integer = G.edges

# Iterate indices of 2 rows and 2 columns of G's adjacency matrix, taking care
# not to double-enumerate 4 points of intersection.
function foreach_incident_edge(
    f::Function,
    G::Graph{Index},
    u::Index,
    v::Index,
) where Index <: Integer
#=     foreach(f, (u, w) for w in outneighbors(G, u))
    foreach(f, (v, w) for w in outneighbors(G, v) if w != u)
    foreach(f, (w, u) for w in inneighbors(G, u) if w != u)
    foreach(f, (w, v) for w in inneighbors(G, v) if w != v && w != u) =#

    foreach(f, (u, rowvals(G.Aᵀ)[i]) for i in nzrange(G.Aᵀ, u))
    foreach(f, (v, rowvals(G.Aᵀ)[j]) for j in nzrange(G.Aᵀ, v) if rowvals(G.Aᵀ)[j] != u)
    foreach(f, (rowvals(G.A)[i], u) for i in nzrange(G.A, u) if rowvals(G.A)[i] != u)
    foreach(f, (rowvals(G.A)[j], v) for j in nzrange(G.A, v) if rowvals(G.A)[j] != u && rowvals(G.A)[j] != v)
end

function random_pair(
    entropy::AbstractRNG,
    G::Graph{Index},
    ω::Real,
) where Index <: Integer
    if rand(entropy, Distributions.Bernoulli(ω))
        (rand(entropy, nodes(G)), rand(entropy, nodes(G)))
    else
        rand(entropy, edges(G))
    end
end