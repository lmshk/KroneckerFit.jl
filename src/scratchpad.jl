
function benchmark_swap_log_likelihood_change(G, P, σ, u, v)
    σ′ = copy(σ)
    σ′[u], σ′[v] = σ′[v], σ′[u]
    display(@benchmark swap_log_likelikood_change($G, $P, $σ, $u, $v))
    print(); print();
    display(@benchmark swap_log_likelikood_change2($G, $P, $σ, $u, $v))
    print(); print();
    display(@benchmark swap_log_likelikood_change($G, $P, $σ, $u, $v))
end


#=     benchmark_swap_log_likelihood_change(
        graph(Kronecker.kronecker([
            0 1 1 0 1 0 1 0
            0 1 0 1 0 0 0 0
            1 1 1 1 0 1 0 1
            0 0 0 1 1 1 0 0
            1 0 0 1 0 1 1 0
            1 0 1 1 0 0 1 0
            0 0 0 0 0 0 0 0
            1 1 1 1 1 1 1 1
        ], 3)),
        Kronecker.kronecker(Θ, 9),
        collect(0x1:UInt16(2^9)),
        UInt16(0x66),
        UInt16(0x101)
    ) =#


function swap_log_likelikood_change2(
    G::Graph{Index},
    P::AbstractMatrix{Probability},
    σ::AbstractVector{Index},
    u::Index,
    v::Index,
) where {Index <: Integer, Probability <: Real}
    # TODO: handle 0s and 1s in P?

    result = 0.0

    link_contribution(u, v) =
        log(P[σ[u], σ[v]]) - log(one(Probability) - P[σ[u], σ[v]])

    for w in outneighbors(G, u)  # (u -> *)
        change =
            if w == u  # (u -> u)
                -link_contribution(u, u) + link_contribution(v, v)
            elseif w == v  # (u -> v)
                -link_contribution(u, v) + link_contribution(v, u)
            else  # normal case
                -link_contribution(u, w) + link_contribution(v, w)
            end
        result += change
    end

    for w in inneighbors(G, u)  # (* -> u)
        change =
            if w == u  # (u -> u)
                0  # already counted above
            elseif w == v  # (v -> u)
                -link_contribution(v, u) + link_contribution(u, v)
            else  # normal case
                -link_contribution(w, u) + link_contribution(w, v)
            end
            result += change
    end

    for w in outneighbors(G, v)  # (v -> *)
        change =
            if w == v  # (v -> v)
                -link_contribution(v, v) + link_contribution(u, u)
            elseif w == u  # (v -> u)
                0  # already counted above
            else  # normal case
                -link_contribution(v, w) + link_contribution(u, w)
            end
        result += change
    end

    for w in inneighbors(G, v)  # (* -> v)
        change =
            if w == v  # (v -> v)
                0  # already counted above
            elseif w == u  # (u -> v)
                0  # already counted above
            else  # normal case
                -link_contribution(w, v) + link_contribution(w, u)
            end
        result += change
    end

    result
end