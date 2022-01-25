using Memoize

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