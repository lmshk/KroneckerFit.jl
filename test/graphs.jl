#= @testset "graphs" begin
#=     G = KroneckerFit.Graph(Bool[
        0 1 1 0 1 0 1 0
        0 1 0 1 0 0 0 0
        1 1 1 1 0 1 0 1
        0 0 0 1 1 1 0 0
        1 0 0 1 0 1 1 0
        1 0 1 1 0 0 1 0
        0 0 0 0 0 0 0 0
        1 1 1 1 1 1 1 1
    ]) =#

    G = KroneckerFit.Graph(Tuple{Int8, Int8}[(1, 2), (2, 3), (3, 1), (5, 5)])

    @warn "XXX" [KroneckerFit.random_pair(Random.GLOBAL_RNG, G, .1337) for _ in 1:100]

    @test false
end =#