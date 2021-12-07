@testset "StaticGraphs" begin
    # `StaticGraphs.StaticDiGraph` allocates an empty graph when passed a fully
    # disconnected `Graphs.SimpleDiGraph`.
    G = Graphs.SimpleDiGraph(zeros(1, 1))
    @test Graphs.nv(G) == 1
    @test Graphs.nv(StaticGraphs.StaticDiGraph(G)) == 0

    # The problem is that it then accesses its data structures out-of-bounds
    # when e.g. calling `outneighbors`. This only hits during testing because
    # then julia is run with `--check-bounds=yes` which ignores the erroneous
    # `@inbounds` in `StaticGraphs._bvrange`, but it could cause undefined
    # behavior in normal operation.
    @test_throws BoundsError Graphs.outneighbors(StaticGraphs.StaticDiGraph(G), 0x1)

    # `StaticGraphs.StaticDiGraph` may use distinct types for vertex and edge
    # indices; this causes `Graphs.adjacency_matrix` to fail, because the
    # storage vectors are passed directly to `SparseArrays.SparseMatrixCSC`,
    # which only supports a single index type.
    @test_throws MethodError ones(64, 64) |> Graphs.SimpleDiGraph |> StaticGraphs.StaticDiGraph |> Graphs.adjacency_matrix
end