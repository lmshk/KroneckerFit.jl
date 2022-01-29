using SparseArrays

using ArgParse
using DataFrames
using Plots
using StatsBase
using KrylovKit

import Arrow

settings = @add_arg_table! ArgParseSettings() begin
    "--analyses", "-a"
        nargs = '*'
        arg_type = Symbol

    "--nodes", "-n"
        required = true
        arg_type = Int

    "--sink"
        default = "./"

    "reference"
        required = true

    "others"
        nargs = '*'
end

load(path) = path |> Arrow.Table |> DataFrame
adjacency(n, edges) = sparse(
    edges.source,
    edges.target,
    fill(true, length(edges.source)),
    n,
    n
)

node_degrees(A) = vec(sum(A, dims = 1)), vec(sum(A, dims = 2))

function counts(xs)
    result = Dict()
    for x in xs
        x == 0 && continue
        if !(x in keys(result))
            result[x] = 0
        end
        result[x] += 1
    end
    zipped = sort(collect(zip(keys(result), values(result))))
    (first.(zipped), last.(zipped))
end

function scree(A)
    result, _, _, convergence = svdsolve(A, 100, krylovdim = 100)
    result[1:convergence.converged]
end

function plot_degree_distribution(paths, counts)
    reference, others... = paths
    result = plot(
        counts[reference]...,
        yaxis = :log10,
        xaxis = :log10,
        markershape = :circle,
        markersize = 2,
        label = basename(reference),
        xlabel = "node degree",
        ylabel = "count",
    )
    for (i, path) in enumerate(others)
        plot!(
            counts[path]...,
            color = :lightblue,
            label = "sample $i",
            markershape = :circle,
            markersize = 1,
        )
    end
    result
end

function plot_scree(paths, screes)
    reference, samples... = paths
    result = plot(
        screes[reference],
        xaxis = :log10,
        yaxis = :log10,
        legend = false,
        title = "Scree",
        xlabel = "rank",
        ylabel = "singular value"
    )
    for sample in samples
        plot!(screes[sample], color = :lightblue)
    end
    result
end

function main(; sink, nodes, analyses, reference, others)
    paths = [reference, others...]

    indegree_counts = Dict()
    outdegree_counts = Dict()
    screes = Dict()
    for (path, A) in zip(paths, adjacency.(nodes, load.(paths)))
        @info path A
        if :degrees in analyses
            indegrees, outdegrees = node_degrees(A)
            indegree_counts[path] = counts(indegrees)
            outdegree_counts[path] = counts(outdegrees)
        end
        if :scree in analyses
            screes[path] = scree(A)
        end
    end

    mkpath(sink)

    if :degrees in analyses
        savefig(
            plot_degree_distribution(paths, indegree_counts),
            joinpath(sink, "$(basename(reference)).indegrees.svg")
        )
        savefig(
            plot_degree_distribution(paths, outdegree_counts),
            joinpath(sink, "$(basename(reference)).outdegrees.svg")
        )
    end

    if :scree in analyses
        savefig(
            plot_scree(paths, screes),
            joinpath(sink, "$(basename(reference)).scree.svg")
        )
    end
end

main(; parse_args(settings, as_symbols = true)...)