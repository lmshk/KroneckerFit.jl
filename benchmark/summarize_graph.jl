using SparseArrays
using LinearAlgebra

using ArgParse
using DataFrames
using Plots
using StatsBase

import Arrow

settings = @add_arg_table! ArgParseSettings() begin
    "--sink"
        default = "./"

    "--nodes", "-n"
        required = true
        arg_type = Int

    "reference"
        required = true

    "samples"
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

function log_histograms(degrees)
    bin_edges = exp.(.0:.5:log(maximum(maximum.(values(degrees)))))
    result = Dict()
    for (path, d) in degrees
        h = normalize(fit(Histogram, d, bin_edges, closed = :right), mode = :density)
        result[path] = h.weights
    end
    (bin_edges[1 : end - 1], result)
end

scree(A) = svdvals(collect(A))

function plot_degree_distribution(paths, bins, degree_histograms)
    reference, samples... = paths
    result = plot(bins, degree_histograms[reference], legend = false, xaxis = :log10, yaxis = :log10)
    for sample in samples
        plot!(bins, degree_histograms[sample], color = :lightblue)
    end
    result
end

function plot_scree(paths, screes)
    reference, samples... = paths
    result = plot(
        screes[reference][1:100],
        xaxis = :log10,
        yaxis = :log10,
        legend = false,
        title = "Scree",
        xlabel = "rank",
        ylabel = "singular value"
    )
    for sample in samples
        plot!(screes[sample][1:100], color = :lightblue)
    end
    result
end

function main(; sink, nodes, reference, samples)
    paths = [reference, samples...]

    indegrees = Dict()
    outdegrees = Dict()
    screes = Dict()
    for (path, A) in zip(paths, adjacency.(nodes, load.(paths)))
        @info path A
        indegrees[path], outdegrees[path] = node_degrees(A)
        screes[path] = scree(A)
    end
    inbins, indegree_histograms = log_histograms(indegrees)
    outbins, outdegree_histograms = log_histograms(outdegrees)

    savefig(
        plot_degree_distribution(paths, inbins, indegree_histograms),
        "$(sink)_indegrees_histogram.png"
    )
    savefig(
        plot_degree_distribution(paths, outbins, outdegree_histograms),
        "$(sink)_outdegrees_histogram.png"
    )
    savefig(
        plot_scree(paths, screes),
        "$(sink)_scree.png"
    )
end

main(; parse_args(settings, as_symbols = true)...)