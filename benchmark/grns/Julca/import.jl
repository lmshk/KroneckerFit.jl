using Random
using Statistics
import CSV

using ArgParse
using DataFrames
import Arrow

settings = @add_arg_table! ArgParseSettings() begin
    "--cull"
        required = true
        arg_type = Float64

    "--sink", "-s"
        required = true

    "networks"
        default = []
        nargs = '*'
end

function main(; cull, sink, networks)
    for network in networks
        coexpressions = CSV.read(network, DataFrame)
        threshold = quantile(coexpressions.pcc, cull)
        culled = filter(row -> (row.pcc ≥ threshold), coexpressions)
        
        genes = union(coexpressions.gene_a, coexpressions.gene_b)
        index = Dict(value => key for (key, value) in enumerate(genes))
        node(gene) = index[gene]

        sources = node.(culled.gene_a)
        targets = node.(culled.gene_b)

        @info(network,
            genes,
            coexpressions = nrow(coexpressions),
            threshold,
            edges = nrow(culled),
        )

        mkpath(sink)
        culltag = "q$(round(Int, 100 * cull))"
        Arrow.write(
            joinpath(sink, "$(basename(network)).$culltag.edges.arrow"),
            DataFrame(source = sources, target = targets)
        )
        Arrow.write(
            joinpath(sink, "$(basename(network)).$culltag.names.arrow"),
            DataFrame(gene = genes)
        )
    end
end

main(; parse_args(settings, as_symbols = true)...)