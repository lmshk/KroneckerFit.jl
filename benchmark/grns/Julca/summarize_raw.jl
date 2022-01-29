using ArgParse
using DataFrames
using Plots

import CSV

settings = @add_arg_table! ArgParseSettings() begin
    "--histogram"
        default = "co-expression-histogram.svg"

    "networks"
        nargs = '+'
end

function main(; histogram, networks)
    data = CSV.read.(networks, DataFrame)
    savefig(
        plot(
            getindex.(data, !, :pcc),
            seriestype = :stephist,
            label = reshape(basename.(networks), 1, 13),
            normalize = true,
            legend = :topleft,
            linewidth = .5
        ),
        histogram
    )
end

main(; parse_args(settings, as_symbols = true)...)