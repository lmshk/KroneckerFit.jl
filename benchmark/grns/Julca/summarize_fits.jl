using ArgParse
using DataFrames
using Plots
import Arrow
import JSON

settings = @add_arg_table! ArgParseSettings() begin
    "--sink", "-s"
        required = true

    "reports"
        nargs = '*'
end

function main(; sink, reports)
    data = []
    for filename in reports
        report = JSON.parsefile(filename)
        order = report["order"]
        network, variant = (
            report["network"]
            |> basename
            |> splitext
            |> first
            |> splitext
            |> first
            |> splitext
        )
        network = network |> splitext |> first
        quantile = parse(Int, last(split(variant, ".q"))) / 100
        push!(data, (;
            network,
            nodes = report["nodes"],
            size = report["size"],
            edges = report["edges"],
            order = report["order"],
            quantile,
            Θs = reshape(
                collect(Iterators.flatten(report["Θs"][1])),
                order,
                order,
            ),
            𝔼σ_logℒ = report["𝔼σ_logℒ"],
            logℒ = report["logℒ"],
            total = report["total"],
            seed = report["seed"],
            tag = report["tag"],
        ))
    end
    summary = DataFrame(data)

    mkpath(sink)
    Arrow.write(joinpath(sink, "fits.summary.arrow"), summary)
end

main(; parse_args(settings, as_symbols = true)...)