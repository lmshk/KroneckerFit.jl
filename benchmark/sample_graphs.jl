using Random

using ArgParse
using DataFrames
using Distributions
using Kronecker

import Arrow

settings = @add_arg_table! ArgParseSettings() begin
    "--jobs"
        required = true

    "--sink"
        required = true

    "--count", "-c"
        arg_type = Int
        default = 1

    "--kind", "-t"
        default = "kronecker"
        range_tester = in((
            "kronecker",
            "K",
            "symmetric_kronecker",
            "sK",
            "preferential_attachment",
            "PA"
        ))
    
    "--seed"
        arg_type = Int
        default = 42
end

function sample_kronecker(nodes, parameters; entropy, symmetric = false)
    n₀ = isqrt(length(parameters))
    P = kronecker(reshape(parameters, n₀, n₀), ceil(Int, log(n₀, nodes)))

    result = []
    for source in 1:nodes, target in 1:nodes
        symmetric && target < source && continue
        if rand(entropy, Bernoulli(P[source, target]))
            push!(result, (; source, target))
            if symmetric && source != target
                push!(result, (; source = target, target = source))
            end
        end
    end

    result
end

sample_symmetric_kronecker(nodes, parameters; entropy) =
    sample_kronecker(nodes, parameters; entropy, symmetric = true)

function sample_preferential_attachment(nodes, parameters)
    throw(ArgumentError(kind))
end

const kinds = Dict(
    "kronecker" => sample_kronecker,
    "K" => sample_kronecker,
    "symmetric_kronecker" => sample_symmetric_kronecker,
    "sK" => sample_symmetric_kronecker,
    "preferential_attachment" => sample_preferential_attachment,
    "PA" => sample_preferential_attachment,
)

function main(; jobs, sink, count, kind, seed)
    fits = DataFrame(Arrow.Table(jobs))
    entropy = MersenneTwister(seed)
    mkpath(sink)
    for row in eachrow(fits)
        @info row.tag row
        for i in 1:count
            name = "$(row.tag)_$(kind)_$i.sample.graph.arrow"
            @info name
            Arrow.write(
                joinpath(sink, name),
                DataFrame(kinds[kind](row.size, row.Θs; entropy))
            )
        end
    end
end

main(; parse_args(settings, as_symbols = true)...)