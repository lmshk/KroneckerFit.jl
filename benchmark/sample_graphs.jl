using Random

using ArgParse
using DataFrames
using Distributions
using Kronecker

import Arrow

settings = @add_arg_table! ArgParseSettings() begin
    "path"
        required = true
    
    "--nodes", "-n"
        required = true
        arg_type = Int
    
    "--prefix"
        default = "sample_"

    "--count", "-c"
        arg_type = Int
        default = 1

    "--kind", "-t"
        default = "kronecker"
        range_tester = in((
            "kronecker",
            "K",
            "preferential_attachment",
            "PA"
        ))
    
    "--seed"
        arg_type = Int
        default = 42

    "parameters"
        nargs = '*'
        arg_type = Float64
end

function sample_kronecker(nodes, parameters; entropy)
    n₀ = isqrt(length(parameters))
    P = kronecker(
        reshape(parameters, n₀, n₀),
        ceil(Int, log(n₀, nodes))
        # TODO ^ reduce order by 1? -- 2022-01-26
    )

    result = []
    for source in 1:nodes, target in 1:nodes
        if rand(entropy, Bernoulli(P[source, target]))
            push!(result, (; source, target))
        end
    end

    result
end

function sample_preferential_attachment(nodes, parameters)
    throw(ArgumentError(kind))
end

const kinds = Dict(
    "kronecker" => sample_kronecker,
    "K" => sample_kronecker,
    "preferential_attachment" => sample_preferential_attachment,
    "PA" => sample_preferential_attachment,
)

function main(; path, nodes, prefix, count, kind, seed, parameters)
    entropy = MersenneTwister(seed)
    for i in 1:count
        Arrow.write(
            joinpath(path, "$(prefix)$(i).graph.arrow"),
            DataFrame(kinds[kind](nodes, parameters; entropy))
        )
    end
end

main(; parse_args(settings, as_symbols = true)...)