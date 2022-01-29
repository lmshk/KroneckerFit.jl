using ArgParse
using DataFrames
using Plots

import Arrow

settings = @add_arg_table! ArgParseSettings() begin
    "path" required = true
end

phase_color(phase) = phase == :burn ? :red : :blue

function main(; path)
    if isfile("$(path).swap.arrow")
        swaps = "$(path).swap.arrow" |> Arrow.Table |> DataFrame
        savefig(
            plot(swaps.Δlogℒ, color = phase_color.(swaps.phase)),
            "$(path).chain.png",
        )
    end

    if isfile("$(path).update.arrow")
        updates = "$(path).update.arrow" |> Arrow.Table |> DataFrame
        savefig(
            plot(
                updates.𝔼σ_logℒ,
                legend = false,
                xlabel = "Θ updates",
                ylabel = "logℒ",
            ),
            "$(path).log_likelihood.svg",
        )
        savefig(
            plot(
                hcat(first.(updates.Θs)...)',
                legend = :bottom,
                xlabel = "Θ updates",
                ylabel = "Θ",
            ),
            "$(path).initiators.svg",
        )
    end
end

main(; parse_args(settings, as_symbols = true)...)