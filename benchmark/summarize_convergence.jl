using ArgParse
using DataFrames
using Plots

import Arrow

settings = @add_arg_table! ArgParseSettings() begin
    "path" required = true
end

phase_color(phase) = phase == :burn ? :red : :blue

function main(; path)
    swaps = joinpath(path, "swaps.arrow") |> Arrow.Table |> DataFrame
    updates = joinpath(path, "updates.arrow") |> Arrow.Table |> DataFrame

    plot(swaps.Δlogℒ, color = phase_color.(swaps.phase))
    savefig(joinpath(path, "chain.png"))

    savefig(
        plot(
            updates.𝔼σ_logℒ,
            legend = false,
            xlabel = "Θ updates",
            ylabel = "logℒ",
        ),
        joinpath(path, "log_likelihood.png"),
    )

    savefig(
        plot(
            hcat(first.(updates.Θs)...)',
            legend = :bottom,
            xlabel = "Θ updates",
            ylabel = "Θ",
        )
        joinpath(path, "initiator.png")
    )
end

main(; parse_args(settings, as_symbols = true)...)