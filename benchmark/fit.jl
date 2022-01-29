using Random
import Dates

using ArgParse
using DataFrames
using LoggingExtras
import Arrow
import JSON
import KroneckerFit

settings = @add_arg_table! ArgParseSettings() begin
    "--order"
        arg_type = Int
        default = 2

    "--seed"
        arg_type = Int
    
    "--progress", "-p"
        nargs = '*'
        arg_type = Symbol

    "--diagnostics", "-d"
        nargs = '*'
        arg_type = Symbol

    "--sink", "-s"

    "--networks"
        default = []
        nargs = '*'
    
    "more_networks"
        default = []
        nargs = '*'
end

function main(;
    order,
    seed,
    sink,
    progress,
    diagnostics,
    networks,
    more_networks
)
    networks = networks..., more_networks...
    diagnostics = isnothing(diagnostics) ? Set() : Set(diagnostics)
    if isnothing(sink)
        @warn "Results won't be saved!"
        diagnostics = Set()
    end
    progress = isnothing(progress) ? Set() : Set(progress)

    @info "Settings" sink progress diagnostics seed
    if isnothing(seed)
        @info "(Picking random seed.)"
        seed = rand(UInt32)
    end

    for network in networks
        edges = DataFrame(Arrow.Table(network))
        networkname = network |> splitext |> first |> splitext |> first
        nodenames = DataFrame(Arrow.Table("$networkname.names.arrow"))
        G = KroneckerFit.Graph(
            edges.source,
            edges.target,
            base = order,
            minimum_nodes = nrow(nodenames)
        )

        policy = KroneckerFit.GradientAscentPolicy{Float64}()
        entropy = MersenneTwister(seed)

        diagnostics_stash = Dict(
            diagnostic => NamedTuple[] for diagnostic in diagnostics
        )

        tag = "$(basename(network))_$(Dates.now())"
        if !isnothing(sink)
            sink = mkpath(sink)
            open(joinpath(sink, "$tag.job.json"), "w") do file
                JSON.print(
                    file,
                    (;
                        network,
                        order,
                        nodes = nrow(nodenames),
                        size = first(size(G.A)),
                        edges = nrow(edges),
                        policy,
                        seed,
                        tag
                    ),
                    4,
                )
            end
        end
        @info("About to fit:",
            network,
            order,
            nodes = nrow(nodenames),
            size = size(G.A),
            edges = nrow(edges),
            policy,
            entropy,
            tag,
        )
        with_logger(
            TeeLogger(
                global_logger(),
                EarlyFilteredLogger(
                    event -> event.id in progress,
                    TransformerLogger(
                        event -> merge(event, (; level=Logging.Info)),
                        ConsoleLogger(stderr, Logging.Debug)
                    )
                ),
                EarlyFilteredLogger(
                    event -> event.id in keys(diagnostics_stash),
                    TransformerLogger(
                        event -> (
                            push!(
                                diagnostics_stash[event.id],
                                deepcopy(NamedTuple(event.kwargs))
                            );
                            event
                        ),
                        NullLogger()
                    )
                )
            )
        ) do
            result = @time KroneckerFit.fit!(; G, entropy)
            @info "Network fit." network tag Dates.now() result
            if !isnothing(sink)
                open(joinpath(sink, "$tag.report.json"), "w") do file
                    JSON.print(
                        file,
                        (;
                            tag,
                            network,
                            order,
                            nodes = nrow(nodenames),
                            size = first(size(G.A)),
                            edges = nrow(edges),
                            policy,
                            seed,
                            result.Θs,
                            result.𝔼σ_logℒ,
                            result.logℒ,
                            result.converged,
                            result.z,
                            result.samples,
                            result.burned,
                            result.total,
                        ),
                        4,
                    )
                end

                nodename(node) = get(nodenames[!, 1], node, missing)
                Arrow.write(
                    joinpath(sink, "$tag.mapping.arrow"),
                    DataFrame(
                        σ = result.σ.σ,
                        mapping = nodename.(result.σ.σ),
                    )
                )
            end
        end

        for (kind, events) in diagnostics_stash
            Arrow.write(joinpath(sink, "$tag.diagnostics.$kind.arrow"), DataFrame(events))
        end
    end
end

main(; parse_args(settings, as_symbols = true)...)