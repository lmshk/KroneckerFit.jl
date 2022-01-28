using Random
import Dates

using ArgParse
using DataFrames
using LoggingExtras
import Arrow
import JSON
import KroneckerFit

settings = @add_arg_table! ArgParseSettings() begin
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
    isnothing(seed) && @info "(Using random seeds.)"

    for network in networks
        edges = DataFrame(Arrow.Table(network))
        networkname = nextwork |> splitext |> first |> splitext |> first
        nodenames = DataFrame(Arrow.Table("$networkname.names.arrow"))
        G = KroneckerFit.Graph(
            edges.source,
            edges.target,
            base = 2,
            minimum_nodes = nrow(nodenames)
        )

        policy = KroneckerFit.GradientAscentPolicy{Float64}()
        entropy = MersenneTwister(
            isnothing(seed) ? rand(UInt32) : seed
        )

        diagnostics_stash = Dict(
            diagnostic => NamedTuple[] for diagnostic in diagnostics
        )

        tag = "$(basename(network))_$(Dates.now())"
        if !isnothing(sink)
            sink = mkpath(sink)
            open(joinpath(sink, "$tag.job.txt"), "w") do file
                JSON.print(
                    file,
                    (;
                        network,
                        genes = length(genes),
                        edges = nrow(coexpressions),
                        threshold,
                        policy,
                        entropy,
                        tag
                    ),
                    4,
                )
            end
            Arrow.write(joinpath(sink, "$tag.graph.arrow"), DataFrame(G.edges))
        end
        @info("About to fit:",
            network,
            genes = length(genes),
            coexpressions = previous_length,
            threshold,
            edges = nrow(coexpressions),
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
                nodename(node) = get(nodenames, node, missing)
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
            Arrow.write(joinpath(sink, "$tag.$kind.arrow"), DataFrame(events))
        end
    end
end

main(; parse_args(settings, as_symbols = true)...)