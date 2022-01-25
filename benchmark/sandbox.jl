using KroneckerFit

using Random
using SparseArrays
import Dates

using DataFrames
using LoggingExtras
import Arrow
import CSV

g = CSV.read("p2p-Gnutella30.txt", DataFrame, comment="#", header = [:s, :t]) .+ 1
G = KroneckerFit.Graph(g.s, g.t, base = 2)

seed = 42
entropy = MersenneTwister(seed)
policy = KroneckerFit.GradientAscentPolicy{Float64}()

sink = Dict(
    :swap => NamedTuple[],
    :chain_estimate => NamedTuple[],
    :chain_convergence => NamedTuple[],
    :estimate => NamedTuple[],
    :update => NamedTuple[],
)

progress = Set((:estimate, :update))

directory = mkdir("diagnostics_$(Dates.now())")

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
            event -> event.id in keys(sink),
            TransformerLogger(
                event -> (push!(sink[event.id], deepcopy(NamedTuple(event.kwargs))); event),
                NullLogger()
            )
        )
    )
) do
    @info "start" Dates.now() policy seed G.A
    result = @time KroneckerFit.fit!(; G, entropy)
    @info "done" Dates.now() policy seed result
    Arrow.write(
        joinpath(directory, "permutation.arrow"),
        DataFrame(σ = result.σ.σ)
    )
end

for (kind, events) in sink
    Arrow.write(
        joinpath(directory, "$(kind)s.arrow"),
        DataFrame(events),
    )
end