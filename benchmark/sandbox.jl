using KroneckerFit

using Random
using SparseArrays
import Dates

using DataFrames
using LoggingExtras
import Arrow
import CSV

g = CSV.read("p2p-Gnutella25.txt", DataFrame, comment="#", header = [:s, :t]) .+ 1
G = KroneckerFit.Graph(g.s, g.t, base = 2)

entropy = MersenneTwister(42)
policy = KroneckerFit.GradientAscentPolicy{Float64}(
    maximum_steps = 100,
)

sink = Dict(
    :swap => NamedTuple[],
    :estimate => NamedTuple[],
    :convergence => NamedTuple[],
    :update => NamedTuple[],
)

progress = Set((:convergence, :update))

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
    @info "start" Dates.now() policy G
    result = @time KroneckerFit.fit!(; G, policy, entropy)
    @info "done" result
end

directory = mkdir("diagnostics_$(Dates.now())")
for (kind, events) in sink
    Arrow.write(
        joinpath(directory, "$(kind)s.arrow"),
        DataFrame(events),
    )
end