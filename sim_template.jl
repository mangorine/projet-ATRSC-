using SimJulia
using Random
using Distributions
using ResumableFunctions
using Statistics
using Printf
using Plots

times = Float64[]
nb_pieces_per_times = Int[]
current_nb_of_pieces = 0

@resumable function piece(env::Environment, name::Int, machine::Resource, current_nbo::Ref{Int})
    current_nbo[] += 1
    println()
    push!(times, now(env))
    push!(nb_pieces_per_times, current_nbo[])

    @yield request(machine)
    service_time = rand(Exponential(1/8))
    @yield timeout(env, service_time)
    @yield release(machine)

    current_nbo[] -= 1
    push!(times, now(env))
    push!(nb_pieces_per_times, current_nbo[])
end

@resumable function arrival_process(env::Environment, machine::Resource, time_limit::Float64, current_nbo::Ref{Int})
    i = 0
    while now(env) < time_limit
        interarrival = rand(Exponential(1/3))
        if now(env) + interarrival > time_limit
            break
        end
        @yield timeout(env, interarrival)
        i += 1
        @process piece(env, i, machine, current_nbo)
    end
end

env = Simulation()
machine = Resource(env, 4)
@process arrival_process(env, machine, 100.0, Ref(current_nb_of_pieces))
run(env)

function detect_steady_state(times, values; window=20, tol=0.05)
    n = length(values)
    for i in (2*window):n
        m1 = mean(values[i-window+1:i])
        m2 = mean(values[i-2*window+1:i-window])
        if abs(m1 - m2) < tol
            return times[i]
        end
    end
    return nothing
end

t_ss = detect_steady_state(times, nb_pieces_per_times)

if t_ss !== nothing
    indices = findall(t -> t >= t_ss, times)
    println("Nombre moyen de pièces (régime permanent) ≈ ", mean(nb_pieces_per_times[indices]))
end

plot(times, nb_pieces_per_times,
     xlabel="Temps",
     ylabel="Nombre de pièces",
     title="Évolution du système M/M/1",
     label="Nb pièces",
     lw=2)
if t_ss !== nothing
    vline!([t_ss], label="Régime permanent")
end