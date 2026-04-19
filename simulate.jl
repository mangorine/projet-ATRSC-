using SimJulia
using Random
using Distributions
using ResumableFunctions
using Statistics
using Printf

# Donées du problème

m = 8  # Nombre de machines
k = 4  # Nombre de types de produits

intensities = [0.29, 0.32, 0.47, 0.38]

paths = [
    [1, 2, 3, 4, 8],   # T1
    [2, 4, 7],          # T2
    [5, 3, 1],          # T3
    [5, 6, 7, 8]        # T4
]

treatment_times = Dict(
    (1, 1) => (0.58, 0.78),
    (1, 2) => (0.23, 0.56),
    (1, 3) => (0.81, 0.93),
    (1, 4) => (0.12, 0.39),
    (1, 8) => (0.82, 1.04),

    (2, 2) => (0.59, 0.68),
    (2, 4) => (0.74, 0.77),
    (2, 7) => (0.30, 0.55),

    (3, 1) => (0.57, 0.64),
    (3, 3) => (0.37, 0.54),
    (3, 5) => (0.35, 0.63),

    (4, 5) => (0.36, 0.51),
    (4, 6) => (0.61, 0.70),
    (4, 7) => (0.78, 0.85),
    (4, 8) => (0.18, 0.37)
)

QI1 = Bool[
    1 1 0 0 0 0 0 0;
    0 0 1 1 0 0 0 0;
    0 0 0 0 1 1 0 0;
    0 0 0 0 0 0 1 1
]

QI2 = Bool[
    1 0 1 0 0 1 0 0;
    0 1 0 0 1 0 1 1;
    0 1 0 1 1 0 0 1;
    1 0 1 1 0 1 1 0
]

QI3 = Bool[
    1 1 0 0 0 0 0 0;
    0 0 1 0 0 0 0 0;
    0 0 0 1 0 1 0 0;
    0 0 0 0 1 0 0 1;
    0 0 1 0 0 1 0 0;
    1 0 0 0 0 0 1 0
]

QI4 = Bool[
    1 1 1 0 0 0 0 0;
    0 0 0 1 1 1 0 0;
    1 0 1 0 0 1 1 1;
    0 0 1 0 1 0 1 1;
    0 1 0 0 0 1 1 0;
    1 0 0 1 0 0 1 1
]

# J'ai fait une structure pour stocker l'etat de la simulation, ca rend le code plus propre et plus facile a gerer que d'avoir plein de variables globales ou de passer plein de parametres a chaque fonction.
mutable struct SimState
    Q::Matrix{Bool}
    n_emp::Int
    n_mach::Int

    machine_queue::Vector{Vector{Tuple{Float64, Int, SimJulia.Event}}}
    waiting_room::Vector{Tuple{Float64, Int, SimJulia.Event}}

    sejourn_records::Vector{Tuple{Float64, Float64}}

    work_time::Vector{Float64}

    lock::Resource
end

function SimState(env::Environment, Q::Matrix{Bool})
    n_emp, n_mach = size(Q)
    SimState(
        Q, n_emp, n_mach,
        [Tuple{Float64,Int,SimJulia.Event}[] for _ in 1:n_mach],
        Tuple{Float64,Int,SimJulia.Event}[],
        Tuple{Float64,Float64}[],
        zeros(n_emp),
        Resource(env, 1)
    )
end

# Loi uniforme
function draw_service_time(ptype::Int, machine_id::Int)::Float64
    a, b = treatment_times[(ptype, machine_id)]
    return a + (b - a) * rand()
end

# Pour chaque machine, trouve l'employe disponible qui est dans la salle d'attente depuis le plus longtemps.
function find_available_employee(state::SimState, mach::Int)::Int
    best_idx = 0
    earliest = Inf
    for (i, (t_enter, emp_id, _)) in enumerate(state.waiting_room)
        if state.Q[emp_id, mach] && t_enter < earliest
            earliest = t_enter
            best_idx = i
        end
    end
    return best_idx
end

# On utilise l'algorithme FIFO (First In First Out).
# Pour chaque employe, on regarde ses qualifications (matrice Q)
# et on choisit la machine qui a la tache la plus ancienne dans sa queue.
# Renvoie (0, 0) s'il n'a rien a faire.
function fifo_algo(state::SimState, emp_id::Int)::Tuple{Int,Int}
    best_mach = 0
    best_qi   = 0
    earliest  = Inf
    for mach in 1:state.n_mach
        if state.Q[emp_id, mach] && !isempty(state.machine_queue[mach])
            t_arr, _, _ = state.machine_queue[mach][1]
            if t_arr < earliest
                earliest  = t_arr
                best_mach = mach
                best_qi   = 1
            end
        end
    end
    return (best_mach, best_qi)
end

# Detection de l'etat stationnaire (on fait comme on a vu en TD)
function detect_steady_state_index(
    sejourn_values::Vector{Float64};
    window::Int      = 30,
    rel_tol::Float64 = 0.01,
    n_stable::Int    = 3
)::Int
    n = length(sejourn_values)
    if n < 2 * window
        return 1
    end

    stable_count = 0

    for i in (2*window):n
        m_prev = mean(sejourn_values[i - 2*window + 1 : i - window])
        m_curr = mean(sejourn_values[i - window + 1  : i])
        rel_diff = abs(m_curr - m_prev) / (abs(m_prev) + 1e-12)

        if rel_diff < rel_tol
            stable_count += 1
            if stable_count >= n_stable
                return i - window + 1
            end
        else
            stable_count = 0
        end
    end

    return 1
end

# Fonctions qui servent pour les processus des :
# - employes : ils cherchent des taches a faire, sinon ils attendent dans la salle d'attente
# - produits : ils arrivent, font la queue pour chaque machine de leur chemin, et enregistrent leur temps de sejour a la fin
# - arrivees : elles generent des produits selon un processus de Poisson

@resumable function employee_process(env::Environment, state::SimState, emp_id::Int)
    # Si l'employe peut faire quelque chose (alors d'apres la logique de la fonction fifo_algo, il est assigne a une machine).
    # Sinon, l'algorithme renvoie 0 et il va en salle d'attente.
    while true
        @yield request(state.lock)
        mach, qi = fifo_algo(state, emp_id)

        if mach == 0
            wakeup = SimJulia.Event(env)
            push!(state.waiting_room, (now(env), emp_id, wakeup))
            release(state.lock)
            @yield wakeup
        else
            t_arr, ptype, signal = state.machine_queue[mach][qi]
            deleteat!(state.machine_queue[mach], qi)
            release(state.lock)

            t_start = now(env)
            @yield timeout(env, draw_service_time(ptype, mach))
            state.work_time[emp_id] += now(env) - t_start

            succeed(signal)
        end
    end
end

@resumable function product_process(
    env::Environment,
    state::SimState,
    ptype::Int,
    arrival_time::Float64
)
    # Le produit doit passer par toutes les machines de son chemin, dans l'ordre.
    for mach in paths[ptype]
        done_signal = SimJulia.Event(env)

        @yield request(state.lock)
        push!(state.machine_queue[mach], (arrival_time, ptype, done_signal))

        emp_idx = find_available_employee(state, mach)
        if emp_idx > 0
            _, _, wakeup = state.waiting_room[emp_idx]
            deleteat!(state.waiting_room, emp_idx)
            succeed(wakeup)
        end
        release(state.lock)

        @yield done_signal
    end

    departure_time = now(env)
    push!(state.sejourn_records, (departure_time, departure_time - arrival_time))
end

@resumable function arrival_process(
    env::Environment,
    state::SimState,
    ptype::Int,
    time_limit::Float64
)
    # Processus de Poisson
    while true
        @yield timeout(env, rand(Exponential(1.0 / intensities[ptype])))
        if now(env) > time_limit
            break
        end
        @process product_process(env, state, ptype, now(env))
    end
end

# Code qui lance tout
function run_simulation(
    Q::Matrix{Bool};
    time_limit::Float64  = 1000.0,
    seed::Int            = 42,
    ss_window::Int       = 30,
    ss_rel_tol::Float64  = 0.01,
    ss_n_stable::Int     = 3,
    verbose::Bool        = true
)
    Random.seed!(seed)
    env   = Simulation()
    state = SimState(env, Q)

    # Pour chaque produit, on donne a l'environnement le processus d'arrivee.
    for ptype in 1:k
        @process arrival_process(env, state, ptype, time_limit)
    end

    # On fait de meme pour les employes.
    for emp_id in 1:state.n_emp
        @process employee_process(env, state, emp_id)
    end

    if verbose
        println("  Starting simulation...")
    end
    
    # On run
    t_sim_start = time()
    run(env, time_limit)
    t_sim_elapsed = time() - t_sim_start
    
    if verbose
        @printf("  Simulation complete in %.2f seconds\n", t_sim_elapsed)
        println("  Processing results...")
    end

    # On extrait les temps de sejour pour detecter l'etat stationnaire et calculer les indicateurs.
    sejourn_values = [s for (_, s) in state.sejourn_records]

    # Etat stationnaire
    ss_idx = detect_steady_state_index(
        sejourn_values;
        window   = ss_window,
        rel_tol  = ss_rel_tol,
        n_stable = ss_n_stable
    )

    ss_time = state.sejourn_records[ss_idx][1]

    if verbose
        n_total = length(sejourn_values)
        n_ss    = n_total - ss_idx + 1
        if ss_idx == 1
            println("  [!] Regime permanent non detecte — toute la simulation est utilisee.")
        else
            @printf("  [ok] Regime permanent detecte a t = %.2f  (%d produits sur %d)\n",
                    ss_time, n_ss, n_total)
        end
    end

    # Indicateurs
    ss_sejourns  = sejourn_values[ss_idx:end]
    mean_sejourn = mean(ss_sejourns)

    work_proportions = state.work_time ./ time_limit

    return (
        mean_sejourn       = mean_sejourn,
        work_proportions   = work_proportions,
        steady_state_time  = ss_time,
        steady_state_index = ss_idx,
        sejourn_records    = state.sejourn_records,
        work_time_total    = state.work_time
    )
end

# Fonction pour afficher les resultats de maniere propre
function print_results(instance_name::String, res)
    n_total = length(res.sejourn_records)
    n_ss    = n_total - res.steady_state_index + 1
    println("\n" * "="^56)
    println("  Instance : $instance_name")
    println("="^56)
    @printf("  Debut regime permanent : t = %.2f\n", res.steady_state_time)
    @printf("  Produits (RP / total)  : %d / %d\n", n_ss, n_total)
    @printf("  Temps de sejour moyen  : %.4f\n", res.mean_sejourn)
    println("  Proportion de temps travaille par employe :")
    for (i, p) in enumerate(res.work_proportions)
        @printf("    Employe %d : %5.1f%%\n", i, 100*p)
    end
end

# Main

function main()
    TIME_LIMIT = 500000.0

    println("--- Instance I1 ---")
    res_I1 = run_simulation(QI1; time_limit=TIME_LIMIT, seed=1, verbose=true)
    print_results("I1 (q=4, specialise)", res_I1)

    println("--- Instance I2 ---")
    res_I2 = run_simulation(QI2; time_limit=TIME_LIMIT, seed=2, verbose=true)
    print_results("I2 (q=4, polyvalent)", res_I2)

    println("--- Instance I3 ---")
    res_I3 = run_simulation(QI3; time_limit=TIME_LIMIT, seed=3, verbose=true)
    print_results("I3 (q=6, specialise)", res_I3)

    println("--- Instance I4 ---")
    res_I4 = run_simulation(QI4; time_limit=TIME_LIMIT, seed=4, verbose=true)
    print_results("I4 (q=6, polyvalent)", res_I4)
end

main()
