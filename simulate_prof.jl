using SimJulia, Distributions, Random, Statistics, ResumableFunctions, Printf
using Plots  # <-- Ajout pour les graphiques

# --- DONNÉES OFFICIELLES (inchangées) ---
const λ = [0.29, 0.32, 0.47, 0.38]
const PARCOURS = [
    [1, 2, 3, 4, 8], # T1
    [2, 4, 7],       # T2
    [3, 5, 1],       # T3
    [5, 6, 7, 8]     # T4
]
const TEMPS = Dict(
    (1,1)=>(0.58,0.78), (1,2)=>(0.23,0.56), (1,3)=>(0.81,0.93), (1,4)=>(0.12,0.39), (1,8)=>(0.82,1.04),
    (2,2)=>(0.59,0.68), (2,4)=>(0.74,0.77), (2,7)=>(0.30,0.55),
    (3,3)=>(0.37,0.54), (3,5)=>(0.35,0.63), (3,1)=>(0.57,0.64),
    (4,5)=>(0.36,0.51), (4,6)=>(0.61,0.70), (4,7)=>(0.78,0.85), (4,8)=>(0.18,0.37)
)

# --- STRUCTURES ---
mutable struct Produit
    id::Int
    type::Int
    arrivee_atelier::Float64
    arrivee_machine::Float64 
    etape::Int
end

mutable struct Machine
    file_attente::Vector{Produit}
    en_service::Union{Produit,Nothing}
    occupee::Bool
    Machine() = new(Produit[], nothing, false)
end

mutable struct Employe
    id::Int
    qualifications::Vector{Int}
    temps_entree_salle::Float64
    est_occupe::Bool
    travail_cumule::Float64
    machine_assignee::Union{Int,Nothing}
    Employe(id, quals) = new(id, quals, 0.0, false, 0.0, nothing)
end

mutable struct Atelier
    sim::Simulation
    machines::Vector{Machine}
    employes::Vector{Employe}
    file_inactifs::Vector{Int}
    events_employes::Vector{Event}
    temps_sejour::Vector{Float64}
    temps_sejour_history::Vector{Tuple{Float64,Float64}}  # (temps de fin, temps de séjour) pour le tracé
    compteur_produits::Int
    fin_transitoire::Float64
    verbose::Bool
    record_history::Bool                                   # active l'enregistrement pour le tracé
    function Atelier(sim, Q, verbose, record_history)
        nb_machines = 8
        nb_emp = size(Q,1)
        machines = [Machine() for _ in 1:nb_machines]
        employes = [Employe(i, findall(j -> Q[i,j]==1, 1:nb_machines)) for i in 1:nb_emp]
        events = [Event(sim) for _ in 1:nb_emp]
        new(sim, machines, employes, Int[], events, Float64[], Tuple{Float64,Float64}[], 0, 0.0, verbose, record_history)
    end
end

# --- FONCTION DE LOG ---
log_event(atelier::Atelier, msg::String) = atelier.verbose && println("[t=$(Printf.@sprintf("%.3f", now(atelier.sim)))] $msg")
                
# --- FONCTIONS AUXILIAIRES (inchangées) ---

function par_temps_entree(atelier::Atelier, id::Int)
    return atelier.employes[id].temps_entree_salle
end

function index_produit_doyen(file::Vector{Produit})
    isempty(file) && return nothing
    idx_min = 1
    val_min = file[1].arrivee_atelier
    for i in 2:length(file)
        val = file[i].arrivee_atelier
        if val < val_min
            val_min = val
            idx_min = i
        end
    end
    return idx_min
end

function entrer_salle_attente!(atelier::Atelier, emp_id::Int, temps::Float64)
    emp = atelier.employes[emp_id]
    emp.est_occupe = false
    emp.temps_entree_salle = temps
    push!(atelier.file_inactifs, emp_id)
    sort!(atelier.file_inactifs, by = id -> par_temps_entree(atelier, id))
    log_event(atelier, "Employé $emp_id entre en salle d'attente (file=$(atelier.file_inactifs))")
    return nothing
end

function sortir_salle_attente!(atelier::Atelier, emp_id::Int)
    filter!(e -> e != emp_id, atelier.file_inactifs)
    atelier.employes[emp_id].est_occupe = true
    log_event(atelier, "Employé $emp_id sort de la salle d'attente")
end

function choisir_employe_pour_machine(atelier::Atelier, machine_id::Int)
    for emp_id in atelier.file_inactifs
        emp = atelier.employes[emp_id]
        if machine_id in emp.qualifications
            sortir_salle_attente!(atelier, emp_id)
            return emp_id
        end
    end
    return nothing
end

function choisir_machine(atelier::Atelier, emp_id::Int, mode_algo::String, mode_choix::String)
    emp = atelier.employes[emp_id]
    candidats = Tuple{Int,Produit}[]
    for m_id in emp.qualifications
        machine = atelier.machines[m_id]
        if !machine.occupee && !isempty(machine.file_attente)
            if mode_choix == "PLUS_TOT_ATELIER"
                idx = index_produit_doyen(machine.file_attente)
                doyen = machine.file_attente[idx]
                push!(candidats, (m_id, doyen))
            else
                error("Mode de choix non implémenté : $mode_choix")
            end
        end
    end
    isempty(candidats) && return nothing

    if mode_algo == "FIFO_ATELIER"
        sort!(candidats, by = x -> x[2].arrivee_atelier)
    else
        error("Algorithme non implémenté : $mode_algo")
    end
    choix = candidats[1][1]
    log_event(atelier, "Employé $emp_id choisit la machine $choix (algo=$mode_algo, choix=$mode_choix)")
    return choix
end

function arriver_sur_machine!(atelier::Atelier, p::Produit, m_id::Int, temps::Float64)
    machine = atelier.machines[m_id]
    log_event(atelier, "Produit $(p.id) (type $(p.type)) arrive sur machine $m_id (étape $(p.etape))")
    if machine.en_service === nothing && !machine.occupee
        emp_id = choisir_employe_pour_machine(atelier, m_id)
        if emp_id !== nothing
            machine.occupee = true
            machine.en_service = p
            atelier.employes[emp_id].machine_assignee = m_id
            log_event(atelier, "Machine $m_id libre -> Employé $emp_id assigné, début service immédiat")
            ev = atelier.events_employes[emp_id]
            if state(ev) == SimJulia.idle
                succeed(ev)
            end
        else
            push!(machine.file_attente, p)
            log_event(atelier, "Machine $m_id libre mais aucun employé qualifié disponible -> produit en attente (file=$(length(machine.file_attente)))")
        end
    else
        push!(machine.file_attente, p)
        log_event(atelier, "Machine $m_id occupée -> produit en attente (file=$(length(machine.file_attente)))")
    end
    return nothing
end

# --- PROCESSUS (avec enregistrement historique si activé) ---

@resumable function processus_employe(sim::Simulation, emp_id::Int, atelier::Atelier, 
                                      mode_algo::String, mode_choix::String)
    emp = atelier.employes[emp_id]
    while true
        if !emp.est_occupe
            m_id = choisir_machine(atelier, emp_id, mode_algo, mode_choix)
            if m_id !== nothing
                machine = atelier.machines[m_id]
                machine.occupee = true
                emp.est_occupe = true
                idx_doyen = index_produit_doyen(machine.file_attente)
                p = machine.file_attente[idx_doyen]
                deleteat!(machine.file_attente, idx_doyen)
                machine.en_service = p
                emp.machine_assignee = nothing
                duree = rand(Uniform(TEMPS[(p.type, m_id)]...))
                log_event(atelier, "Employé $emp_id commence service sur machine $m_id pour produit $(p.id) (durée=$(round(duree,digits=3)))")
                @yield timeout(sim, duree)
                if now(sim) >= atelier.fin_transitoire
                    emp.travail_cumule += duree
                end
                machine.en_service = nothing
                machine.occupee = false
                emp.est_occupe = false
                log_event(atelier, "Employé $emp_id termine service sur machine $m_id")
                p.etape += 1
                if p.etape <= length(PARCOURS[p.type])
                    m_suiv = PARCOURS[p.type][p.etape]
                    p.arrivee_machine = now(sim)
                    arriver_sur_machine!(atelier, p, m_suiv, now(sim))
                else
                    t_fin = now(sim)
                    ts = t_fin - p.arrivee_atelier
                    if now(sim) >= atelier.fin_transitoire
                        push!(atelier.temps_sejour, ts)
                    end
                    if atelier.record_history
                        push!(atelier.temps_sejour_history, (t_fin, ts))
                    end
                    log_event(atelier, "Produit $(p.id) termine son parcours (temps séjour=$ts)")
                end
                continue
            end
        end

        if !emp.est_occupe && emp.machine_assignee === nothing
            entrer_salle_attente!(atelier, emp_id, now(sim))
            atelier.events_employes[emp_id] = Event(sim)
            ev = atelier.events_employes[emp_id]
            log_event(atelier, "Employé $emp_id attend un événement (salle d'attente)")
            @yield ev
            m_id = emp.machine_assignee
            emp.machine_assignee = nothing
            if m_id === nothing
                error("Employé $emp_id réveillé sans machine assignée")
            end
            machine = atelier.machines[m_id]
            p = machine.en_service
            duree = rand(Uniform(TEMPS[(p.type, m_id)]...))
            log_event(atelier, "Employé $emp_id réveillé, commence service sur machine $m_id pour produit $(p.id) (durée=$(round(duree,digits=3)))")
            @yield timeout(sim, duree)
            if now(sim) >= atelier.fin_transitoire
                emp.travail_cumule += duree
            end
            machine.en_service = nothing
            machine.occupee = false
            emp.est_occupe = false
            log_event(atelier, "Employé $emp_id termine service sur machine $m_id")
            p.etape += 1
            if p.etape <= length(PARCOURS[p.type])
                m_suiv = PARCOURS[p.type][p.etape]
                p.arrivee_machine = now(sim)
                arriver_sur_machine!(atelier, p, m_suiv, now(sim))
            else
                t_fin = now(sim)
                ts = t_fin - p.arrivee_atelier
                if now(sim) >= atelier.fin_transitoire
                    push!(atelier.temps_sejour, ts)
                end
                if atelier.record_history
                    push!(atelier.temps_sejour_history, (t_fin, ts))
                end
                log_event(atelier, "Produit $(p.id) termine son parcours (temps séjour=$ts)")
            end
            continue
        end
    end
end

@resumable function generateur(sim::Simulation, type::Int, atelier::Atelier)
    while true
        @yield timeout(sim, rand(Exponential(1/λ[type])))
        atelier.compteur_produits += 1
        t_now = now(sim)
        p = Produit(atelier.compteur_produits, type, t_now, t_now, 1)
        log_event(atelier, "Nouveau produit $(p.id) de type $type arrive dans l'atelier")
        premiere_machine = PARCOURS[type][1]
        arriver_sur_machine!(atelier, p, premiere_machine, t_now)
    end
end

@resumable function processus_reinitialisation(sim::Simulation, atelier::Atelier, duree_transient::Float64)
    @yield timeout(sim, duree_transient)
    log_event(atelier, "=== FIN PÉRIODE TRANSITOIRE (t=$duree_transient) - RÉINITIALISATION STATISTIQUES ===")
    for emp in atelier.employes
        emp.travail_cumule = 0.0
    end
end

# --- FONCTION DE SIMULATION AVEC OPTION GRAPHIQUE ---

function etude_performance(Q, label, mode_algo="FIFO_ATELIER", mode_choix="PLUS_TOT_ATELIER", 
                           n_runs=20, duree_transient=10000.0, duree_permanent=1000.0;
                           verbose=false, plot_convergence=false)
    sejours_moyens = Float64[]
    nb_emp = size(Q, 1)
    occupations = [Float64[] for _ in 1:nb_emp]

    for r in 1:n_runs
        record_this_run = plot_convergence && (r == 1)   # seulement le premier run si demandé
        verbose && println("\n--- RUN $r ---")
        sim = Simulation()
        atelier = Atelier(sim, Q, verbose, record_this_run)
        atelier.fin_transitoire = duree_transient
        
        @process processus_reinitialisation(sim, atelier, duree_transient)
        
        for i in 1:nb_emp
            @process processus_employe(sim, i, atelier, mode_algo, mode_choix)
        end
        for t in 1:4
            @process generateur(sim, t, atelier)
        end
        
        run(sim, duree_transient + duree_permanent)
        
        if !isempty(atelier.temps_sejour)
            push!(sejours_moyens, mean(atelier.temps_sejour))
        end
        for i in 1:nb_emp
            push!(occupations[i], atelier.employes[i].travail_cumule / duree_permanent * 100)
        end
        verbose && println("Fin run $r : produits terminés en phase permanente = $(length(atelier.temps_sejour))")

        # --- Tracé pour le premier run si demandé ---
        if record_this_run && !isempty(atelier.temps_sejour_history)
            # Calculer la moyenne cumulée
            history = atelier.temps_sejour_history
            sort!(history, by = x -> x[1])  # tri par temps de fin
            temps = [t for (t, _) in history]
            sejours = [s for (_, s) in history]
            cum_mean = cumsum(sejours) ./ (1:length(sejours))
            
            p = plot(temps, cum_mean, 
                     label = "Moyenne cumulée",
                     xlabel = "Temps simulé",
                     ylabel = "Temps de séjour moyen",
                     title = "Convergence du temps de séjour - Instance $label (Run 1)")
            vline!([duree_transient], linestyle=:dash, color=:red, label="Fin transitoire (t=$duree_transient)")
            display(p)
            println("Graphique de convergence affiché pour l'instance $label.")
        end
    end

    m = mean(sejours_moyens)
    ic = 1.96 * std(sejours_moyens) / sqrt(n_runs)
    occ_moy = [mean(occupations[i]) for i in 1:nb_emp]
    
    println("-"^60)
    @printf("Instance %s | Algo: %-12s | Choix: %-15s\n", label, mode_algo, mode_choix)
    @printf(" > Temps de séjour moyen : %.2f ± %.2f\n", m, ic)
    @printf(" > Occupation employés : %s\n", join([@sprintf("%.1f%%", x) for x in occ_moy], ", "))
end

# --- EXÉCUTION AVEC GRAPHIQUES ---
Q1 = [1 1 0 0 0 0 0 0; 0 0 1 1 0 0 0 0; 0 0 0 0 1 1 0 0; 0 0 0 0 0 0 1 1]
Q2 = [1 0 1 0 0 1 0 0; 0 1 0 0 1 0 1 1; 0 1 0 1 1 0 0 1; 1 0 1 1 0 1 1 0]
Q3 = [1 1 0 0 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 1 0 1 0 0; 0 0 0 0 1 0 0 1; 0 0 1 0 0 1 0 0; 1 0 0 0 0 0 1 0]
Q4 = [1 1 1 0 0 0 0 0; 0 0 0 1 1 1 0 0; 1 0 1 0 0 1 1 1; 0 0 1 0 1 0 1 1; 0 1 0 0 0 1 1 0; 1 0 0 1 0 0 1 1]

Random.seed!(123)
println("LANCEMENT DE L'ÉTUDE COMPARATIVE AVEC GRAPHIQUES DE CONVERGENCE")

# Attention, verbose:true affiche tous les événements ; à n'utiliser que sur des simulations très très courtes 
# Utile essentiellement pour débuguer

etude_performance(Q1, "I1", "FIFO_ATELIER", "PLUS_TOT_ATELIER", verbose=false, plot_convergence=true)
etude_performance(Q2, "I2", "FIFO_ATELIER", "PLUS_TOT_ATELIER", verbose=false, plot_convergence=true)
etude_performance(Q3, "I3", "FIFO_ATELIER", "PLUS_TOT_ATELIER", verbose=false, plot_convergence=true)
etude_performance(Q4, "I4", "FIFO_ATELIER", "PLUS_TOT_ATELIER", verbose=false, plot_convergence=true)