# Fonction de calcul du score PT+WINQ
function calculer_pt_winq(atelier::Atelier, p::Produit, m_id::Int)
    # PT : Espérance du temps sur la machine actuelle
    a, b = TEMPS[(p.type, m_id)]
    pt = (a + b) / 2.0
    
    # WINQ : Charge de travail déjà en attente sur la PROCHAINE machine
    winq = 0.0
    if p.etape < length(PARCOURS[p.type])
        next_m_id = PARCOURS[p.type][p.etape + 1]
        next_machine = atelier.machines[next_m_id]
        
        for p_attente in next_machine.file_attente
            a_next, b_next = TEMPS[(p_attente.type, next_m_id)]
            winq += (a_next + b_next) / 2.0
        end
    end
    
    return pt + winq
end

# Trouver le produit avec le meilleur score PT+WINQ dans une file
function index_produit_pt_winq(atelier::Atelier, machine::Machine, m_id::Int)
    isempty(machine.file_attente) && return nothing
    idx_min = 1
    val_min = calculer_pt_winq(atelier, machine.file_attente[1], m_id)
    
    for i in 2:length(machine.file_attente)
        val = calculer_pt_winq(atelier, machine.file_attente[i], m_id)
        if val < val_min
            val_min = val
            idx_min = i
        # Départage avec la règle FIFO classique en cas d'égalité
        elseif val == val_min && machine.file_attente[i].arrivee_atelier < machine.file_attente[idx_min].arrivee_atelier
            idx_min = i 
        end
    end
    return idx_min
end

# Modification du choix de la machine (Remplacement de la fonction existante)
function choisir_machine(atelier::Atelier, emp_id::Int, mode_algo::String, mode_choix::String)
    emp = atelier.employes[emp_id]
    candidats = Tuple{Int,Produit,Float64}[] # On stocke: (machine_id, produit, score)
    
    for m_id in emp.qualifications
        machine = atelier.machines[m_id]
        if !machine.occupee && !isempty(machine.file_attente)
            if mode_choix == "PLUS_TOT_ATELIER"
                idx = index_produit_doyen(machine.file_attente)
                score = machine.file_attente[idx].arrivee_atelier
            elseif mode_choix == "PT_WINQ"
                idx = index_produit_pt_winq(atelier, machine, m_id)
                score = calculer_pt_winq(atelier, machine.file_attente[idx], m_id)
            else
                error("Mode de choix non implémenté : $mode_choix")
            end
            push!(candidats, (m_id, machine.file_attente[idx], score))
        end
    end
    isempty(candidats) && return nothing

    # L'employé trie les machines éligibles selon le meilleur score trouvé
    if mode_algo == "FIFO_ATELIER"
        sort!(candidats, by = x -> x[2].arrivee_atelier)
    elseif mode_algo == "PT_WINQ"
        sort!(candidats, by = x -> x[3]) # Trie par le score PT+WINQ le plus bas
    else
        error("Algorithme non implémenté : $mode_algo")
    end
    
    choix = candidats[1][1]
    log_event(atelier, "Employé $emp_id choisit la machine $choix (algo=$mode_algo, choix=$mode_choix)")
    return choix
end

# Modification mineure du processus employé pour gérer le nouveau mode
@resumable function processus_employe(sim::Simulation, emp_id::Int, atelier::Atelier, mode_algo::String, mode_choix::String)
    emp = atelier.employes[emp_id]
    while true
        if !emp.est_occupe
            m_id = choisir_machine(atelier, emp_id, mode_algo, mode_choix)
            if m_id !== nothing
                machine = atelier.machines[m_id]
                machine.occupee = true
                emp.est_occupe = true
                
                if mode_choix == "PLUS_TOT_ATELIER"
                    idx_produit = index_produit_doyen(machine.file_attente)
                elseif mode_choix == "PT_WINQ"
                    idx_produit = index_produit_pt_winq(atelier, machine, m_id)
                else
                    idx_produit = 1
                end
                
                p = machine.file_attente[idx_produit]
                deleteat!(machine.file_attente, idx_produit)
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

Q1 = [1 1 0 0 0 0 0 0; 0 0 1 1 0 0 0 0; 0 0 0 0 1 1 0 0; 0 0 0 0 0 0 1 1]
Q2 = [1 0 1 0 0 1 0 0; 0 1 0 0 1 0 1 1; 0 1 0 1 1 0 0 1; 1 0 1 1 0 1 1 0]
Q3 = [1 1 0 0 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 1 0 1 0 0; 0 0 0 0 1 0 0 1; 0 0 1 0 0 1 0 0; 1 0 0 0 0 0 1 0]
Q4 = [1 1 1 0 0 0 0 0; 0 0 0 1 1 1 0 0; 1 0 1 0 0 1 1 1; 0 0 1 0 1 0 1 1; 0 1 0 0 0 1 1 0; 1 0 0 1 0 0 1 1]

Random.seed!(123)
println("LANCEMENT DE L'ÉTUDE (QUESTION 6)")

etude_performance(Q1, "I1", "PT_WINQ", "PT_WINQ", verbose=false, plot_convergence=true)
etude_performance(Q2, "I2", "PT_WINQ", "PT_WINQ", verbose=false, plot_convergence=true)
etude_performance(Q3, "I3", "PT_WINQ", "PT_WINQ", verbose=false, plot_convergence=true)
etude_performance(Q4, "I4", "PT_WINQ", "PT_WINQ", verbose=false, plot_convergence=true)