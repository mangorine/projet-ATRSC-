import simpy
import random
import numpy as np
from statistics import mean
import time as time_module

# Données du problème
m = 8  # Nombre de machines
k = 4  # Nombre de types de produits

intensities = [0.29, 0.32, 0.47, 0.38]

paths = [
    [1, 2, 3, 4, 8],   # T1
    [2, 4, 7],          # T2
    [5, 3, 1],          # T3
    [5, 6, 7, 8]        # T4
]

treatment_times = {
    (1, 1): (0.58, 0.78),
    (1, 2): (0.23, 0.56),
    (1, 3): (0.81, 0.93),
    (1, 4): (0.12, 0.39),
    (1, 8): (0.82, 1.04),
    
    (2, 2): (0.59, 0.68),
    (2, 4): (0.74, 0.77),
    (2, 7): (0.30, 0.55),
    
    (3, 1): (0.57, 0.64),
    (3, 3): (0.37, 0.54),
    (3, 5): (0.35, 0.63),
    
    (4, 5): (0.36, 0.51),
    (4, 6): (0.61, 0.70),
    (4, 7): (0.78, 0.85),
    (4, 8): (0.18, 0.37)
}

QI1 = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1]
], dtype=bool)

QI2 = np.array([
    [1, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0]
], dtype=bool)

QI3 = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0]
], dtype=bool)

QI4 = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 1]
], dtype=bool)

# J'ai fait une classe pour stocker l'état de la simulation, ça rend le code plus propre et plus facile à gérer que d'avoir plein de variables globales ou de passer pleins de paramètres à chaque fonction.
class SimState:
    def __init__(self, env, Q):
        self.env = env
        self.Q = Q
        self.n_emp, self.n_mach = Q.shape
        
        # Queues des machines: list of lists of (arrival_time, product_type, service_event)
        self.machine_queue = [[] for _ in range(self.n_mach)]
        
        # Salle d'attente des employés: list of (enter_time, emp_id, wakeup_event)
        self.waiting_room = []
        
        # Temps de séjour des produits: list of (arrival_time, sejourn_time)
        self.sejourn_records = []
        
        # Temps de travail des employés (optionnel pour l'instant mais intéressant pour la suite)
        self.work_time = [0.0] * self.n_emp

def draw_service_time(ptype, machine_id):
    """Loi uniforme"""
    a, b = treatment_times[(ptype, machine_id)]
    return a + (b - a) * random.random()

def find_available_employee(state, mach):
    """Pour chaque machine, trouve l'employé disponible qui est dans la salle d'attente depuis le plus longtemps"""
    best_idx = -1
    earliest = float('inf')
    for i, (t_enter, emp_id, _) in enumerate(state.waiting_room):
        if state.Q[emp_id, mach] and t_enter < earliest:
            earliest = t_enter
            best_idx = i
    return best_idx

def fifo_algo(state, emp_id):
    """
    On utilise l'algorithme FIFO (First In First Out)
    Pour chaque employé, on regarde ses qualifications (matrice Q)
    et on choisit la machine qui a la tâche la plus ancienne dans sa queue.
    Renvoie -1 s'il n'a rien à faire.
    """
    best_mach = -1
    earliest = float('inf')
    for mach in range(state.n_mach):
        if state.Q[emp_id, mach] and len(state.machine_queue[mach]) > 0:
            t_arr, _, _ = state.machine_queue[mach][0]
            if t_arr < earliest:
                earliest = t_arr
                best_mach = mach
    return best_mach

# Détection de l'état stationnaire (on fait comme on a vu en TD)
def detect_steady_state_index(sejourn_values, window=30, tol=0.01, n_stable=3):
    """Détecte l'indice à partir duquel les temps de séjour sont stables"""
    n = len(sejourn_values)
    if n < 2 * window:
        return 0
    
    stable_count = 0
    
    for i in range(2 * window, n):
        m_prev = mean(sejourn_values[i - 2*window : i - window])
        m_curr = mean(sejourn_values[i - window : i])
        rel_diff = abs(m_curr - m_prev) / (abs(m_prev) + 1e-12)
        
        if rel_diff < tol:
            stable_count += 1
            if stable_count >= n_stable:
                return i - window
        else:
            stable_count = 0
    
    return 0

# Fonctions qui servent pour les processus des :
# - employés : ils cherchent des tâches à faire, sinon ils attendent dans la salle d'attente
# - produits : ils arrivent, font la queue pour chaque machine de leur chemin, et enregistrent leur temps de séjour à la fin
# - arrivées : ils génèrent des produits selon un processus de Poisson
def employee_process(env, state, emp_id):
    """
    Si l'employé peut faire quelque chose (alors d'après la logique de la fonction fifo_algo, il est assigné à un machine)
    Sinon, l'algorithme renvoie -1 et il va en salle d'attente.
    """
    while True:
        mach = fifo_algo(state, emp_id)
        
        if mach == -1:
            # Aucun travail, aller en salle d'attente
            wakeup = simpy.Event(env)
            state.waiting_room.append((env.now, emp_id, wakeup))
            
            # On attend qu'un produit arrive
            yield wakeup
        else:
            # Il y a du travail à faire, on prend la tâche la plus ancienne de la machine
            t_arr, ptype, service_event = state.machine_queue[mach].pop(0)
            
            # On effectue le service
            t_start = env.now
            service_time = draw_service_time(ptype, mach + 1)
            yield env.timeout(service_time)

            state.work_time[emp_id] += env.now - t_start
            
            # On signale que le service est terminé pour le produit
            service_event.succeed()

def product_process(env, state, ptype, arrival_time):
    """Le produit doit passer par toutes les machines de son chemin, dans l'ordre."""
    for mach_id in paths[ptype - 1]:
        mach = mach_id - 1
        
        # On crée l'événement (c'est comme ça que ça marche avec SimPy)
        service_done = simpy.Event(env)
        
        # On met le produit dans la queue de la machine correspondante
        state.machine_queue[mach].append((arrival_time, ptype, service_done))
        
        # On essaie de prendre un employé disponible pour cette machine
        emp_idx = find_available_employee(state, mach)
        if emp_idx >= 0:
            _, _, wakeup_event = state.waiting_room.pop(emp_idx)
            wakeup_event.succeed()
        
        # On attend que le service soit terminé
        yield service_done
    
    # On enregistre le temps de séjour du produit
    state.sejourn_records.append((env.now, env.now - arrival_time))

def arrival_process(env, state, ptype):
    """Processus de Poisson"""
    while True:
        inter_arrival = random.expovariate(intensities[ptype - 1])
        yield env.timeout(inter_arrival)
        
        # Produit qui arrive
        env.process(product_process(env, state, ptype, env.now))

# Simulation
def run_simulation(Q, time_limit=1000.0, seed=42):
    """Code qui lance tout"""
    random.seed(seed)
    np.random.seed(seed)
    
    env = simpy.Environment()
    state = SimState(env, Q)
    
    # Pour chaque produit, on donne à l'environnement le processus d'arrivée du produit
    for ptype in range(1, k + 1):
        env.process(arrival_process(env, state, ptype))
    
    # On fait de même pour les employés
    for emp_id in range(state.n_emp):
        env.process(employee_process(env, state, emp_id))
    
    # On run
    env.run(until=time_limit)
    
    # On extrait les temps de séjour pour détecter l'état stationnaire et calculer les indicateurs
    sejourn_values = [s for (_, s) in state.sejourn_records]
    
    # Etat stationnaire
    statio_idx = detect_steady_state_index(sejourn_values)
    statio_time = state.sejourn_records[statio_idx][0] if state.sejourn_records else 0
    
    # Indicateurs
    statio_sejourns = sejourn_values[statio_idx:] if sejourn_values else []
    mean_sejourn = mean(statio_sejourns) if statio_sejourns else 0
    
    work_proportions = [wt / time_limit for wt in state.work_time]
    
    return {
        'mean_sejourn': mean_sejourn,
        'work_proportions': work_proportions,
        'steady_state_time': statio_time,
        'steady_state_index': statio_idx,
        'sejourn_records': state.sejourn_records,
        'work_time': state.work_time,
        'n_records': len(state.sejourn_records)
    }

# Fonction pour afficher les résultats de manière lisible
def print_results(instance_name, results):
    print("\n" + "=" * 56)
    print(f"  Instance : {instance_name}")
    print("=" * 56)
    print(f"  État stationnaire : {results['steady_state_time']:.2f}")
    print(f"  Produits (Stationnaires/Total) : {results['n_records'] - results['steady_state_index']} / {results['n_records']}")
    print(f"  Temps de séjour moyen : {results['mean_sejourn']:.4f}")
    print("  Proportion de temps de travail par employé:")
    for i, prop in enumerate(results['work_proportions']):
        print(f"    Employé {i+1} : {100*prop:5.1f}%")


# Main
def main():
    time_limit_full = 500000.0
    
    print("--- Instance I1 ---")
    res_I1 = run_simulation(QI1, time_limit=time_limit_full, seed=1)
    print_results("I1 (q=4, specialise)", res_I1)
    
    print("--- Instance I2 ---")
    res_I2 = run_simulation(QI2, time_limit=time_limit_full, seed=2)
    print_results("I2 (q=4, polyvalent)", res_I2)
    
    print("--- Instance I3 ---")
    res_I3 = run_simulation(QI3, time_limit=time_limit_full, seed=3)
    print_results("I3 (q=6, specialise)", res_I3)
    
    print("--- Instance I4 ---")
    res_I4 = run_simulation(QI4, time_limit=time_limit_full, seed=4)
    print_results("I4 (q=6, polyvalent)", res_I4)

if __name__ == "__main__":
    main()
