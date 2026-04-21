# ============================================================
#  parameters.jl  —  parámetros del algoritmo
# ============================================================

"""
    GRASPParameters

Parámetros completos del GRASP GA/EDA con tres niveles de respuesta
al estancamiento, LNS e ILS como intensificación secundaria.
"""
Base.@kwdef struct GRASPParameters
    # --- Generales ---
    maxIter          :: Int     = 100     # máximo de iteraciones
    indv             :: Int     = 10      # individuos por iteración (sin el ×1.5 anterior)

    # --- EDA ---
    alpha            :: Float64 = 0.679   # prob. de construir desde cero vs best_sol_gen
    rho              :: Float64 = 0.2     # tasa de actualización del average (learning rate EDA)
    top_k            :: Int     = 3       # cuántas soluciones élite se usan para actualizar average

    # --- Élite ---
    N_elite          :: Int     = 5       # tamaño del conjunto élite

    # --- GA ---
    p_crossover      :: Float64 = 0.7     # probabilidad de aplicar cruce por individuo
    p_mutation       :: Float64 = 0.3     # probabilidad de aplicar mutación por individuo
    p_mut_cell       :: Float64 = 0.15    # fracción de celdas no-cero a mutar por solución

    # --- Prefiltro adaptativo ---
    filter_start     :: Float64 = 2.0     # multiplicador inicial  (iter=1)
    filter_end       :: Float64 = 1.2     # multiplicador final    (iter=maxIter)

    # --- Path relinking ---
    elite_local      :: Float64 = 0.564   # prob. de usar local_sol vs élite en PR
    min_hamming      :: Float64 = 0.1     # diversidad mínima del élite para activar PR
                                          # (fracción de celdas distintas promedio)

    # --- LNS continuo ---
    p_lns            :: Float64 = 0.3     # probabilidad de aplicar LNS por iteración
    lns_destroy_frac :: Float64 = 0.3     # fracción de celdas no-cero a destruir

    # --- Tres niveles de estancamiento ---
    N_Iter_No        :: Int     = 19      # umbral base (nivel 3 = reinicio)
    # nivel 1 = N_Iter_No * lvl1_ratio  → ILS + pesos exponenciales
    # nivel 2 = N_Iter_No * lvl2_ratio  → VNS + perturbación average
    lvl1_ratio       :: Float64 = 0.4
    lvl2_ratio       :: Float64 = 0.7

    # --- Pesos exponenciales (nivel 1+) ---
    beta_base        :: Float64 = 1.0     # β base para exp(-β·cost/best_cost)
    beta_max         :: Float64 = 5.0     # β máximo cuando no_improve → umbral2

    # --- ILS (nivel 1) ---
    ils_perturb_frac :: Float64 = 0.2     # fracción de celdas a perturbar en ILS

    # --- VNS (nivel 2) ---
    vns_n_samples    :: Int     = 5       # muestras aleatorias por vecindario en VNS

    # --- Reinicio (nivel 3) ---
    porcentaje_elite :: Float64 = 0.3     # fracción del élite que se conserva al reiniciar
    penalty_factor   :: Float64 = 0.8     # factor de penalización en average para celdas sobreusadas

    # --- Traza ---
    trace_infeas_rate :: Float64 = 1.0    # fracción de infactibles (que pasaron prefiltro) a guardar
                                          # 1.0 = todas, 0.5 = mitad aleatoria
end

"""
    default_params()

Devuelve los parámetros por defecto listos para usar.
"""
default_params() = GRASPParameters()

"""
    show_params(p::GRASPParameters)

Imprime un resumen legible de los parámetros.
"""
function show_params(p::GRASPParameters)
    println("=" ^ 50)
    println("GRASPParameters")
    println("=" ^ 50)
    println("  General:      maxIter=$(p.maxIter), indv=$(p.indv)")
    println("  EDA:          alpha=$(p.alpha), rho=$(p.rho), top_k=$(p.top_k)")
    println("  Élite:        N_elite=$(p.N_elite)")
    println("  GA:           p_cross=$(p.p_crossover), p_mut=$(p.p_mutation), p_mut_cell=$(p.p_mut_cell)")
    println("  Prefiltro:    $(p.filter_start)→$(p.filter_end) × best_cost")
    println("  PR:           elite_local=$(p.elite_local), min_hamming=$(p.min_hamming)")
    println("  LNS:          p_lns=$(p.p_lns), destroy_frac=$(p.lns_destroy_frac)")
    println("  Estancamiento: N_Iter_No=$(p.N_Iter_No) | lvl1@$(round(Int,p.N_Iter_No*p.lvl1_ratio)) | lvl2@$(round(Int,p.N_Iter_No*p.lvl2_ratio))")
    println("  β:            base=$(p.beta_base), max=$(p.beta_max)")
    println("=" ^ 50)
end
