# ============================================================
#  types.jl  —  structs de resultados y trazabilidad
# ============================================================

"""
    SolutionRecord

Registro de una solución evaluada con ACOPF que pasó el prefiltro.
Guarda la topología, costo, factibilidad, iteración, operador que la generó,
y si mejoró el mejor global en ese momento. Útil para RL guiado futuro.
"""
struct SolutionRecord
    sol       :: Matrix{Int}     # topología evaluada
    cost      :: Float64         # costo ACOPF real
    feasible  :: Bool            # factibilidad
    iter      :: Int             # iteración en que fue evaluada
    origin    :: Symbol          # :construction | :ga_crossover | :ga_mutation
                                 # :path_relinking | :lns | :ils | :vns
    improved  :: Bool            # ¿mejoró el mejor global en ese momento?
end

"""
    RunResult

Resultado completo de una corrida del algoritmo.
Contiene la mejor solución, historial de convergencia y traza de soluciones.
"""
struct RunResult
    # Mejor solución encontrada
    best_cost        :: Float64
    best_sol         :: Matrix{Int}
    best_iter        :: Int             # iteración donde se encontró
    best_n_eval      :: Int             # evaluaciones ACOPF al momento de encontrarla

    # Historial por iteración
    cost_history     :: Vector{Float64} # costo del mejor en cada iteración

    # Evaluaciones totales
    total_evals      :: Int

    # Tiempo
    elapsed_seconds  :: Float64

    # Traza de soluciones para RL guiado
    # - todas las factibles evaluadas con ACOPF
    # - infactibles que pasaron el prefiltro (evaluadas con ACOPF pero infactibles)
    trace            :: Vector{SolutionRecord}
end

"""
    SimulationResult

Resultado agregado de múltiples corridas independientes.
"""
struct SimulationResult
    runs          :: Vector{RunResult}  # resultado individual de cada corrida

    # Estadísticas agregadas
    best_cost     :: Float64            # mejor costo global entre todas las corridas
    best_sol      :: Matrix{Int}        # mejor solución global
    mean_cost     :: Float64
    std_cost      :: Float64
    mean_evals    :: Float64
    mean_time     :: Float64
    hit_rate      :: Float64            # % de corridas que encontraron el mejor global
end
