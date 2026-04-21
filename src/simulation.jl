# ============================================================
#  simulation.jl  —  API pública del paquete
# ============================================================

"""
    run_simulation(data, p; verbose=true) -> RunResult

Ejecuta una única corrida del GRASP GA/EDA y devuelve el `RunResult`
completo con:
- mejor solución y costo
- historial de costos por iteración
- traza de todas las soluciones evaluadas con ACOPF (factibles +
  infactibles que pasaron el prefiltro), con origen y si mejoraron el best.

# Argumentos
- `data`: diccionario del caso de red (salida de `ACOPF_Extensions.setup_case`)
- `p`:    `GRASPParameters` (usa `default_params()` si no se especifica)

# Ejemplo
```julia
using GraspTEP
data   = ACOPF_Extensions.setup_case("path/to/case", false, false)
params = GRASPParameters(maxIter=100, indv=10, N_elite=5)
result = run_simulation(data, params)
println("Mejor costo: ", result.best_cost)
```
"""
function run_simulation(data::Dict,
                         p::GRASPParameters = default_params();
                         verbose::Bool = true)::RunResult

    verbose && show_params(p)
    result = grasp_evolutivo(p, deepcopy(data))
    if verbose
        println("\n── Resultado ──────────────────────────────")
        println("  Mejor costo : ", round(result.best_cost, digits=3))
        println("  Iteración   : ", result.best_iter)
        println("  Evaluaciones: ", result.best_n_eval, " / ", result.total_evals)
        println("  Tiempo      : ", round(result.elapsed_seconds, digits=1), " s")
        println("  Traza       : ", length(result.trace), " soluciones registradas")
        println("───────────────────────────────────────────")
    end
    return result
end

"""
    run_simulations(data, p, n_runs; verbose=true, save_path=nothing) -> SimulationResult

Ejecuta `n_runs` corridas independientes y agrega los resultados.

# Argumentos
- `data`:       diccionario del caso
- `p`:          `GRASPParameters`
- `n_runs`:     número de corridas independientes
- `verbose`:    imprime resumen por corrida y resumen final
- `save_path`:  si se especifica, guarda el `SimulationResult` en un `.jld2`

# Retorno
`SimulationResult` con:
- `runs`:        vector de `RunResult` individuales
- `best_cost`:   mejor costo global entre todas las corridas
- `best_sol`:    mejor topología global
- `mean_cost`:   media de los mejores costos por corrida
- `std_cost`:    desviación estándar
- `mean_evals`:  media de evaluaciones ACOPF por corrida
- `mean_time`:   media de tiempo por corrida
- `hit_rate`:    porcentaje de corridas que encontraron el mejor global

# Ejemplo
```julia
sim = run_simulations(data, params, 10, save_path="results/caso118.jld2")
println("Hit rate: ", sim.hit_rate, "%")

# acceder a la traza de la mejor corrida
best_run = sim.runs[argmin([r.best_cost for r in sim.runs])]
mejoras  = filter(r -> r.improved, best_run.trace)
```
"""
function run_simulations(data      :: Dict,
                           p        :: GRASPParameters,
                           n_runs   :: Int;
                           verbose  :: Bool = true,
                           save_path :: Union{String, Nothing} = nothing)::SimulationResult

    verbose && show_params(p)
    runs = RunResult[]

    for i in 1:n_runs
        verbose && println("\n══ Corrida $i / $n_runs ══════════════════════")
        result = grasp_evolutivo(p, deepcopy(data))
        push!(runs, result)
        verbose && println("  → Costo: $(round(result.best_cost, digits=3)) | Evals: $(result.total_evals) | t: $(round(result.elapsed_seconds,digits=1))s")
    end

    # ---- estadísticas agregadas ----
    costs      = [r.best_cost for r in runs]
    best_idx   = argmin(costs)
    best_cost  = costs[best_idx]
    best_sol   = runs[best_idx].best_sol
    hit_rate   = count(==(best_cost), costs) * 100.0 / n_runs

    sim = SimulationResult(
        runs,
        best_cost,
        best_sol,
        mean(costs),
        std(costs),
        mean([r.total_evals for r in runs]),
        mean([r.elapsed_seconds for r in runs]),
        hit_rate
    )

    if verbose
        println("\n══ Resumen ($n_runs corridas) ══════════════════")
        println("  Mejor costo global : ", round(best_cost, digits=3))
        println("  Media ± std        : ", round(sim.mean_cost, digits=3),
                " ± ", round(sim.std_cost, digits=3))
        println("  Media evaluaciones : ", round(sim.mean_evals, digits=1))
        println("  Media tiempo       : ", round(sim.mean_time, digits=1), " s")
        println("  Hit rate           : ", round(hit_rate, digits=1), "%")
        println("═══════════════════════════════════════════════")
    end

    # ---- guardar si se pidió ----
    if !isnothing(save_path)
        FileIO.save(save_path,
            "sim",    sim,
            "params", p,
            "costs",  costs
        )
        verbose && println("  Guardado en: $save_path")
    end

    return sim
end

# ---- Utilidades de análisis sobre RunResult ----

"""
    improvement_trace(result::RunResult) -> Vector{SolutionRecord}

Filtra la traza para devolver solo las soluciones que mejoraron
el mejor global en el momento en que fueron evaluadas.
Útil para reconstruir el camino al óptimo.
"""
improvement_trace(r::RunResult) = filter(s -> s.improved, r.trace)

"""
    trace_by_origin(result::RunResult) -> Dict{Symbol, Vector{SolutionRecord}}

Agrupa la traza por operador de origen.
Permite analizar qué operador fue más efectivo.
"""
function trace_by_origin(r::RunResult)::Dict{Symbol, Vector{SolutionRecord}}
    d = Dict{Symbol, Vector{SolutionRecord}}()
    for rec in r.trace
        push!(get!(d, rec.origin, SolutionRecord[]), rec)
    end
    return d
end

"""
    operator_stats(result::RunResult) -> DataFrame

Devuelve un DataFrame con estadísticas por operador:
evaluaciones, mejoras, tasa de éxito, costo medio.
"""
function operator_stats(r::RunResult)
    by_op = trace_by_origin(r)
    rows  = []
    for (op, recs) in by_op
        total    = length(recs)
        improved = count(x -> x.improved, recs)
        feasible = count(x -> x.feasible, recs)
        costs    = [x.cost for x in recs if x.feasible]
        push!(rows, (
            operator   = string(op),
            evaluations = total,
            improvements = improved,
            feasibles  = feasible,
            hit_rate   = round(improved * 100.0 / max(total, 1), digits=1),
            mean_cost  = isempty(costs) ? NaN : round(mean(costs), digits=3)
        ))
    end
    return DataFrame(rows)
end
