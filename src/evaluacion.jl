# ============================================================
#  evaluacion.jl  —  evaluación de soluciones
# ============================================================

"""
    evaluate_solutions(data, solutions, solutions_dict, solutions_dict_f,
                       filter_threshold, iter, trace, origin) 
    -> (solutions_dict, solutions_dict_f, costs, feas, n_evals, new_records)

Evalúa un vector de soluciones usando ACOPF con cache y prefiltro adaptativo.

- `filter_threshold`: umbral de costo de inversión para pasar a ACOPF
  (calculado con `adaptive_filter` en cada iteración).
- `origin`: símbolo que identifica el operador que generó las soluciones
  (`:construction`, `:ga_crossover`, etc.) para el SolutionTrace.
- Devuelve además `new_records`: vector de SolutionRecord con las soluciones
  que pasaron el prefiltro (factibles e infactibles evaluadas con ACOPF).
"""
function evaluate_solutions(data         :: Dict,
                             solutions    :: Vector{Matrix{Int}},
                             solutions_dict   :: Dict,
                             solutions_dict_f :: Dict,
                             filter_threshold :: Float64,
                             iter         :: Int,
                             origin       :: Symbol,
                             trace_infeas_rate :: Float64,
                             current_best :: Float64)::Tuple{Dict, Dict, Vector{Float64}, Vector{Float64}, Int, Vector{SolutionRecord}}
    n           = length(solutions)
    cost_FO     = fill(10.0^39, n)
    feas        = zeros(Float64, n)
    new_records = SolutionRecord[]

    function evaluate_single(sol::Matrix{Int})
        id_top = generate_solution_id(sol)

        # cache hit
        if haskey(solutions_dict, id_top)
            return [solutions_dict[id_top][1], solutions_dict[id_top][2], id_top, 0]
        end

        # prefiltro adaptativo
        c_inv = cost_top(sol, data)
        if c_inv > filter_threshold || sum(sol) > data["n_lines_max"]
            solutions_dict[id_top] = Dict(1 => 10.0^39, 2 => 0.0)
            return [10.0^39, 0.0, id_top, 0]
        end

        # evaluación cara ACOPF
        _, cost_fo, feasible, _ = ACOPF_Extensions.run_acopf_topology(data, sol)
        feass = feasible2int(feasible)
        if feass == 0.0
            cost_fo = round((cost_fo+1)*1000, digits=3)
        else
            cost_fo = round(cost_fo, digits=3)
        end
        return [cost_fo, feass, id_top, 1]
    end

    results = pmap(evaluate_single, solutions)

    n_evals = 0
    for j in 1:n
        cost_FO[j]  = results[j][1]
        feas[j]     = results[j][2]
        id_top      = results[j][3]
        evaluated   = results[j][4] == 1
        n_evals    += results[j][4]

        solutions_dict[id_top] = Dict(1 => results[j][1], 2 => results[j][2])
        if feas[j] == 1.0
            solutions_dict_f[2][id_top] = solutions_dict[id_top]
        end

        # SolutionTrace: guarda factibles siempre; infactibles evaluadas con ACOPF
        # según trace_infeas_rate
        if evaluated
            is_feasible = feas[j] == 1.0
            if is_feasible || rand() < trace_infeas_rate
                improved = cost_FO[j] < current_best
                push!(new_records, SolutionRecord(
                    copy(solutions[j]),
                    cost_FO[j],
                    is_feasible,
                    iter,
                    origin,
                    improved
                ))
            end
        end
    end

    return solutions_dict, solutions_dict_f, cost_FO, feas, n_evals, new_records
end

function feasible2int(feasible::MathOptInterface.TerminationStatusCode)::Float16
    return Int(feasible) in [1, 4, 7, 9] ? 1.0 : 0.0
end
