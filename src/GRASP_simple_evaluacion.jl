# ============================================================
#  GRASP_simple_evaluacion.jl
#  Evaluación de soluciones — reutiliza generate_solution_id,
#  cost_top y feasible2int de los archivos originales
# ============================================================

"""
    evaluate_solutions_simple(data, solutions, solutions_dict,
                               solutions_dict_f, best_inv_cost, p)
    -> (solutions_dict, solutions_dict_f, costs, feas, n_evals)

Evalúa un vector de soluciones con prefiltro por costo de inversión
y cache. Compatible con el diccionario de soluciones del GRASP original
— pueden compartir el mismo cache si se desea.

Penaliza infactibles con (cost_fo + 1) * 1000 para que no entren
al average ni al pool de factibles.
"""
function evaluate_solutions_simple(data             :: Dict,
                                    solutions        :: Vector{Matrix{Int}},
                                    solutions_dict   :: Dict,
                                    solutions_dict_f :: Dict,
                                    best_inv_cost    :: Float64,
                                    p                :: GRASPSimpleParameters
                                    )::Tuple{Dict, Dict, Vector{Float64}, Vector{Float64}, Int}

    n       = length(solutions)
    cost_FO = fill(10.0^39, n)
    feas    = zeros(Float64, n)
    n_evals = 0

    # umbral del prefiltro: permisivo si aún no hay referencia
    filter_thr = isinf(best_inv_cost) ? Inf : p.filter_factor * best_inv_cost

    for j in 1:n
        sol    = solutions[j]
        id_top = generate_solution_id(sol)

        # cache hit
        if haskey(solutions_dict, id_top)
            cost_FO[j] = solutions_dict[id_top][1]
            feas[j]    = solutions_dict[id_top][2]
            continue
        end

        # prefiltro por costo de inversión (barato)
        c_inv = cost_top(sol, data)
        if c_inv > filter_thr
            solutions_dict[id_top] = Dict(1 => 10.0^39, 2 => 0.0)
            cost_FO[j] = 10.0^39
            feas[j]    = 0.0
            continue
        end

        # evaluación ACOPF
        _, cost_fo, feasible, _ = ACOPF_Extensions.run_acopf_topology(data, sol)
        feass   = feasible2int(feasible)
        cost_fo = feass == 0.0 ? round((cost_fo + 1) * 1000, digits=3) :
                                  round(cost_fo, digits=3)
        n_evals += 1

        solutions_dict[id_top] = Dict(1 => cost_fo, 2 => feass)
        if feass == 1.0
            solutions_dict_f[2][id_top] = solutions_dict[id_top]
        end

        cost_FO[j] = cost_fo
        feas[j]    = feass
    end

    return solutions_dict, solutions_dict_f, cost_FO, feas, n_evals
end
