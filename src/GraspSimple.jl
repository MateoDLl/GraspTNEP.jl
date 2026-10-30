# ============================================================
#  GraspSimple.jl
#  GRASP simple EDA + Beam Search
#  Sin búsqueda local, sin RL, sin GA, sin path relinking
#
#  Incluir en el módulo principal junto a los archivos originales:
#
#  include("GRASP_simple_parameters.jl")
#  include("GRASP_simple_construccion.jl")
#  include("GRASP_simple_eda.jl")
#  include("GRASP_simple_evaluacion.jl")
#  include("GraspSimple.jl")
#
#  Uso:
#  best_cost, best_sol, history = grasp_simple(params_simple, case_data)
# ============================================================

"""
    grasp_simple(p, data) -> (best_cost, best_sol, cost_history, n_evals_total)

GRASP simple EDA + Beam Search.

Por iteración:
  1. Genera candidatos: mitad EDA, mitad Beam Search por stage
  2. Evalúa con ACOPF (con prefiltro por inversión)
  3. Actualiza average con las factibles encontradas (top_k)
  4. Actualiza mejor global solo con factibles

Sin búsqueda local, sin RL, sin GA, sin path relinking.
Objetivo: encontrar soluciones factibles sub-óptimas rápidamente.
"""
function grasp_simple(p    :: GRASPSimpleParameters,
                       data :: Dict, semilla :: Int = 1111)::Tuple{Float64, Matrix{Int}, Vector{Float64}, Int}
    Random.seed!(semilla)
    # ---- inicialización ----
    data["Inicial_Top"] = Int.(data["Inicial_Top"])
    sz = size(data["Inicial_Top"])

    # average inicial conservador — arranca en 1.0 para no generar
    # topologías gigantes infactibles desde el inicio
    average = ones(Float64, sz)

    # pool de soluciones factibles encontradas
    best_sol_gen = Tuple{Matrix{Int}, Float64}[]

    # cache compartible con el GRASP original si se desea
    solutions_dict   = Dict{String, Dict{Int, Float64}}()
    solutions_dict_f = Dict(1 => data["Inicial_Top"],
                             2 => Dict{String, Dict{Int, Float64}}())

    # mejor solución global
    best_cost    = Inf
    best_sol     = data["Inicial_Top"]
    best_inv_cost = Inf    # mejor costo de inversión conocido (para prefiltro)

    # historial
    cost_history  = Float64[]
    total_evals   = 0
    no_improve    = 0

    println("=" ^ 50)
    println("GRASP Simple EDA + Beam Search")
    println("maxIter=$(p.maxIter) | indv=$(p.indv) | beam_width=$(p.beam_width)")
    println("=" ^ 50)

    for iter in 1:p.maxIter

        # ==============================================
        # 1. GENERAR CANDIDATOS (EDA + Beam Search)
        # ==============================================
        candidates = generar_candidatos(average, best_sol_gen, data, p)

        # ==============================================
        # 2. EVALUAR CON ACOPF
        # ==============================================
        solutions_dict, solutions_dict_f, costs, feas, n_ev =
            evaluate_solutions_simple(data, candidates, solutions_dict,
                                       solutions_dict_f, best_inv_cost, p)
        total_evals += n_ev

        # ==============================================
        # 3. ACTUALIZAR MEJOR GLOBAL (solo factibles)
        # ==============================================
        iter_best = Inf
        for (sol, cost, f) in zip(candidates, costs, feas)
            f == 1.0 || continue
            if cost < best_cost
                best_cost     = cost
                best_sol      = copy(sol)
                best_inv_cost = cost_top(sol, data)
                no_improve    = 0
            end
            iter_best = min(iter_best, cost)
            # agregar al pool de factibles
            push!(best_sol_gen, (copy(sol), cost))
        end

        isinf(iter_best) && (no_improve += 1)
        push!(cost_history, best_cost)

        # ==============================================
        # 4. ACTUALIZAR AVERAGE (solo con factibles)
        # ==============================================
        fact_sols  = [sol for (sol, f) in zip(candidates, feas) if f == 1.0]
        fact_costs = [c   for (c, f)   in zip(costs, feas)      if f == 1.0]

        update_average_simple!(average, fact_sols, fact_costs, p)

        # limpiar pool (evitar crecer indefinidamente)
        if length(best_sol_gen) > 20
            sort!(best_sol_gen, by = x -> x[2])
            best_sol_gen = best_sol_gen[1:20]
        end

        println("Iter $iter | best=$(round(best_cost, digits=3)) | " *
                "factibles=$(length(fact_sols))/$(length(candidates)) | " *
                "evals=$total_evals | no_improve=$no_improve")

        # parada anticipada si no hay progreso por muchas iteraciones
        if no_improve >= round(Int, p.maxIter * 0.4)
            println("  → Sin mejora en $(no_improve) iteraciones, deteniendo.")
            break
        end
    end

    println("=" ^ 50)
    println("Mejor costo : $(round(best_cost, digits=3))")
    println("Evaluaciones: $total_evals")
    println("=" ^ 50)

    return best_cost, best_sol, cost_history, total_evals
end
