# ============================================================
#  grasp.jl  —  bucle principal GRASP GA/EDA
# ============================================================

"""
    grasp_evolutivo(p, data) -> RunResult

Ejecuta el GRASP GA/EDA con tres niveles de respuesta al estancamiento.
Devuelve un `RunResult` con la mejor solución, historial de costos
y traza completa de soluciones evaluadas.
"""
function grasp_evolutivo(p::GRASPParameters, data::Dict)::RunResult

    # ---- inicialización ----
    data["Inicial_Top"] = Int.(data["Inicial_Top"])
    sz = size(data["Inicial_Top"])

    elite        = Tuple{Matrix{Int}, Float64}[]
    elite_costs  = Float64[]
    best_sol_gen = Tuple{Matrix{Int}, Float64}[]

    solutions_dict   = Dict{String, Dict{Int, Float64}}()
    solutions_dict_f = Dict(1 => data["Inicial_Top"],
                             2 => Dict{String, Dict{Int, Float64}}())

    best_solution = data["Inicial_Top"]
    best_cost     = Inf
    data["cost_top"] = Inf 
    data["n_lines_max"] = Inf

    # EDA average
    average   = ones(sz) .* 0.7
    freq      = zeros(Float64, sz)   # contador de frecuencia por celda

    # contadores
    no_improve   = 0
    total_evals  = 0
    best_n_eval  = 0

    # umbrales de estancamiento
    thr1 = round(Int, p.N_Iter_No * p.lvl1_ratio)
    thr2 = round(Int, p.N_Iter_No * p.lvl2_ratio)

    # historial y traza
    cost_history = Float64[]
    trace        = SolutionRecord[]

    t_start = time()

    for iter in 1:p.maxIter

        # ---- prefiltro adaptativo ----
        filter_thr = isinf(data["cost_top"]) ? Inf :
                     adaptive_filter(iter, p.maxIter, data["cost_top"], p)
            # DEBUG — quitar después
        println("=== Iter $iter ===")
        println("  filter_thr = $filter_thr")
        # ==============================================
        # 1. FASE CONSTRUCTIVA (EDA) con oversample
        # ==============================================
        constructed = constructive_phase(p, data, average, best_sol_gen)

        solutions_dict, solutions_dict_f, c_costs, c_feas, n_ev, new_recs =
            evaluate_solutions(data, constructed, solutions_dict, solutions_dict_f,
                            filter_thr, iter, :construction,
                            p.trace_infeas_rate, best_cost)
        total_evals += n_ev
        append!(trace, new_recs)

        # ordenar y quedarse con los mejores indv — igual que versión original
        ord        = sortperm(c_costs)
        ord        = ord[1:p.indv]
        c_costs    = c_costs[ord]
        c_feas     = c_feas[ord]
        constructed = constructed[ord]

        # ==============================================
        # 2. OPERADORES GA (cruce + mutación)
        # ==============================================
        # usamos las soluciones construidas como local_sols para el cruce
        offspring, ga_origins = generate_offspring(p, data, average, elite,
                                                    best_sol_gen, constructed)

        solutions_dict, solutions_dict_f, ga_costs, ga_feas, n_ev2, ga_recs =
            evaluate_solutions(data, offspring, solutions_dict, solutions_dict_f,
                               filter_thr, iter, :ga_crossover,
                               p.trace_infeas_rate, best_cost)
        # corregir origins reales (pueden ser :ga_crossover o :ga_mutation)
        for (rec, orig) in zip(ga_recs, ga_origins)
            push!(trace, SolutionRecord(rec.sol, rec.cost, rec.feasible,
                                         rec.iter, orig, rec.improved))
        end
        total_evals += n_ev2

        # unir construcción + GA
        all_sols  = vcat(constructed, offspring)
        all_costs = vcat(c_costs, ga_costs)
        all_feas  = vcat(c_feas, ga_feas)

        # ==============================================
        # 2b. BÚSQUEDA LOCAL LIGERA sobre top-k hijos GA
        # ==============================================
        solutions_dict, solutions_dict_f, offspring, ga_costs, total_evals =
            light_local_search(offspring, ga_costs, ga_feas, data,
                            solutions_dict, solutions_dict_f,
                            filter_thr, iter, p, total_evals, best_cost, trace)
        # ==============================================
        # 3. LNS CONTINUO (sobre élite)
        # ==============================================
        lns_cands, lns_origins = generate_lns_candidates(elite, average, data, p)

        if !isempty(lns_cands)
            solutions_dict, solutions_dict_f, lns_costs, lns_feas, n_ev3, lns_recs =
                evaluate_solutions(data, lns_cands, solutions_dict, solutions_dict_f,
                                   filter_thr, iter, :lns,
                                   p.trace_infeas_rate, best_cost)
            total_evals  += n_ev3
            append!(trace, lns_recs)
            all_sols  = vcat(all_sols, lns_cands)
            all_costs = vcat(all_costs, lns_costs)
            all_feas  = vcat(all_feas, lns_feas)
        end

        # ==============================================
        # 4. PATH RELINKING (intensificación continua)
        # ==============================================
        pr_cands, _ = generate_pr_candidates(elite, offspring, p)

        if !isempty(pr_cands)
            solutions_dict, solutions_dict_f, pr_costs, pr_feas, n_ev4, pr_recs =
                evaluate_solutions(data, pr_cands, solutions_dict, solutions_dict_f,
                                   filter_thr, iter, :path_relinking,
                                   p.trace_infeas_rate, best_cost)
            total_evals  += n_ev4
            append!(trace, pr_recs)
            all_sols  = vcat(all_sols, pr_cands)
            all_costs = vcat(all_costs, pr_costs)
            all_feas  = vcat(all_feas, pr_feas)
        end

        # ==============================================
        # 5. RESPUESTA AL ESTANCAMIENTO
        # ==============================================

        # Nivel 1: ILS + pesos exponenciales (se maneja en EDA, aquí el ILS)
        if no_improve >= thr1 && !isinf(best_cost)
            ils_sol, _ = apply_ils(best_solution, best_sol_gen, p, data)
            solutions_dict, solutions_dict_f, ils_costs, ils_feas, n_ev5, ils_recs =
                evaluate_solutions(data, [ils_sol], solutions_dict, solutions_dict_f,
                                   filter_thr, iter, :ils,
                                   p.trace_infeas_rate, best_cost)
            total_evals += n_ev5
            append!(trace, ils_recs)
            all_sols  = vcat(all_sols, [ils_sol])
            all_costs = vcat(all_costs, ils_costs)
            all_feas  = vcat(all_feas, ils_feas)
        end

        # Nivel 2: VNS muestreado
        if no_improve >= thr2 && !isinf(best_cost)
            vns_cands, _ = apply_vns(best_solution, p, data)
            if !isempty(vns_cands)
                solutions_dict, solutions_dict_f, vns_costs, vns_feas, n_ev6, vns_recs =
                    evaluate_solutions(data, vns_cands, solutions_dict, solutions_dict_f,
                                       filter_thr, iter, :vns,
                                       p.trace_infeas_rate, best_cost)
                total_evals += n_ev6
                append!(trace, vns_recs)
                all_sols  = vcat(all_sols, vns_cands)
                all_costs = vcat(all_costs, vns_costs)
                all_feas  = vcat(all_feas, vns_feas)
            end
        end

        # ==============================================
        # 6. ACTUALIZAR ÉLITE Y best_sol_gen
        # ==============================================
        for (sol, cost, feas) in zip(all_sols, all_costs, all_feas)
            # actualizar élite
            feas == 1.0 || continue
            if length(elite) < p.N_elite || cost < elite_costs[end]
                insert_sorted!(elite, elite_costs, (sol, cost))
                if length(elite) > p.N_elite
                    pop!(elite); pop!(elite_costs)
                end
            end
            # pool de factibles
            if feas == 1.0
                push!(best_sol_gen, (sol, cost))
            end
        end

        unique!(best_sol_gen)
        if length(best_sol_gen) > p.N_elite * 2
            best_sol_gen = sort(best_sol_gen, by = x -> x[2])[1:p.N_elite*2]
        end

        # ==============================================
        # 7. ACTUALIZAR MEJOR GLOBAL
        # ==============================================
        iter_best_cost = isempty(all_costs) ? Inf : minimum(
            cost for (cost, f) in zip(all_costs, all_feas) if f == 1.0;
            init = Inf
        )
        if iter_best_cost < best_cost
            best_cost      = iter_best_cost
            best_solution  = all_sols[argmin(all_costs)]
            best_n_eval    = total_evals
            no_improve     = 0
            data["cost_top"] = (best_solution.*data["Mat_cost"]*data["Discount"]) |> sum
            data["n_lines_max"] = sum(best_solution) + 2
        else
            no_improve += 1
        end

        push!(cost_history, best_cost)

        # ==============================================
        # 8. ACTUALIZAR FRECUENCIA Y AVERAGE (EDA)
        # ==============================================
        factible_sols = [sol for (sol, _, feas) in
                         zip(all_sols, all_costs, all_feas) if feas == 1.0]
        update_freq!(freq, factible_sols)

        update_average_eda!(average, elite, no_improve, p, freq, iter)

        # ==============================================
        # 9. NIVEL 3: REINICIO PARCIAL
        # ==============================================
        if no_improve >= p.N_Iter_No
            restart_elite!(elite, elite_costs, best_sol_gen, freq, average, iter, p)
            no_improve = 0
        end

        println("Iter $iter | best=$best_cost | no_improve=$no_improve | evals=$total_evals")
    end

    elapsed = time() - t_start

    # mejor costo final desde history (por si best_cost quedó en Inf por falta de factibles)
    final_best_cost = isinf(best_cost) ? get(cost_history, p.maxIter, Inf) : best_cost

    return RunResult(
        final_best_cost,
        best_solution,
        findfirst(==(final_best_cost), cost_history) |> x -> isnothing(x) ? 0 : x,
        best_n_eval,
        cost_history,
        total_evals,
        elapsed,
        trace
    )
end
