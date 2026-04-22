# ============================================================
#  intensification.jl  —  LNS, ILS, VNS y Path Relinking
# ============================================================

# ---- LNS (Large Neighborhood Search) ----

"""
    lns_destroy(sol, p) -> Matrix{Int}

Destruye una fracción `p.lns_destroy_frac` de las celdas no-cero
de la solución, elegidas aleatoriamente. Devuelve la solución parcial.
"""
function lns_destroy(sol::Matrix{Int}, p::GRASPParameters)::Matrix{Int}
    destroyed   = copy(sol)
    nonzero_idx = findall(>(0), sol)
    n_destroy   = max(1, round(Int, length(nonzero_idx) * p.lns_destroy_frac))
    chosen      = sample(nonzero_idx, min(n_destroy, length(nonzero_idx)), replace=false)
    for idx in chosen
        destroyed[idx] = 0
    end
    return destroyed
end

"""
    lns_repair(partial_sol, average, data) -> Matrix{Int}

Reconstruye las celdas destruidas usando muestreo Poisson del average.
Las celdas no destruidas (> 0) se mantienen intactas.
"""
function lns_repair(partial_sol::Matrix{Int}, average::Matrix{Float64},
                     data::Dict)::Matrix{Int}
    repaired = copy(partial_sol)
    for idx in CartesianIndices(partial_sol)
        if partial_sol[idx] == 0
            repaired[idx] = min(rand(Poisson(max(average[idx], 0.01))), MAX_LINES)
        end
    end
    return repaired
end

"""
    apply_lns(sol, average, data, p) -> (Matrix{Int}, Symbol)

Aplica destroy + repair sobre una solución. Devuelve la nueva solución
y el símbolo de origen `:lns`.
"""
function apply_lns(sol::Matrix{Int}, average::Matrix{Float64},
                    data::Dict, p::GRASPParameters)::Tuple{Matrix{Int}, Symbol}
    partial  = lns_destroy(sol, p)
    repaired = lns_repair(partial, average, data)
    return repaired, :lns
end

"""
    generate_lns_candidates(elite, average, data, p)
    -> (Vector{Matrix{Int}}, Vector{Symbol})

Genera candidatos LNS aplicando destroy+repair sobre soluciones
del élite. Solo se aplica con probabilidad `p.p_lns`.
"""
function generate_lns_candidates(elite        :: Vector{Tuple{Matrix{Int}, Float64}},
                                   average      :: Matrix{Float64},
                                   data         :: Dict,
                                   p            :: GRASPParameters)::Tuple{Vector{Matrix{Int}}, Vector{Symbol}}
    candidates = Matrix{Int}[]
    origins    = Symbol[]
    isempty(elite) && return candidates, origins

    for (sol, _) in elite
        rand() < p.p_lns || continue
        new_sol, origin = apply_lns(sol, average, data, p)
        push!(candidates, new_sol)
        push!(origins, origin)
    end

    return candidates, origins
end

# ---- ILS (Iterated Local Search) — nivel 1 ----

"""
    apply_ils(best_sol, best_sol_gen, p, data) -> (Matrix{Int}, Symbol)

Perturba la mejor solución actual cambiando `p.ils_perturb_frac`
de sus celdas intercambiando valores con una solución aleatoria
del pool `best_sol_gen`. Escape puntual de óptimo local.
"""
function apply_ils(best_sol     :: Matrix{Int},
                    best_sol_gen :: Vector{Tuple{Matrix{Int}, Float64}},
                    p            :: GRASPParameters,
                    data         :: Dict)::Tuple{Matrix{Int}, Symbol}

    perturbed   = copy(best_sol)
    nonzero_idx = findall(>(0), best_sol)
    n_perturb   = max(1, round(Int, length(nonzero_idx) * p.ils_perturb_frac))
    chosen      = sample(nonzero_idx, min(n_perturb, length(nonzero_idx)), replace=false)

    if !isempty(best_sol_gen)
        donor = rand(best_sol_gen)[1]
        for idx in chosen
            perturbed[idx] = clamp(
                round(Int, (best_sol[idx] + donor[idx]) / 2) + rand(-1:1),
                0, MAX_LINES
            )
        end
    else
        # sin pool: perturbación aleatoria pura
        for idx in chosen
            perturbed[idx] = clamp(best_sol[idx] + rand(-1:1), 0, MAX_LINES)
        end
    end

    return perturbed, :ils
end

# ---- VNS (Variable Neighborhood Search) — nivel 2 ----

"""
    apply_vns(best_sol, p, data, n_eval_budget) -> (Vector{Matrix{Int}}, Vector{Symbol})

Genera candidatos explorando vecindarios N2 y N3 de la mejor solución
mediante muestreo aleatorio (no exhaustivo), respetando un presupuesto
de muestras `p.vns_n_samples` por vecindario.

- N2: cambia 2 celdas simultáneamente (±1 cada una)
- N3: cambia 3 celdas simultáneamente (±1 cada una)
"""
function apply_vns(best_sol :: Matrix{Int},
                    p        :: GRASPParameters,
                    data     :: Dict)::Tuple{Vector{Matrix{Int}}, Vector{Symbol}}

    candidates = Matrix{Int}[]
    origins    = Symbol[]
    nonzero    = findall(>(0), best_sol)
    isempty(nonzero) && return candidates, origins

    for neighborhood in (2, 3)
        for _ in 1:p.vns_n_samples
            new_sol = copy(best_sol)
            chosen  = sample(nonzero, min(neighborhood, length(nonzero)), replace=false)
            for idx in chosen
                new_sol[idx] = clamp(new_sol[idx] + rand(-1:1), 0, MAX_LINES)
            end
            push!(candidates, new_sol)
            push!(origins, :vns)
        end
    end

    return candidates, origins
end

# ---- Path Relinking ----

"""
    apply_path_relinking(sol1, sol2) -> Matrix{Int}

Interpola aleatoriamente entre sol1 y sol2.
"""
function apply_path_relinking(sol1::Matrix{Int}, sol2::Matrix{Int})::Matrix{Int}
    new_sol = copy(sol1)
    for idx in CartesianIndices(sol1)
        diff = sol1[idx] - sol2[idx]
        if diff > 0
            new_sol[idx] -= rand(0:diff)
        elseif diff < 0
            new_sol[idx] += rand(0:-diff)
        end
    end
    return new_sol
end

"""
    generate_pr_candidates(elite, local_sols, p)
    -> (Vector{Matrix{Int}}, Vector{Symbol})

Genera candidatos de path relinking entre pares del élite o entre
élite y hijos GA. Solo se activa si la diversidad Hamming del élite
supera `p.min_hamming`.
"""
function generate_pr_candidates(elite      :: Vector{Tuple{Matrix{Int}, Float64}},
                                  local_sols :: Vector{Matrix{Int}},
                                  p          :: GRASPParameters)::Tuple{Vector{Matrix{Int}}, Vector{Symbol}}

    candidates = Matrix{Int}[]
    origins    = Symbol[]

    length(elite) < 2 && return candidates, origins

    # control de diversidad: no activar PR si el élite convergió
    diversity = elite_hamming_diversity(elite)
    diversity < p.min_hamming && return candidates, origins

    n = min(p.indv, length(local_sols))
    for i in 1:n
        if rand() > p.elite_local
            idx1, idx2 = sample(1:length(elite), 2, replace=false)
            new_sol = apply_path_relinking(elite[idx1][1], elite[idx2][1])
        else
            elite_sol   = rand(elite)[1]
            current_sol = local_sols[i]
            new_sol     = apply_path_relinking(elite_sol, current_sol)
        end
        push!(candidates, new_sol)
        push!(origins, :path_relinking)
    end

    return candidates, origins
end

# ---- Reinicio nivel 3 ----

"""
    restart_elite!(elite, elite_costs, best_sol_gen, freq, average, p)

Reinicio parcial del élite:
- conserva `porcentaje_elite` del élite actual
- completa con soluciones aleatorias de `best_sol_gen`
- penaliza celdas sobreusadas en `average`
"""
function restart_elite!(elite        :: Vector{Tuple{Matrix{Int}, Float64}},
                         elite_costs  :: Vector{Float64},
                         best_sol_gen :: Vector{Tuple{Matrix{Int}, Float64}},
                         freq         :: Matrix{Float64},
                         average      :: Matrix{Float64},
                         iter         :: Int,
                         p            :: GRASPParameters)

    num_keep = min(round(Int, p.porcentaje_elite * p.N_elite), length(elite))
    kept     = num_keep > 0 ? sample(elite, num_keep, replace=false) : Tuple{Matrix{Int}, Float64}[]

    needed   = p.N_elite - length(kept)
    new_sols = if needed > 0 && !isempty(best_sol_gen)
        sample(best_sol_gen, min(needed, length(best_sol_gen)), replace=false)
    else
        Tuple{Matrix{Int}, Float64}[]
    end

    combined = vcat(kept, new_sols)
    if !isempty(combined)
        new_elite, new_costs = sort_elite(combined)
        resize!(elite, length(new_elite))
        resize!(elite_costs, length(new_costs))
        elite      .= new_elite
        elite_costs .= new_costs
    end

    # penalizar celdas sobreusadas en average
    over_mask = overused_mask(freq, iter, 0.8)
    average[over_mask] .*= p.penalty_factor
    average .= clamp.(average, 0.0, 5.0)
end

"""
    light_local_search(sols, costs, feas, data, solutions_dict, solutions_dict_f,
                       filter_thr, iter, p, total_evals, best_cost, trace)

Aplica un único paso first-improvement sobre los mejores k hijos factibles.
Prueba eliminar la línea de mayor costo de inversión — si mejora, acepta.
Máximo 1 evaluación ACOPF por individuo. No itera hasta convergencia.
"""
function light_local_search(sols             :: Vector{Matrix{Int}},
                              costs            :: Vector{Float64},
                              feas             :: Vector{Float64},
                              data             :: Dict,
                              solutions_dict   :: Dict,
                              solutions_dict_f :: Dict,
                              filter_thr       :: Float64,
                              iter             :: Int,
                              p                :: GRASPParameters,
                              total_evals      :: Int,
                              best_cost        :: Float64,
                              trace            :: Vector{SolutionRecord})

    best_sols  = deepcopy(sols)
    best_costs = copy(costs)
    n_ls       = min(p.n_ls_top, count(==(1.0), feas))   # solo sobre factibles top-k

    # ordenar por costo para trabajar sobre los mejores primero
    ord = sortperm(costs)

    checked = 0
    for id in ord
        feas[id] == 1.0 || continue
        checked >= n_ls && break
        checked += 1

        sol = sols[id]

        # encontrar la celda no-cero de mayor costo de inversión
        cost_matrix  = sol./max.(sol, 1) .* (data["Mat_cost"] * data["Discount"]')
        nonzero_idx  = findall(>(0), sol)
        isempty(nonzero_idx) && continue
        best_idx     = nonzero_idx[argmax([cost_matrix[i] for i in nonzero_idx])]

        # probar eliminar esa línea
        new_sol = copy(sol)
        new_sol[best_idx] -= 1

        id_top = generate_solution_id(new_sol)

        # usar cache si existe
        if haskey(solutions_dict, id_top)
            new_cost = solutions_dict[id_top][1]
            new_feas = solutions_dict[id_top][2]
        else
            # prefiltro barato
            c_inv = cost_top(new_sol, data)
            if c_inv > filter_thr
                continue
            end
            # evaluación ACOPF
            _, cost_fo, feasible, _ = ACOPF_Extensions.run_acopf_topology(data, new_sol)
            new_cost = round(cost_fo, digits=3)
            new_feas = feasible2int(feasible)
            if new_feas == 0.0
                cost_fo = round((cost_fo + 1) * 1000, digits=3)
            else
                cost_fo = round(cost_fo, digits=3)
            end
            solutions_dict[id_top]   = Dict(1 => cost_fo, 2 => new_feas)
            if new_feas == 1.0
                solutions_dict_f[2][id_top] = solutions_dict[id_top]
            end
            total_evals += 1

            # registrar en traza
            improved = cost_fo < best_cost
            push!(trace, SolutionRecord(copy(new_sol), cost_fo, new_feas == 1.0,
                                         iter, :light_ls, improved))
        end

        # aceptar si mejora
        if new_cost < best_costs[id]
            best_sols[id]  = new_sol
            best_costs[id] = new_cost
        end
    end

    return solutions_dict, solutions_dict_f, best_sols, best_costs, total_evals
end