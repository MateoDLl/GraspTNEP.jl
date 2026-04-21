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
