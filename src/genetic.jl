# ============================================================
#  genetic.jl  —  operadores GA y fase constructiva EDA
# ============================================================

const MAX_LINES = 3

# ---- Fase constructiva EDA ----

"""
    constructive_phase(p, data, average, best_sol_gen) -> Vector{Matrix{Int}}

Genera exactamente `p.indv` soluciones usando EDA (Poisson por celda).
Con probabilidad `p.alpha` construye desde cero; con `1-alpha` parte
de una solución aleatoria del pool `best_sol_gen`.
"""
function constructive_phase(p            :: GRASPParameters,
                             data         :: Dict,
                             average      :: Matrix{Float64},
                             best_sol_gen :: Vector{Tuple{Matrix{Int}, Float64}})::Vector{Matrix{Int}}

    sz          = size(data["Inicial_Top"])
    n_construct = round(Int, p.indv * p.oversample_ratio)   # genera más candidatos
    candidates  = [zeros(Int, sz) for _ in 1:n_construct]

    for i in 1:n_construct
        if rand() < p.alpha || isempty(best_sol_gen)
            candidates[i] = [min(rand(Distributions.Poisson(average[j, k])), MAX_LINES)
                             for j in 1:sz[1], k in 1:sz[2]]
        else
            base = rand(best_sol_gen)[1]
            candidates[i] = [min(base[j, k], MAX_LINES)
                             for j in 1:sz[1], k in 1:sz[2]]
        end
    end

    return candidates   # retorna todos — el ordenamiento se hace en grasp.jl
                        # después de evaluar con ACOPF, igual que la versión original
end

# ---- Cruce (reutiliza apply_path_relinking) ----

"""
    apply_crossover(sol1, sol2) -> Matrix{Int}

Interpola aleatoriamente entre sol1 y sol2 celda a celda.
Es el mismo operador que apply_path_relinking — se reutiliza
como cruce GA con semántica explícita.
"""
function apply_crossover(sol1::Matrix{Int}, sol2::Matrix{Int})::Matrix{Int}
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

# ---- Mutación ----

"""
    apply_mutation(sol, p, Idx_cost_cap) -> Matrix{Int}

Muta una fracción `p.p_mut_cell` de las celdas no-cero de la solución.
El cambio es ±1, con sesgo hacia las celdas de mayor valor heurístico
(Idx_cost_cap). Celdas con valor 0 pueden activarse con probabilidad baja.
"""
function apply_mutation(sol          :: Matrix{Int},
                         p            :: GRASPParameters,
                         Idx_cost_cap :: Matrix{Float64})::Matrix{Int}
    new_sol    = copy(sol)
    nonzero_idx = findall(>(0), sol)
    n_mutate   = max(1, round(Int, length(nonzero_idx) * p.p_mut_cell))

    if !isempty(nonzero_idx)
        chosen = sample(nonzero_idx, min(n_mutate, length(nonzero_idx)), replace=false)
        for idx in chosen
            delta = rand((-1, 1))
            new_sol[idx] = clamp(new_sol[idx] + delta, 0, MAX_LINES)
        end
    end

    # con baja probabilidad activa una celda nueva (exploración)
    if rand() < 0.1
        zero_idx = findall(==(0), sol)
        if !isempty(zero_idx)
            # elige la celda cero con mayor valor heurístico
            best_zero = zero_idx[argmax([Idx_cost_cap[i] for i in zero_idx])]
            new_sol[best_zero] = 1
        end
    end

    return new_sol
end

# ---- Generación de hijos GA ----

"""
    generate_offspring(p, data, average, elite, best_sol_gen, local_sols)
    -> (Vector{Matrix{Int}}, Vector{Symbol})

Genera `p.indv` hijos combinando cruce y mutación.
Retorna las soluciones y el vector de origins correspondiente
(`:ga_crossover` o `:ga_mutation`) para el SolutionTrace.
"""
function generate_offspring(p            :: GRASPParameters,
                              data         :: Dict,
                              average      :: Matrix{Float64},
                              elite        :: Vector{Tuple{Matrix{Int}, Float64}},
                              best_sol_gen :: Vector{Tuple{Matrix{Int}, Float64}},
                              local_sols   :: Vector{Matrix{Int}})::Tuple{Vector{Matrix{Int}}, Vector{Symbol}}

    sz           = size(data["Inicial_Top"])
    offspring    = Vector{Matrix{Int}}(undef, p.indv)
    origins      = Vector{Symbol}(undef, p.indv)

    Idx_cost_cap = average .* (1.0 ./ max.(data["Mat_cost"], 1e-8))
    s = sum(Idx_cost_cap)
    Idx_cost_cap = s > 0 ? Idx_cost_cap ./ s : fill(1.0 / length(Idx_cost_cap), sz)

    pool = vcat(
        isempty(elite)        ? Matrix{Int}[] : [e[1] for e in elite],
        isempty(best_sol_gen) ? Matrix{Int}[] : [b[1] for b in best_sol_gen],
        local_sols
    )

    for i in 1:p.indv
        use_crossover = rand() < p.p_crossover && length(pool) >= 2

        if use_crossover
            idx1, idx2 = sample(1:length(pool), 2, replace=false)
            child = apply_crossover(pool[idx1], pool[idx2])
            # mutación adicional con probabilidad p_mutation
            if rand() < p.p_mutation
                child = apply_mutation(child, p, Idx_cost_cap)
            end
            offspring[i] = child
            origins[i]   = :ga_crossover
        else
            # solo mutación sobre una solución del pool
            base = isempty(pool) ? zeros(Int, sz) : rand(pool)
            offspring[i] = apply_mutation(base, p, Idx_cost_cap)
            origins[i]   = :ga_mutation
        end
    end

    return offspring, origins
end
