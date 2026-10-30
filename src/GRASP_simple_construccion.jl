# ============================================================
#  GRASP_simple_construccion.jl
#  Fase constructiva EDA y Beam Search por stage
#  Reutiliza: cost_top, generate_solution_id de auxiliares.jl
# ============================================================

const MAX_LINES_S = 2

# ---- Construcción EDA estándar ----

"""
    construir_solucion_eda(average, best_sol_gen, p, sz) -> Matrix{Int}

Construye una solución completa muestreando Poisson(average) por celda.
Con probabilidad alpha construye desde cero; si no, parte del pool.
"""
function construir_solucion_eda(average      :: Matrix{Float64},
                                 best_sol_gen :: Vector{Tuple{Matrix{Int}, Float64}},
                                 p            :: GRASPSimpleParameters,
                                 sz           :: Tuple{Int,Int})::Matrix{Int}
    if rand() < p.alpha || isempty(best_sol_gen)
        return [min(rand(Poisson(max(average[j, k], 0.01))), MAX_LINES_S)
                for j in 1:sz[1], k in 1:sz[2]]
    else
        base = rand(best_sol_gen)[1]
        # perturba levemente la solución base con Poisson
        return [min(rand(Poisson(max(0.5 * average[j,k] + 0.5 * base[j,k], 0.01))), MAX_LINES_S)
                for j in 1:sz[1], k in 1:sz[2]]
    end
end

# ---- Beam Search por stage ----

"""
    beam_search_construct(average, data, p) -> Vector{Matrix{Int}}

Construye soluciones stage por stage manteniendo un beam de las mejores
soluciones parciales evaluadas por costo de inversión (barato, sin ACOPF).

Por cada stage:
  1. Extiende cada solución del beam con beam_samples variantes
  2. Evalúa costo de inversión acumulado (sin ACOPF)
  3. Conserva las beam_width mejores

Solo las beam_width soluciones finales se evalúan con ACOPF.
Esto reduce drásticamente el número de evaluaciones ACOPF.
"""
function beam_search_construct(average :: Matrix{Float64},
                                data    :: Dict,
                                p       :: GRASPSimpleParameters)::Vector{Matrix{Int}}

    n_lines  = size(average, 1)
    n_stages = size(average, 2)

    # inicializar beam con soluciones vacías
    beam = [zeros(Int, n_lines, n_stages) for _ in 1:p.beam_width]

    for stage in 1:n_stages
        candidates = Matrix{Int}[]

        for sol_partial in beam
            # generar beam_samples extensiones para este stage
            for _ in 1:p.beam_samples
                new_sol = copy(sol_partial)
                # rellenar solo el stage actual con Poisson(average)
                for line in 1:n_lines
                    new_sol[line, stage] = min(
                        rand(Poisson(max(average[line, stage], 0.01))),
                        MAX_LINES_S
                    )
                end
                push!(candidates, new_sol)
            end
        end

        # evaluar candidatos por costo de inversión acumulado (sin ACOPF)
        inv_costs = [cost_top(c, data) for c in candidates]

        # filtrar duplicados por ID
        seen     = Set{String}()
        filtered = Tuple{Matrix{Int}, Float64}[]
        for (c, ic) in zip(candidates, inv_costs)
            id = generate_solution_id(c)
            if id ∉ seen
                push!(seen, id)
                push!(filtered, (c, ic))
            end
        end

        # conservar beam_width mejores por costo de inversión
        sort!(filtered, by = x -> x[2])
        beam = [f[1] for f in filtered[1:min(p.beam_width, length(filtered))]]
    end

    return beam   # beam_width soluciones completas listas para ACOPF
end

# ---- Generación de candidatos por iteración ----

"""
    generar_candidatos(average, best_sol_gen, data, p) 
    -> Vector{Matrix{Int}}

Combina construcción EDA estándar y beam search para generar
los candidatos de la iteración. La mitad viene de EDA, la otra
mitad del beam search. Así se balancea diversidad e intensificación.
"""
function generar_candidatos(average      :: Matrix{Float64},
                             best_sol_gen :: Vector{Tuple{Matrix{Int}, Float64}},
                             data         :: Dict,
                             p            :: GRASPSimpleParameters)::Vector{Matrix{Int}}

    sz = size(average)

    # mitad EDA puro
    n_eda  = max(1, p.indv ÷ 2)
    n_beam = p.indv - n_eda

    eda_sols  = [construir_solucion_eda(average, best_sol_gen, p, sz)
                 for _ in 1:n_eda]

    # beam search genera beam_width soluciones — tomar las primeras n_beam
    beam_sols = if n_beam > 0
        bs = beam_search_construct(average, data, p)
        bs[1:min(n_beam, length(bs))]
    else
        Matrix{Int}[]
    end

    return vcat(eda_sols, beam_sols)
end
