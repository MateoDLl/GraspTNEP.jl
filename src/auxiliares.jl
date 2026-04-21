# ============================================================
#  auxiliares.jl  —  funciones auxiliares
# ============================================================

# ---- Élite ----

function insert_sorted!(
    elite  :: Vector{Tuple{Matrix{Int}, Float64}},
    costs  :: Vector{Float64},
    item   :: Tuple{Matrix{Int}, Float64}
)
    cost = item[2]
    idx  = searchsortedfirst(costs, cost)
    insert!(elite, idx, item)
    insert!(costs, idx, cost)
    @assert length(elite) == length(costs)
end

function sort_elite(elite::Vector{Tuple{Matrix{Int}, Float64}})
    sort!(elite, by = x -> x[2])
    costs = [x[2] for x in elite]
    return elite, costs
end

# ---- Costo de inversión (prefiltro barato) ----

function cost_top(sol::Matrix, data::Dict)::Float64
    cost = 0.0
    for st in 1:size(sol, 2)
        cost += sum(sol[:, st] .* data["Mat_cost"] .* data["Discount"][st])
    end
    return cost
end

# ---- Umbral adaptativo del prefiltro ----

"""
    adaptive_filter(iter, maxIter, best_cost, p) -> Float64

Devuelve el umbral de coste para el prefiltro. Decrece linealmente
desde `filter_start × best_cost` (iter=1) hasta `filter_end × best_cost`
(iter=maxIter). Así el filtro se vuelve más agresivo conforme avanza
la búsqueda.
"""
function adaptive_filter(iter::Int, maxIter::Int, best_cost::Float64,
                          p::GRASPParameters)::Float64
    t = (iter - 1) / max(maxIter - 1, 1)          # ∈ [0, 1]
    factor = p.filter_start + t * (p.filter_end - p.filter_start)
    return factor * best_cost
end

# ---- Selección ponderada ----

function weighted_random_selection(Idx::Matrix{Float64})
    cumulative_sum = cumsum(Idx, dims=1)
    val = rand()
    add = 0.0
    for i in 1:size(Idx, 2), j in 1:size(Idx, 1)
        if val <= cumulative_sum[j, i] + add
            return CartesianIndex(j, i)
        end
    end
    add = deepcopy(cumulative_sum[end, i])
    return rand(collect(CartesianIndices(Idx)))
end

# ---- ID de solución ----

function generate_solution_id(sol::Matrix{Int})::String
    io = IOBuffer()
    for idx in findall(>(0), sol)
        print(io, idx[1], '.', idx[2], '.', sol[idx], '.')
    end
    return String(take!(io))
end

# ---- Diversidad Hamming entre soluciones del élite ----

"""
    elite_hamming_diversity(elite) -> Float64

Calcula la fracción media de celdas distintas entre todos los pares
del élite normalizada por el total de celdas. Devuelve ∈ [0, 1].
"""
function elite_hamming_diversity(elite::Vector{Tuple{Matrix{Int}, Float64}})::Float64
    n = length(elite)
    n < 2 && return 0.0
    total_cells = length(elite[1][1])
    diffs = 0.0
    pairs = 0
    for i in 1:n, j in (i+1):n
        diffs += sum(elite[i][1] .!= elite[j][1])
        pairs += 1
    end
    return (diffs / pairs) / total_cells
end

# ---- Contador de frecuencia de celdas ----

"""
    update_freq!(freq, solutions)

Incrementa el contador de frecuencia por celda para cada solución
factible. Se usa para detectar celdas sobreusadas (nivel 3) y
celdas poco exploradas (nivel 2).
"""
function update_freq!(freq::Matrix{Float64}, solutions::Vector{Matrix{Int}})
    for sol in solutions
        freq .+= (sol .> 0)
    end
end

"""
    underused_mask(freq, iter, threshold) -> BitMatrix

Devuelve una máscara de celdas cuya frecuencia normalizada está
por debajo de `threshold`. Usada en nivel 2 para añadir ruido
dirigido al average.
"""
function underused_mask(freq::Matrix{Float64}, iter::Int,
                         threshold::Float64 = 0.2)::BitMatrix
    norm_freq = freq ./ max(iter, 1)
    return norm_freq .< threshold
end

"""
    overused_mask(freq, iter, threshold) -> BitMatrix

Devuelve una máscara de celdas cuya frecuencia normalizada supera
`threshold`. Usada en nivel 3 para penalizar el average.
"""
function overused_mask(freq::Matrix{Float64}, iter::Int,
                        threshold::Float64 = 0.8)::BitMatrix
    norm_freq = freq ./ max(iter, 1)
    return norm_freq .> threshold
end

# ---- Cálculo promedio estadístico (sin RL) ----

function calc_average_solutions(solution_dict::Dict{String, <:Dict},
                                 n_lines::Int, stages::Int)::Tuple{Matrix{Float64}, Int}
    sum_matrix     = zeros(Float64, n_lines, stages)
    total_solutions = length(solution_dict)
    total_lines    = 100
    pattern        = r"(\d+)\.(\d+)\.(\d+)\."

    for (sol_str, _) in solution_dict
        solution_lines = 0.0
        for m in eachmatch(pattern, sol_str)
            row = parse(Int, m.captures[1])
            col = parse(Int, m.captures[2])
            val = parse(Float64, m.captures[3])
            if checkbounds(Bool, sum_matrix, row, col)
                sum_matrix[row, col] += val
                solution_lines += val
            end
        end
        if solution_lines < total_lines && solution_lines > 0
            total_lines = solution_lines
        end
    end

    if total_solutions > 0
        return sum_matrix ./ total_solutions, total_lines
    else
        return zeros(Float64, n_lines, stages), 0
    end
end
