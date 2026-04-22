# ============================================================
#  eda.jl  —  actualización del average (EDA) con tres niveles
# ============================================================

"""
    compute_beta(no_improve, thr1, thr2, beta_base, beta_max) -> Float64

Calcula β para los pesos exponenciales. Crece linealmente desde
`beta_base` (en no_improve = thr1) hasta `beta_max` (en no_improve = thr2).
"""
function compute_beta(no_improve::Int, thr1::Int, thr2::Int,
                       beta_base::Float64, beta_max::Float64)::Float64
    t = clamp((no_improve - thr1) / max(thr2 - thr1, 1), 0.0, 1.0)
    return beta_base + t * (beta_max - beta_base)
end

"""
    exponential_weights(costs, best_cost, beta) -> Vector{Float64}

Calcula pesos proporcionales a exp(-β · cost / best_cost).
Normaliza para que sumen 1.
"""
function exponential_weights(costs::Vector{Float64}, best_cost::Float64,
                              beta::Float64)::Vector{Float64}
    w = exp.(-beta .* costs ./ max(best_cost, 1e-8))
    s = sum(w)
    return s > 0 ? w ./ s : fill(1.0 / length(costs), length(costs))
end

"""
    update_average_eda!(average, elite, no_improve, p, freq, iter)

Actualiza `average` según el nivel de estancamiento detectado:

- Nivel 0 (normal):       EDA uniforme sobre top-k del élite.
- Nivel 1 (≥ thr1):       EDA con pesos exponenciales, β creciente.
- Nivel 2 (≥ thr2):       EDA exponencial + ruido dirigido a celdas poco usadas.

El reinicio de nivel 3 se maneja en `grasp.jl` directamente sobre el élite.
Retorna el β usado (para logging).
"""
function update_average_eda!(average    :: Matrix{Float64},
                              elite      :: Vector{Tuple{Matrix{Int}, Float64}},
                              no_improve :: Int,
                              p          :: GRASPParameters,
                              freq       :: Matrix{Float64},
                              iter       :: Int)::Float64

    isempty(elite) && return 0.0

    thr1 = round(Int, p.N_Iter_No * p.lvl1_ratio)
    thr2 = round(Int, p.N_Iter_No * p.lvl2_ratio)

    # top-k soluciones del élite (ya están ordenadas por costo)
    k       = min(p.top_k, length(elite))
    top_k   = elite[1:k]
    sols    = [x[1] for x in top_k]
    costs   = [x[2] for x in top_k]
    best_c  = elite[1][2]

    # ---- calcular pesos ----
    beta = 0.0
    weights = if no_improve >= thr1
        beta = compute_beta(no_improve, thr1, thr2, p.beta_base, p.beta_max)
        exponential_weights(costs, best_c, beta)
    else
        fill(1.0 / k, k)   # uniforme
    end

    # ---- término de actualización ponderada ----
    update_term = zeros(size(average))
    for (sol, w) in zip(sols, weights)
        update_term .+= sol .* w
    end

    # ---- interpolación EDA base ----
    average .= (1 - p.rho) .* average .+ p.rho .* update_term

    # ---- Nivel 2: ruido dirigido a celdas poco usadas ----
    if no_improve >= thr2
        mask   = underused_mask(freq, iter, 0.2)
        noise  = rand(size(average)...) .* mask   # ruido ∈ [0,1] solo en poco usadas
        average .+= p.rho .* 0.5 .* noise         # perturbación suave
    end

    average .= clamp.(average, 0.0, 3.0)
    average[isnan.(average)] .= 0.0
    average .= round.(average, digits=3)

    return beta
end
