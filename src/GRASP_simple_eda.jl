# ============================================================
#  GRASP_simple_eda.jl
#  Actualización del average — versión simple sin RL ni niveles
# ============================================================

"""
    update_average_simple!(average, factible_sols, costs, p)

Actualiza el average con las top_k soluciones factibles de la iteración.
Pesos uniformes — sin pesos exponenciales ni niveles de estancamiento.
Tasa de aprendizaje rho alta para convergencia rápida.
"""
function update_average_simple!(average       :: Matrix{Float64},
                                 factible_sols :: Vector{Matrix{Int}},
                                 costs         :: Vector{Float64},
                                 p             :: GRASPSimpleParameters)

    isempty(factible_sols) && return

    # ordenar por costo y tomar top_k
    ord    = sortperm(costs)
    k      = min(p.top_k, length(factible_sols))
    top_k  = factible_sols[ord[1:k]]

    # promedio uniforme de las top_k
    update_term = mean(Float64.(s) for s in top_k)

    # interpolación EDA
    average .= (1 - p.rho) .* average .+ p.rho .* update_term
    average .= clamp.(average, 0.0, 5.0)
    average[isnan.(average)] .= 0.0
end
