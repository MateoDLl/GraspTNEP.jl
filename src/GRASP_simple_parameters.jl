# ============================================================
#  GRASP_simple_parameters.jl
#  Parámetros para el GRASP simple EDA + Beam Search
#  Archivo independiente — no modifica GRASP_parameters.jl
# ============================================================

Base.@kwdef struct GRASPSimpleParameters
    maxIter      :: Int     = 100      # pocas iteraciones — objetivo: rápido
    indv         :: Int     = 10       # soluciones por iteración
    alpha        :: Float64 = 0.679   # prob. construir desde cero vs pool
    rho          :: Float64 = 0.4     # learning rate del average (alto = converge rápido)
    beam_width   :: Int     = 3       # soluciones que sobreviven el beam por stage
    beam_samples :: Int     = 10      # candidatos generados por stage en beam search
    top_k        :: Int     = 3       # cuántas soluciones actualizan el average
    filter_factor:: Float64 = 1.2     # umbral prefiltro: filter_factor * best_inv_cost
end

params_simple = GRASPSimpleParameters(
    maxIter      = 50,
    indv         = 10,
    alpha        = 0.679,
    rho          = 0.4,
    beam_width   = 3,
    beam_samples = 10,
    top_k        = 3,
    filter_factor= 1.2
)
