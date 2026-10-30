"""
    GraspTNEP

Paquete de optimización para el Problema de Expansión de Transmisión (TEP)
basado en GRASP con EDA adaptativo, operadores GA, y tres niveles de
respuesta al estancamiento (ILS, VNS, reinicio parcial).

# Uso básico
```julia
using GraspTNEP

# configurar caso
data   = ACOPF_Extensions.setup_case("path/to/case", false, false)

# parámetros (todos tienen valores por defecto)
params = GRASPParameters(maxIter=100, indv=10, N_elite=5)

# una corrida
result = run_simulation(data, params)

# múltiples corridas con guardado
sim = run_simulations(data, params, 10, save_path="results/caso.jld2")

# análisis de operadores
stats = operator_stats(result)

# traza de mejoras (para RL guiado futuro)
mejoras = improvement_trace(result)
```
"""
module GraspTNEP

using Random, Dates, Distributions, Statistics, StatsBase
using DataFrames

# Dependencias externas del problema (no declaradas en Project.toml
# porque son del entorno del usuario)
using MathOptInterface
import ACOPF_Extensions
import Distributed: pmap
import BSON: @save, @load


# ---- tipos y parámetros ----
include("types.jl")
include("parameters.jl")

# ---- utilidades ----
include("auxiliares.jl")
include("evaluacion.jl")

# ---- núcleo del algoritmo ----
include("eda.jl")
include("genetic.jl")
include("intensification.jl")
include("grasp.jl")

# ---- API pública ----
include("simulation.jl")

include("GRASP_simple_parameters.jl")
include("GRASP_simple_construccion.jl")
include("GRASP_simple_eda.jl")
include("GRASP_simple_evaluacion.jl")
include("GraspSimple.jl")

export GRASPParameters, default_params, show_params, GRASPSimpleParameters, params_simple
export RunResult, SimulationResult, SolutionRecord
export run_simulation, run_simulations, generate_solution_id, feasible2int
export improvement_trace, trace_by_origin, operator_stats, cost_top, grasp_simple

end # module GraspTNEP
