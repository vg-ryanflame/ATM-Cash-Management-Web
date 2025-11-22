import os
import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
import pyswarms as ps
from deap import base, creator, tools, algorithms
from numba import njit
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import tempfile 

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- Utility: load predicted demand ----------
def load_predicted_demand():
    if os.path.exists("predicted_demand.npy"):
        return np.load("predicted_demand.npy").astype(float)
    return np.array([10000, 12000, 8000, 15000, 11000, 9000, 13000] * 5).astype(float)

# ---------- Core simulation functions (optimized with Numba) ----------
@njit
def simulate_miller_orr(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine,
                        lower_bound, upper_bound, target_balance):
    n_days = len(predicted_demand)
    L, U, refill_target = lower_bound, upper_bound, target_balance 
    cash = initial_cash
    records = np.zeros((n_days, 8)) 

    for t in range(n_days):
        demand_today = predicted_demand[t]
        cash -= demand_today
        fine_cost = 0.0
        refill_cost_day = 0.0
        oppo_cost = oppo_rate * cash if cash > 0 else 0.0

        if cash < 0:
            fine_cost = shortage_fine
            cash = 0.0
        refill_amount = 0.0
        if cash < L:
            refill_cost_day = refill_cost_val
            refill_amount = max(0.0, refill_target - cash)
            cash += refill_amount

        daily_total = oppo_cost + refill_cost_day + fine_cost
        records[t] = [t, demand_today, cash, refill_amount, oppo_cost, refill_cost_day, fine_cost, daily_total]
        
    return records

@njit
def simulate_pso_policy(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine,
                        lookahead_days, alpha, threshold):
    n_days = len(predicted_demand)
    cash = initial_cash
    total_cost = 0.0
    for t in range(n_days):
        demand_today = predicted_demand[t]
        cash -= demand_today
        fine_cost = 0.0
        if cash < 0:
            fine_cost = shortage_fine
            cash = 0.0
        oppo_cost = oppo_rate * cash if cash > 0 else 0.0
        refill_cost_day = 0.0
        if cash < threshold:
            end_index = min(n_days, t + 1 + lookahead_days)
            future_sum = np.sum(predicted_demand[t+1 : end_index])
            refill_amount = max(0.0, alpha * future_sum - cash)
            if refill_amount > 0.0:
                cash += refill_amount
                refill_cost_day = refill_cost_val
        total_cost += fine_cost + oppo_cost + refill_cost_day
    return total_cost

def simulate_pso_policy_details(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days, params):
    alpha, threshold = float(params[0]), float(params[1])
    n_days = len(predicted_demand)
    cash = float(initial_cash)
    records = []
    for t in range(n_days):
        demand_today = float(predicted_demand[t])
        cash -= demand_today
        fine_cost = 0.0
        if cash < 0:
            fine_cost = shortage_fine
            cash = 0.0
        oppo_cost = oppo_rate * cash if cash > 0 else 0.0
        refill_cost_day = 0.0
        refill_amount = 0.0
        if cash < threshold:
            future_sum = np.sum(predicted_demand[t+1 : min(n_days, t+1+lookahead_days)])
            refill_amount = max(0.0, alpha * float(future_sum) - cash)
            if refill_amount > 0.0:
                cash += refill_amount
                refill_cost_day = refill_cost_val
        daily_total = fine_cost + oppo_cost + refill_cost_day
        records.append({
            "Day": int(t), "PredictedDemand": demand_today, "CashBalance": float(cash),
            "RefillAmount": float(refill_amount), "OpportunityCost": float(oppo_cost),
            "RefillCost": float(refill_cost_day), "FineCost": float(fine_cost), "TotalCost": float(daily_total)
        })
    return pd.DataFrame(records)


# ---------- Optimization Wrappers ----------
def pso_optimize(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days,
                 n_particles=30, iters=40, alpha_bound=3.0, threshold_mult=1.5):
    def cost_for_particles(X):
        costs = []
        for p in X:
            costs.append(simulate_pso_policy(
                predicted_demand, initial_cash, oppo_rate, refill_cost_val,
                shortage_fine, lookahead_days, p[0], p[1]
            ))
        return np.array(costs)

    MAX_THRESHOLD = np.max(predicted_demand) * threshold_mult
    # Allow negative thresholds so refills can be avoided entirely
    bounds = (np.array([0.0, -np.max(predicted_demand)]), np.array([alpha_bound, MAX_THRESHOLD]))
    # Improved PSO options for better convergence
    options = {'c1': 2.0, 'c2': 2.0, 'w': 0.9, 'k': 3, 'p': 2}
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=2, options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(cost_for_particles, iters=iters, verbose=False)
    return best_cost, best_pos

# --- GA Optimization ---
try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
except ValueError: pass 

def ga_optimize(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days,
                pop_size=30, n_generations=40, alpha_bound=3.0, threshold_mult=1.5):
    MAX_THRESHOLD = np.max(predicted_demand) * threshold_mult
    MIN_THRESHOLD = -np.max(predicted_demand)  # Allow negative thresholds
    toolbox = base.Toolbox()
    toolbox.register("attr_alpha", np.random.uniform, 0.0, alpha_bound)
    toolbox.register("attr_threshold", np.random.uniform, MIN_THRESHOLD, MAX_THRESHOLD)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_alpha, toolbox.attr_threshold), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate_individual(individual):
        cost = simulate_pso_policy(predicted_demand, initial_cash, oppo_rate, refill_cost_val,
                                   shortage_fine, lookahead_days, individual[0], individual[1])
        return (cost,)
    
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=[0.0, 0.0], sigma=[0.5, 0.1 * MAX_THRESHOLD], indpb=0.2)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations, stats=None, halloffame=hof, verbose=False)
    
    return hof[0].fitness.values[0], np.array(hof[0])

# --- ABC Optimization ---
def abc_optimize(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days,
                 sn=30, max_iterations=40, alpha_bound=3.0, threshold_mult=1.5):
    D = 2; LIMIT = sn * D
    LOWER_BOUNDS = np.array([0.0, -np.max(predicted_demand)])
    UPPER_BOUNDS = np.array([alpha_bound, np.max(predicted_demand) * threshold_mult])
    
    def evaluate_source(source):
        cost = simulate_pso_policy(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days, source[0], source[1])
        return cost, 1.0 / (1.0 + cost)

    population = LOWER_BOUNDS + np.random.rand(sn, D) * (UPPER_BOUNDS - LOWER_BOUNDS)
    trials = np.zeros(sn)
    costs, fitnesses = zip(*[evaluate_source(p) for p in population])
    costs, fitnesses = np.array(costs), np.array(fitnesses)
    best_cost = np.min(costs)
    best_params = population[np.argmin(costs)].copy()

    for _ in range(max_iterations):
        # Employed Phase
        for i in range(sn):
            k = i
            while k == i: k = np.random.randint(sn)
            phi = np.random.uniform(-1, 1, D)
            new_source = np.clip(population[i] + phi * (population[i] - population[k]), LOWER_BOUNDS, UPPER_BOUNDS)
            new_cost, new_fitness = evaluate_source(new_source)
            if new_fitness > fitnesses[i]:
                population[i], fitnesses[i], trials[i] = new_source, new_fitness, 0
                if new_cost < best_cost: best_cost, best_params = new_cost, new_source.copy()
            else: trials[i] += 1
        
        # Onlooker Phase
        probs = fitnesses / np.sum(fitnesses)
        for _ in range(sn):
            i = np.random.choice(sn, p=probs)
            k = i
            while k == i: k = np.random.randint(sn)
            phi = np.random.uniform(-1, 1, D)
            new_source = np.clip(population[i] + phi * (population[i] - population[k]), LOWER_BOUNDS, UPPER_BOUNDS)
            new_cost, new_fitness = evaluate_source(new_source)
            if new_fitness > fitnesses[i]:
                population[i], fitnesses[i], trials[i] = new_source, new_fitness, 0
                if new_cost < best_cost: best_cost, best_params = new_cost, new_source.copy()
            else: trials[i] += 1

        # Scout Phase
        abandoned = np.argmax(trials)
        if trials[abandoned] >= LIMIT:
            population[abandoned] = LOWER_BOUNDS + np.random.rand(D) * (UPPER_BOUNDS - LOWER_BOUNDS)
            costs[abandoned], fitnesses[abandoned] = evaluate_source(population[abandoned])
            trials[abandoned] = 0
            if costs[abandoned] < best_cost: best_cost, best_params = costs[abandoned], population[abandoned].copy()

    return best_cost, best_params



# =========================================================================
# 🚀 HIGH-SPEED ANIMATION LOGIC
# =========================================================================
@njit
def objective_function_plot(x, y):
    return (np.sin(np.sqrt(x**2 + y**2)) / np.sqrt(x**2 + y**2)) * 5 + 0.5 * np.cos(3 * x) + 0.5 * np.sin(3 * y)

def create_pso_animation_file(n_particles, max_iter):
    """
    Generates GIF using pre-calculated history.
    OPTIMIZATIONS:
    1. Wireframe instead of Surface (Faster rendering)
    2. Low DPI and Figure Size (Faster saving)
    3. Reduced Smoothing Steps (Fewer frames)
    """
    C1, C2, W = 0.55, 0.45, 0.75
    BOUNDS = [-5.0, 3.5]
    DIM = 2
    # OPTIMIZATION: Reduce frames to minimum needed for smoothness
    SMOOTHING_STEPS = 2 
    
    particles_pos = np.random.uniform(low=BOUNDS[0], high=BOUNDS[1], size=(n_particles, DIM))
    particles_vel = np.random.uniform(low=-0.5, high=0.5, size=(n_particles, DIM))
    p_best_pos = particles_pos.copy()
    p_best_scores = objective_function_plot(p_best_pos[:, 0], p_best_pos[:, 1])
    g_best_index = np.argmin(p_best_scores)
    g_best_pos = p_best_pos[g_best_index].copy()
    g_best_score = p_best_scores[g_best_index]

    pos_history = [particles_pos.copy()]
    gbest_history = [g_best_pos.copy()]
    
    for _ in range(max_iter):
        scores = objective_function_plot(particles_pos[:, 0], particles_pos[:, 1])
        better = scores < p_best_scores
        p_best_scores[better] = scores[better]
        p_best_pos[better] = particles_pos[better]
        best_idx = np.argmin(scores)
        if scores[best_idx] < g_best_score:
            g_best_score, g_best_pos = scores[best_idx], particles_pos[best_idx].copy()
        
        r1, r2 = np.random.rand(n_particles, DIM), np.random.rand(n_particles, DIM)
        particles_vel = W * particles_vel + C1 * r1 * (p_best_pos - particles_pos) + C2 * r2 * (g_best_pos - particles_pos)
        particles_pos = np.clip(particles_pos + particles_vel, BOUNDS[0], BOUNDS[1])
        pos_history.append(particles_pos.copy())
        gbest_history.append(g_best_pos.copy())

    # OPTIMIZATION: Smaller figure size = Fewer pixels to render
    fig = plt.figure(figsize=(9, 4.5)) 
    ax3D = fig.add_subplot(121, projection='3d')
    ax2D = fig.add_subplot(122)
    
    # OPTIMIZATION: Reduced grid density (25 vs 60)
    X, Y = np.meshgrid(np.linspace(BOUNDS[0], BOUNDS[1], 15), np.linspace(BOUNDS[0], BOUNDS[1], 19))
    Z = objective_function_plot(X, Y)
    
    # OPTIMIZATION: Wireframe is significantly faster than Surface
    ax3D.plot_wireframe(X, Y, Z, alpha=0.3, color='gray', linewidth=0.5)
    ax2D.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.8)
    
    # Remove axes labels/ticks to save rendering time
    ax3D.set_axis_off()
    # ax2D.axis('off') 
    plt.tight_layout()

    scatter3D = ax3D.scatter([], [], [], c='red', s=10)
    best_mark3D = ax3D.scatter([], [], [], c='orange', marker='*', s=80)
    scatter2D = ax2D.scatter([], [], c='red', s=10)
    best_mark2D = ax2D.scatter([], [], c='orange', marker='*', s=80)

    TOTAL_FRAMES = max_iter * SMOOTHING_STEPS
    def update(frame):
        segment = frame // SMOOTHING_STEPS
        progress = (frame % SMOOTHING_STEPS) / SMOOTHING_STEPS
        if segment >= len(pos_history) - 1:
            current_pos, current_gbest = pos_history[-1], gbest_history[-1]
        else:
            current_pos = pos_history[segment] * (1 - progress) + pos_history[segment + 1] * progress
            current_gbest = gbest_history[segment]
        
        z_vals = objective_function_plot(current_pos[:, 0], current_pos[:, 1])
        scatter3D._offsets3d = (current_pos[:, 0], current_pos[:, 1], z_vals)
        best_mark3D._offsets3d = ([current_gbest[0]], [current_gbest[1]], [g_best_score])
        scatter2D.set_offsets(current_pos)
        best_mark2D.set_offsets([current_gbest[0], current_gbest[1]])
        return scatter3D, scatter2D

    # OPTIMIZATION: Interval 30ms (Fast playback)
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=100, blit=False)
    
    f = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    fname = f.name
    f.close()
    try: 
        # OPTIMIZATION: Low DPI (40) = Fast Saving
        anim.save(fname, writer='pillow', fps=6, dpi=60)
    except Exception as e: 
        os.remove(fname)
        raise e
    plt.close(fig)
    return fname

# =========================================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.get_json(force=True)
    
    # Inputs
    oppo_rate = float(data.get("oppo_rate", 0.01))
    refill_cost_val = float(data.get("refill_cost", 1000.0))
    shortage_fine = float(data.get("shortage_fine", 10000.0))
    lookahead_days = int(data.get("lookahead", 2))
    n_pop = int(data.get("n_particles", 30))
    iters = int(data.get("iters", 40))
    alpha_bound = float(data.get("alpha_bound", 3.0))
    threshold_mult = float(data.get("threshold_mult", 1.5))
    
    m_lower = float(data.get("m_lower")) if data.get("m_lower") else None
    m_upper = float(data.get("m_upper")) if data.get("m_upper") else None
    m_target = float(data.get("m_target")) if data.get("m_target") else None

    predicted_demand = load_predicted_demand()
    initial_cash = float(np.median(predicted_demand))
    
    if m_lower is None: m_lower = np.percentile(predicted_demand, 25)
    if m_upper is None: m_upper = np.percentile(predicted_demand, 75)
    if m_target is None: m_target = m_upper

    # 1. Miller Orr Simulation
    miller_records = simulate_miller_orr(
        predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine,
        m_lower, m_upper, m_target
    )
    df_miller = pd.DataFrame(miller_records, columns=["Day", "PredictedDemand", "CashBalance", "RefillAmount", "OpportunityCost", "RefillCost", "FineCost", "TotalCost"])

    # 2. PSO Optimization
    best_cost_pso, best_pos_pso = pso_optimize(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days, n_pop, iters, alpha_bound, threshold_mult)
    df_pso = simulate_pso_policy_details(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days, best_pos_pso)
    
    # 3. GA Optimization
    best_cost_ga, best_pos_ga = ga_optimize(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days, n_pop, iters, alpha_bound, threshold_mult)
    df_ga = simulate_pso_policy_details(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days, best_pos_ga)
    
    # 4. ABC Optimization
    best_cost_abc, best_pos_abc = abc_optimize(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days, n_pop, iters, alpha_bound, threshold_mult)
    df_abc = simulate_pso_policy_details(predicted_demand, initial_cash, oppo_rate, refill_cost_val, shortage_fine, lookahead_days, best_pos_abc)

    # Summary Helper with Breakdown
    def summarize(df, pos=None, cost=None):
        summary = {
            "TotalCost": float(df["TotalCost"].sum()),
            "AvgDailyCost": float(df["TotalCost"].mean()),
            "NumberRefills": int((df["RefillAmount"] > 0).sum()),
            "AvgRefillAmount": float(df[df["RefillAmount"]>0]["RefillAmount"].mean()) if (df["RefillAmount"]>0).any() else 0.0,
            "NumberFines": int((df["FineCost"] > 0).sum()),
            # Critical for Charts:
            "breakdown": {
                "OpportunityCost": float(df["OpportunityCost"].sum()),
                "RefillCost": float(df["RefillCost"].sum()),
                "FineCost": float(df["FineCost"].sum())
            }
        }
        if pos is not None and cost is not None:
            summary["best_params"] = [float(pos[0]), float(pos[1])]
            summary["best_cost"] = float(cost)
        return summary

    return jsonify({
        "miller": {"summary": summarize(df_miller), "daily": df_miller.to_dict(orient="list")},
        "pso": {"summary": summarize(df_pso, best_pos_pso, best_cost_pso), "daily": df_pso.to_dict(orient="list")},
        "ga": {"summary": summarize(df_ga, best_pos_ga, best_cost_ga), "daily": df_ga.to_dict(orient="list")},
        "abc": {"summary": summarize(df_abc, best_pos_abc, best_cost_abc), "daily": df_abc.to_dict(orient="list")}
    })

@app.route("/animate_pso")
def animate_pso():
    try:
        # Generate GIF to a temp file with aggressively reduced parameters
        # We ignore the user params here for speed, forcing a fast preset
        gif_path = create_pso_animation_file(n_particles=12, max_iter=15)
        return send_file(gif_path, mimetype='image/gif')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- WARMUP ---
print("🔥 Warming up Numba...")
_dummy_demand = np.array([1000.0, 2000.0])
simulate_miller_orr(_dummy_demand, 5000.0, 0.01, 100.0, 100.0, 1000.0, 5000.0, 4000.0)
simulate_pso_policy(_dummy_demand, 5000.0, 0.01, 100.0, 100.0, 1, 1.5, 2000.0)
objective_function_plot(np.array([1.0]), np.array([1.0]))
print("✅ Warmup complete.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)