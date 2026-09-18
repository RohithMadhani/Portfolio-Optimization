import os, time, copy, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict
import torch
import torch.nn as nn
import cvxpy as cp
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from cptopt.utility import CPTUtility, CPTUtilityGA
from cptopt.optimizer import (
    MinorizationMaximizationOptimizer,
    ConvexConcaveOptimizer,
    GradientOptimizer,
    MeanVarianceFrontierOptimizer,
)

warnings.filterwarnings("ignore")
np.random.seed(0)
torch.manual_seed(0)

os.makedirs("figures", exist_ok=True)

GAMMA_POS, GAMMA_NEG = 8.4, 11.4
DELTA_POS, DELTA_NEG = 0.77, 0.79

utility = CPTUtility(
    gamma_pos=GAMMA_POS, gamma_neg=GAMMA_NEG,
    delta_pos=DELTA_POS, delta_neg=DELTA_NEG
)

CSV_PATH = "cpt_proxy_returns.csv"


def load_asset_subset(csv_path, asset_names):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if 'US_CorpBond_TR' in df.columns:
        df = df.drop(columns=['US_CorpBond_TR'])
    df = df[asset_names]
    first_valid = df.apply(lambda c: c.first_valid_index()).max()
    df = df.loc[first_valid:]
    assert df.notna().all().all(), "NaNs remain after slicing!"
    R = df.values.astype(np.float64)
    dates = df.index
    print(f"Assets: {list(asset_names)}")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"N={R.shape[0]} rows, n={R.shape[1]} cols")
    return R, dates


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AdamOptimizer:
    def __init__(self, utility_obj, lr=1e-2, max_iter=100_000, max_time=600):
        self._utility = copy.deepcopy(utility_obj)
        CPTUtilityGA.convert_to_class(self._utility)
        self.lr = lr
        self.max_iter = max_iter
        self.max_time = max_time
        self._weights = None
        self._weights_history = []
        self._wall_time = []

    @property
    def weights(self):
        return self._weights

    @property
    def wall_time(self):
        wt = np.array(self._wall_time)
        return wt - wt[0]

    def optimize(self, R, initial_weights=None, **kwargs):
        R_t = torch.tensor(R, dtype=torch.float64, device=DEVICE)
        n = R.shape[1]
        w0 = np.ones(n) / n if initial_weights is None else initial_weights.copy()
        x0 = np.log(w0 + 1e-8)
        x = nn.Parameter(torch.tensor(x0, device=DEVICE, dtype=torch.float64, requires_grad=True))
        optimizer = torch.optim.Adam([x], lr=self.lr, betas=(0.9, 0.999))

        self._weights_history = []
        self._wall_time = []
        t0 = time.time()

        for i in range(self.max_iter):
            w = torch.softmax(x, dim=0)
            loss = -self._utility.evaluate_with_gradient(w, R_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._weights_history.append(w.detach().cpu().numpy())
            self._wall_time.append(time.time())
            if time.time() - t0 > self.max_time:
                break

        self._weights = torch.softmax(x, dim=0).detach().cpu().numpy()


def cpt_eval(w, R):
    return utility.evaluate(w, R)[0]

if __name__ == '__main__':
    total_start = time.time()

    print("\n" + "="*60)
    # Task 1 done by Manas
    print("TASK 1: TOY EXAMPLE")
    print("="*60)
    t2_start = time.time()

    TOY_ASSETS = ['EM_Equity_TR', 'Commodities_TR', 'Silver']
    R_toy, dates_toy = load_asset_subset(CSV_PATH, TOY_ASSETS)

    grid_n = 100
    w1_vals = np.linspace(0, 1, grid_n)
    w2_vals = np.linspace(0, 1, grid_n)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    U_grid = np.full_like(W1, np.nan)

    print("Computing CPT surface (100x100)...")
    for i in range(grid_n):
        for j in range(grid_n):
            w1, w2 = W1[i, j], W2[i, j]
            if w1 + w2 <= 1 + 1e-10 and w1 >= 0 and w2 >= 0:
                w = np.array([w1, w2, 1.0 - w1 - w2])
                U_grid[i, j] = cpt_eval(w, R_toy)

    flat_idx = np.nanargmax(U_grid)
    i_star, j_star = np.unravel_index(flat_idx, U_grid.shape)
    w_star = np.array([W1[i_star, j_star], W2[i_star, j_star],
                       1 - W1[i_star, j_star] - W2[i_star, j_star]])
    print(f"Global optimum w*={w_star}, utility={U_grid[i_star, j_star]:.4f}")

    U_masked = U_grid.copy()
    for i in range(grid_n):
        for j in range(grid_n):
            dist = np.sqrt((W1[i, j] - w_star[0])**2 + (W2[i, j] - w_star[1])**2)
            if dist < 0.1:
                U_masked[i, j] = np.nan
    flat_idx2 = np.nanargmax(U_masked)
    i_bar, j_bar = np.unravel_index(flat_idx2, U_masked.shape)
    w_bar = np.array([W1[i_bar, j_bar], W2[i_bar, j_bar],
                      1 - W1[i_bar, j_bar] - W2[i_bar, j_bar]])
    print(f"Local optimum w_bar={w_bar}, utility={U_masked[i_bar, j_bar]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(W1, W2, U_grid, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='CPT Utility')
    ax.plot(w_star[0], w_star[1], '*', color='red', markersize=15, label='w* (global)')
    ax.plot(w_bar[0], w_bar[1], 's', color='orange', markersize=10, label='w̄ (local)')
    ax.set_xlabel('w1 (EM Equity)')
    ax.set_ylabel('w2 (Commodities)')
    ax.set_title('CPT Utility Surface (Toy Example)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/fig1_cpt_surface.png', dpi=150)
    plt.close()
    print("Saved figures/fig1_cpt_surface.png")

    mv_opt = MeanVarianceFrontierOptimizer(utility)
    mv_opt.optimize(R_toy, samples=100)
    mv_frontier_weights = mv_opt._weights_history
    w_mv = mv_opt.weights

    mu_toy = R_toy.mean(axis=0)
    Sigma_toy = np.cov(R_toy, rowvar=False)
    denom = 2 * w_mv @ Sigma_toy @ w_mv
    gamma_mv = (mu_toy @ w_mv) / denom if denom > 1e-12 else 1.0
    print(f"w_mv={w_mv}, gamma_mv={gamma_mv:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(W1, W2, U_grid, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='CPT Utility')
    fw = mv_frontier_weights
    ax.plot(fw[:, 0], fw[:, 1], '-', color='red', linewidth=3, label='MV Frontier', zorder=5)
    ax.plot(fw[:, 0], fw[:, 1], 'w--', linewidth=1.5, alpha=0.6, zorder=6)
    ax.plot(w_mv[0], w_mv[1], 'o', color='cyan', markersize=12, label='w_mv',
            markeredgecolor='black', markeredgewidth=1.5, zorder=7)
    ax.plot(w_star[0], w_star[1], '*', color='yellow', markersize=18, label='w* (global)',
            markeredgecolor='black', markeredgewidth=1, zorder=7)
    ax.set_xlabel('w1 (EM Equity)')
    ax.set_ylabel('w2 (Commodities)')
    ax.set_title('MV Frontier on CPT Surface')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/fig1a_mv_frontier.png', dpi=150)
    plt.close()
    print("Saved figures/fig1a_mv_frontier.png")

    U_mv_grid = np.full_like(W1, np.nan)
    for i in range(grid_n):
        for j in range(grid_n):
            w1, w2 = W1[i, j], W2[i, j]
            if w1 + w2 <= 1 + 1e-10 and w1 >= 0 and w2 >= 0:
                w = np.array([w1, w2, 1.0 - w1 - w2])
                U_mv_grid[i, j] = w @ mu_toy - gamma_mv * (w @ Sigma_toy @ w)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(W1, W2, U_mv_grid, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='MV Utility')
    ax.plot(w_mv[0], w_mv[1], 'o', color='cyan', markersize=10, label='w_mv')
    ax.set_xlabel('w1 (EM Equity)')
    ax.set_ylabel('w2 (Commodities)')
    ax.set_title(f'MV Utility Surface (gamma={gamma_mv:.2f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/fig1b_mv_surface.png', dpi=150)
    plt.close()
    print("Saved figures/fig1b_mv_surface.png")

    w0_equal  = np.array([1/3, 1/3, 1/3])
    w0_asset1 = np.array([0.90, 0.05, 0.05])
    w0_asset2 = np.array([0.05, 0.90, 0.05])
    w0_asset3 = np.array([0.05, 0.05, 0.90])
    w0_mv_start = w_mv.copy()

    starts = OrderedDict([
        ('Equal', w0_equal), ('EM Equity', w0_asset1), ('Commodities', w0_asset2),
        ('Silver', w0_asset3), ('MV', w0_mv_start)
    ])
    colors_traj = ['blue', 'red', 'green', 'purple', 'orange']

    def plot_trajectories(ax, trajectories, title):
        ax.contourf(W1, W2, U_grid, levels=20, cmap='viridis', alpha=0.8)
        for idx, (name, traj) in enumerate(trajectories.items()):
            c = colors_traj[idx]
            ax.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=1.5, label=name)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=c, markersize=6)
            ax.plot(traj[-1, 0], traj[-1, 1], 'o', color=c, markersize=8,
                    markeredgecolor='black', markeredgewidth=1.5)
            if len(traj) > 2:
                mid = len(traj) // 2
                ax.annotate('', xy=(traj[min(mid+1, len(traj)-1), 0],
                                    traj[min(mid+1, len(traj)-1), 1]),
                            xytext=(traj[mid, 0], traj[mid, 1]),
                            arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
        ax.set_xlabel('w1 (EM Equity)')
        ax.set_ylabel('w2 (Commodities)')
        ax.set_title(title)
        ax.legend(fontsize=7)

    print("Running GA convergence...")
    ga_trajs = OrderedDict()
    for name, w0 in starts.items():
        ga = GradientOptimizer(utility, max_iter=50000)
        ga.optimize(R_toy, initial_weights=w0.copy(), keep_history=True, max_time=30)
        ga_trajs[name] = ga._weights_history

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_trajectories(ax, ga_trajs, 'GA Convergence')
    plt.tight_layout()
    plt.savefig('figures/fig1c_ga.png', dpi=150)
    plt.close()
    print("Saved figures/fig1c_ga.png")

    print(f"Task 1 elapsed: {time.time()-t2_start:.1f}s")

    print("\n" + "="*60)
    # Task 2 done by Rohith
    print("TASK 2: MULTI-ASSET EXAMPLE")
    print("="*60)
    t3_start = time.time()

    ALL_ASSETS = [
        'US_Equity_TR', 'Europe_Equity_TR', 'Japan_Equity_TR',
        'EM_Equity_TR', 'US_GovtBond_TR', 'EU_GovtBond_TR',
        'JP_GovtBond_TR', 'US_Bills', 'EU_Bills', 'JP_Bills',
        'Commodities_TR', 'Gold', 'Silver'
    ]
    R_full, dates_full = load_asset_subset(CSV_PATH, ALL_ASSETS)
    n_full = R_full.shape[1]

    mv_full = MeanVarianceFrontierOptimizer(utility)
    mv_full.optimize(R_full, samples=100)
    w_mv_full = mv_full.weights
    print(f"MV optimal utility: {cpt_eval(w_mv_full, R_full):.4f}")

    def run_walltime_experiment(R, w0, label, savepath):
        print(f"Running wall-time experiment ({label})...")
        results = OrderedDict()

        t0 = time.time()
        mv_o = MeanVarianceFrontierOptimizer(utility)
        mv_o.optimize(R, samples=100)
        results['MV'] = {'times': [time.time() - t0], 'utils': [cpt_eval(mv_o.weights, R)]}

        try:
            mm_o = MinorizationMaximizationOptimizer(utility, max_iter=10)
            mm_o.optimize(R, initial_weights=w0.copy(), max_time=120)
            wh, wt = mm_o._weights_history, mm_o.wall_time
            mm_utils = [cpt_eval(wh[i], R) for i in range(len(wh))]
            results['MM'] = {'times': wt[:len(mm_utils)].tolist(), 'utils': mm_utils}
        except Exception as e:
            print(f"  MM failed: {e}")

        try:
            cc_o = ConvexConcaveOptimizer(utility, max_iter=10)
            cc_o.optimize(R, initial_weights=w0.copy(), max_time=120)
            wh, wt = cc_o._weights_history, cc_o.wall_time
            cc_utils = [cpt_eval(wh[i], R) for i in range(len(wh))]
            results['CC'] = {'times': wt[:len(cc_utils)].tolist(), 'utils': cc_utils}
        except Exception as e:
            print(f"  CC failed: {e}")

        ga_o = GradientOptimizer(utility, max_iter=100_000)
        ga_o.optimize(R, initial_weights=w0.copy(), keep_history=True, max_time=60)
        wh, wt = ga_o._weights_history, ga_o.wall_time
        step = max(1, len(wh) // 500)
        results['GA'] = {
            'times': wt[::step].tolist(),
            'utils': [cpt_eval(wh[i], R) for i in range(0, len(wh), step)]
        }

        adam_o = AdamOptimizer(utility, lr=1e-2, max_iter=100_000, max_time=60)
        adam_o.optimize(R, initial_weights=w0.copy())
        wh, wt = adam_o._weights_history, adam_o.wall_time
        step = max(1, len(wh) // 500)
        results['Adam'] = {
            'times': wt[::step].tolist(),
            'utils': [cpt_eval(wh[i], R) for i in range(0, len(wh), step)]
        }

        best_util = max(max(v['utils']) for v in results.values())
        fig, ax = plt.subplots(figsize=(10, 6))
        markers = {'MV': 's', 'MM': 'o', 'CC': '^', 'GA': 'v', 'Adam': 'D'}
        for name, data in results.items():
            t_arr = np.maximum(np.array(data['times']), 1e-4)
            ax.plot(t_arr, data['utils'], '-' + markers.get(name, 'o'),
                    label=name, markersize=3, linewidth=1.5)
        ax.axhline(best_util, color='gray', linestyle='--', alpha=0.7, label='Best utility')
        ax.set_xscale('log')
        ax.set_xlabel('Wall time (s)')
        ax.set_ylabel('CPT Utility')
        ax.set_title(f'Wall-time vs Utility ({label})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
        plt.close()
        print(f"Saved {savepath}")
        return results

    w0_eq_full = np.ones(n_full) / n_full
    results_5a = run_walltime_experiment(R_full, w0_eq_full, 'equal-weight start',
                                         'figures/fig2a_walltime.png')
    results_5b = run_walltime_experiment(R_full, w_mv_full, 'MV start',
                                         'figures/fig2b_walltime_mv_start.png')

    print(f"Task 2 elapsed: {time.time()-t3_start:.1f}s")

    print("\n" + "="*60)
    # Task 3 done by Rohith
    print("TASK 3: ROLLING WINDOW BACKTEST")
    print("="*60)
    t5_start = time.time()

    def rolling_backtest(R, methods_dict, window=60):
        N, n = R.shape
        steps = N - window
        realized_ret = {name: [] for name in methods_dict}
        insample_cpt = {name: [] for name in methods_dict}
        prev_w = {name: np.ones(n) / n for name in methods_dict}

        for t in tqdm(range(steps), desc="Rolling backtest"):
            train_R = R[t:t+window]
            for name, opt_factory in methods_dict.items():
                opt = opt_factory()
                w0 = prev_w[name]
                if isinstance(opt, MeanVarianceFrontierOptimizer):
                    opt.optimize(train_R, samples=100)
                elif isinstance(opt, GradientOptimizer):
                    opt.optimize(train_R, initial_weights=w0, keep_history=False, max_time=10)
                elif isinstance(opt, AdamOptimizer):
                    opt.optimize(train_R, initial_weights=w0)
                else:
                    opt.optimize(train_R, initial_weights=w0)
                w = opt.weights
                prev_w[name] = w
                realized_ret[name].append(R[t+window] @ w)
                insample_cpt[name].append(cpt_eval(w, train_R))

        results = {}
        for name in methods_dict:
            rets = np.array(realized_ret[name])
            wealth = np.cumprod(1 + rets)
            running_max = np.maximum.accumulate(wealth)
            results[name] = {
                'ann_return': np.mean(rets) * 12,
                'ann_sharpe': np.mean(rets) / np.std(rets) * np.sqrt(12) if np.std(rets) > 0 else 0,
                'max_drawdown': np.max((running_max - wealth) / running_max),
                'mean_is_cpt': np.mean(insample_cpt[name]),
                'realized_ret': rets,
                'wealth': wealth
            }
        return results

    backtest_methods = OrderedDict([
        ('MV',   lambda: MeanVarianceFrontierOptimizer(utility)),
        ('MM',   lambda: MinorizationMaximizationOptimizer(utility, max_iter=50)),
        ('CC',   lambda: ConvexConcaveOptimizer(utility, max_iter=50)),
        ('GA',   lambda: GradientOptimizer(utility, max_iter=5000)),
        ('Adam', lambda: AdamOptimizer(utility, lr=1e-2, max_iter=5000, max_time=10)),
    ])

    bt_results = rolling_backtest(R_full, backtest_methods, window=60)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res in bt_results.items():
        ax.plot(res['wealth'], label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Wealth')
    ax.set_title('Rolling Backtest: Cumulative Wealth')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/fig_rolling_wealth.png', dpi=150)
    plt.close()
    print("Saved figures/fig_rolling_wealth.png")

    n_methods_bt = len(bt_results)
    fig, axes = plt.subplots(n_methods_bt, 1, figsize=(12, 3*n_methods_bt), sharex=True)
    if n_methods_bt == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, bt_results.items()):
        ax.bar(range(len(res['realized_ret'])), res['realized_ret'], alpha=0.7)
        ax.set_ylabel('Return')
        ax.set_title(f'{name} Monthly Returns')
    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    plt.savefig('figures/fig_rolling_monthly.png', dpi=150)
    plt.close()
    print("Saved figures/fig_rolling_monthly.png")

    print("\n" + "="*80)
    print(f"{'Method':<10} {'Ann.Return':>12} {'Ann.Sharpe':>12} {'MaxDrawdown':>12} {'Mean IS CPT':>12}")
    print("-"*80)
    for name, res in bt_results.items():
        print(f"{name:<10} {res['ann_return']:>12.4f} {res['ann_sharpe']:>12.4f} "
              f"{res['max_drawdown']:>12.4f} {res['mean_is_cpt']:>12.4f}")
    print("="*80)

    print(f"Task 3 elapsed: {time.time()-t5_start:.1f}s")

    print("\n" + "="*60)
    # Task 4 done by Lohith
    print("TASK 4: SCALING TEST")
    print("="*60)
    t6_start = time.time()

    print("Fitting GaussianMixture to R_full...")
    gm = GaussianMixture(n_components=3, random_state=0)
    gm.fit(R_full)
    R_synth = gm.sample(6000)[0].astype(np.float64)
    print(f"Synthetic returns: {R_synth.shape}")

    mv_synth = MeanVarianceFrontierOptimizer(utility)
    mv_synth.optimize(R_synth, samples=100)
    w_mv_synth = mv_synth.weights
    mv_synth_util = cpt_eval(w_mv_synth, R_synth)
    print(f"MV utility on synth: {mv_synth_util:.4f}")

    print("Running GA-vanilla on synthetic (max 120s)...")
    ga_synth = GradientOptimizer(utility, max_iter=500_000)
    ga_synth.optimize(R_synth, initial_weights=w_mv_synth.copy(),
                      keep_history=True, max_time=120)
    ga_wh, ga_wt = ga_synth._weights_history, ga_synth.wall_time
    step_g = max(1, len(ga_wh) // 200)
    ga_synth_times = ga_wt[::step_g]
    ga_synth_utils = [cpt_eval(ga_wh[i], R_synth)
                      for i in tqdm(range(0, len(ga_wh), step_g), desc="GA eval")]

    print("Running GA-Adam on synthetic (max 120s)...")
    adam_synth = AdamOptimizer(utility, lr=1e-2, max_iter=500_000, max_time=120)
    adam_synth.optimize(R_synth, initial_weights=w_mv_synth.copy())
    adam_wh, adam_wt = adam_synth._weights_history, adam_synth.wall_time
    step_a = max(1, len(adam_wh) // 200)
    adam_synth_times = adam_wt[::step_a]
    adam_synth_utils = [cpt_eval(adam_wh[i], R_synth)
                        for i in tqdm(range(0, len(adam_wh), step_a), desc="Adam eval")]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ga_synth_times[:len(ga_synth_utils)], ga_synth_utils,
            '-o', label='GA-vanilla', markersize=2)
    ax.plot(adam_synth_times[:len(adam_synth_utils)], adam_synth_utils,
            '-D', label='GA-Adam', markersize=2)
    ax.axhline(mv_synth_util, color='gray', linestyle='--', alpha=0.7, label='MV utility')
    ax.set_xlabel('Wall time (s)')
    ax.set_ylabel('CPT Utility')
    ax.set_title('Scaling Test: 6000 Synthetic Returns (13 assets)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/fig4_scaling.png', dpi=150)
    plt.close()
    print("Saved figures/fig4_scaling.png")

    print(f"Task 4 elapsed: {time.time()-t6_start:.1f}s")

    print("\n" + "="*90)
    print("FINAL SUMMARY TABLE")
    print("="*90)

    summary_data = OrderedDict()
    for name in ['MV', 'MM', 'CC', 'GA', 'Adam']:
        if name in results_5a:
            final_util = max(results_5a[name]['utils'])
            wtime = max(results_5a[name]['times'])
        else:
            final_util = wtime = 0
        bt_r = bt_results.get(name, {})
        summary_data[name] = {
            'util': final_util,
            'wtime': wtime,
            'sharpe': bt_r.get('ann_sharpe', 0),
            'maxdd': bt_r.get('max_drawdown', 0),
        }

    print(f"{'Method':<10} {'Final Utility':>15} {'Wall-Time (s)':>15} "
          f"{'Ann.Sharpe':>12} {'Max Drawdown':>14}")
    print("-"*90)
    for name, d in summary_data.items():
        print(f"{name:<10} {d['util']:>15.4f} {d['wtime']:>15.2f} "
              f"{d['sharpe']:>12.4f} {d['maxdd']:>14.4f}")
    print("="*90)

    print(f"\nTotal elapsed: {time.time()-total_start:.1f}s")