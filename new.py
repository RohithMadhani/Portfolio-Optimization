import os, time, copy, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn as nn
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA_POS, GAMMA_NEG = 8.4, 11.4
DELTA_POS, DELTA_NEG = 0.77, 0.79
CSV_PATH = "cpt_proxy_returns.csv"

utility = CPTUtility(gamma_pos=GAMMA_POS, gamma_neg=GAMMA_NEG,
                     delta_pos=DELTA_POS, delta_neg=DELTA_NEG)

def cpt_eval(w, R):
    return utility.evaluate(w, R)[0]

def load_asset_subset(csv_path, asset_names):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if 'US_CorpBond_TR' in df.columns:
        df = df.drop(columns=['US_CorpBond_TR'])
    df = df[asset_names]
    first_valid = df.apply(lambda c: c.first_valid_index()).max()
    df = df.loc[first_valid:]
    assert df.notna().all().all()
    return df.values.astype(np.float64), df.index

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
        x = nn.Parameter(torch.tensor(np.log(w0 + 1e-8), device=DEVICE, dtype=torch.float64))
        optimizer = torch.optim.Adam([x], lr=self.lr, betas=(0.9, 0.999))
        self._weights_history = []
        self._wall_time = []
        t0 = time.time()
        for _ in range(self.max_iter):
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


def run_walltime_experiment(R, w0, label, savepath):
    print(f"Running wall-time experiment ({label})...")
    results = OrderedDict()

    # MV
    t0 = time.time()
    mv_o = MeanVarianceFrontierOptimizer(utility)
    mv_o.optimize(R, samples=100)
    results['MV'] = {'times': [time.time() - t0], 'utils': [cpt_eval(mv_o.weights, R)]}

    # MM
    try:
        mm_o = MinorizationMaximizationOptimizer(utility, max_iter=10)
        mm_o.optimize(R, initial_weights=w0.copy(), max_time=120)
        wh, wt = mm_o._weights_history, mm_o.wall_time
        mm_utils = [cpt_eval(wh[i], R) for i in range(len(wh))]
        results['MM'] = {'times': wt[:len(mm_utils)].tolist(), 'utils': mm_utils}
    except Exception as e:
        print(f"  MM failed: {e}")

    # CC
    try:
        cc_o = ConvexConcaveOptimizer(utility, max_iter=10)
        cc_o.optimize(R, initial_weights=w0.copy(), max_time=120)
        wh, wt = cc_o._weights_history, cc_o.wall_time
        cc_utils = [cpt_eval(wh[i], R) for i in range(len(wh))]
        results['CC'] = {'times': wt[:len(cc_utils)].tolist(), 'utils': cc_utils}
    except Exception as e:
        print(f"  CC failed: {e}")

    # GA-vanilla
    ga_o = GradientOptimizer(utility, max_iter=100_000)
    ga_o.optimize(R, initial_weights=w0.copy(), keep_history=True, max_time=60)
    wh, wt = ga_o._weights_history, ga_o.wall_time
    step = max(1, len(wh) // 500)
    results['GA'] = {
        'times': wt[::step].tolist()[:len(range(0, len(wh), step))],
        'utils': [cpt_eval(wh[i], R) for i in range(0, len(wh), step)]
    }

    # GA-Adam
    adam_o = AdamOptimizer(utility, lr=1e-2, max_iter=100_000, max_time=60)
    adam_o.optimize(R, initial_weights=w0.copy())
    wh, wt = adam_o._weights_history, adam_o.wall_time
    step = max(1, len(wh) // 500)
    results['Adam'] = {
        'times': wt[::step].tolist()[:len(range(0, len(wh), step))],
        'utils': [cpt_eval(wh[i], R) for i in range(0, len(wh), step)]
    }

    # Plot
    best_util = max(max(v['utils']) for v in results.values())
    markers = {'MV': 's', 'MM': 'o', 'CC': '^', 'GA': 'v', 'Adam': 'D'}
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        t_arr = np.maximum(np.array(data['times']), 1e-4)
        ax.plot(t_arr, data['utils'], '-' + markers.get(name, 'o'), label=name,
                markersize=3, linewidth=1.5)
    ax.axhline(best_util, color='gray', linestyle='--', alpha=0.7, label='Best utility')
    ax.set_xscale('log')
    ax.set_xlabel('Wall time (s)')
    ax.set_ylabel('CPT Utility')
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.set_title(f'Wall-time vs Utility ({label})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"Saved {savepath}")
    return results


if __name__ == '__main__':
    ALL_ASSETS = ['US_Equity_TR', 'Europe_Equity_TR', 'Japan_Equity_TR',
                  'EM_Equity_TR', 'US_GovtBond_TR', 'EU_GovtBond_TR',
                  'JP_GovtBond_TR', 'US_Bills', 'EU_Bills', 'JP_Bills',
                  'Commodities_TR', 'Gold', 'Silver']
    R_full, dates_full = load_asset_subset(CSV_PATH, ALL_ASSETS)

    mv_full = MeanVarianceFrontierOptimizer(utility)
    mv_full.optimize(R_full, samples=100)
    w_mv_full = mv_full.weights

    n_full = R_full.shape[1]
    w0_eq_full = np.ones(n_full) / n_full

    results_5a = run_walltime_experiment(R_full, w0_eq_full, 'equal-weight start',
                                         'figures/fig5a_walltime.png')
    results_5b = run_walltime_experiment(R_full, w_mv_full, 'MV start',
                                         'figures/fig5b_walltime_mv_start.png')