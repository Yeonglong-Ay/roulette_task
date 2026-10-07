# analyze_loss_chasing.py
"""
Loss-chasing analysis — extends Studer, Limbrick-Oldfield & Clark (2015), Exp. 2.
Journal of Behavioral Decision Making, 28, 239-249.

Two dependent variables
-----------------------
1. bet_high      (binary replication of the paper)
     1 if the participant bet ABOVE their own session mean, 0 otherwise.
     Controls for individual baseline bet levels.

2. bet_magnitude (new — captures effect size, not just direction)
     Bet minus that participant's session mean  (e.g. +2 means $2 above average).
     A person who always bets $8 has magnitude 0 on every trial; only
     deviations from their own baseline are analysed.  This avoids the
     confound where high-rollers inflate group means.

Streak models (Model 3 / Model 4)
----------------------------------
For each DV, a linear regression (OLS with participant dummies) is fitted
separately for winning streaks and losing streaks, with streak_length (1–4)
as a continuous predictor.  The slope tells you:

    Winning streaks: does the bet go up or down as the winning run grows?
    Losing streaks:  does the bet go up (loss chasing) as losses accumulate?

The figure shows:
    - Observed group means ± SEM at each streak length (dots + error bars)
    - OLS trend line fit to the individual trial data (replacing the
      logistic "predicted" line from the paper)
    - Two rows: top = bet_high, bottom = bet_magnitude
    - Two columns: winning streaks (left), losing streaks (right)

Logistic models (Models 1–3) are retained for the paper replication.

Usage
-----
    python analyze_loss_chasing.py data/

Output
------
    results/loss_chasing_analysis_data.csv
    results/loss_chasing_report.txt
    results/figure_streaks.png    ← main figure (OLS, two DVs)
    results/figure3_paper.png     ← paper replication (logistic, original layout)
"""

import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

MAX_STREAK_FOR_MODEL = 4   # paper excludes streak_length >= 5
MIN_COLOR_RUN        = 1
MAX_COLOR_RUN        = 5
MIN_OBS_PER_BIN      = 5    # bins with fewer observations are excluded from
                             # OLS fitting and shown as open markers with n label

# =============================================================================
# HELPERS
# =============================================================================

def load_data(path_arg: str) -> pd.DataFrame:
    """Load one CSV file or all pilot_results_*.csv under a directory."""
    p = Path(path_arg)
    if p.is_file():
        files = [p]
    else:
        files = list(p.rglob('pilot_results_*.csv'))

    if not files:
        sys.exit(f"No pilot_results_*.csv files found under: {path_arg}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} file(s).")
    return df


def fit_logistic(X: np.ndarray, y: np.ndarray):
    """
    Fit a logistic regression (no penalty) and return log-likelihood.
    Uses lbfgs with a very high C to approximate no regularization.
    """
    model = LogisticRegression(
        penalty=None, solver='lbfgs', max_iter=1000, fit_intercept=True
    )
    model.fit(X, y)
    probs   = model.predict_proba(X)[:, 1]
    probs   = np.clip(probs, 1e-10, 1 - 1e-10)
    log_lik = np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
    return model, log_lik


def null_log_likelihood(y: np.ndarray) -> float:
    """Log-likelihood of intercept-only model."""
    p = np.mean(y)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def mcfadden_r2(log_lik_full: float, log_lik_null: float) -> float:
    return 1 - (log_lik_full / log_lik_null)


def lr_chi2(log_lik_full: float, log_lik_null: float, df: int):
    """Likelihood-ratio chi-squared test."""
    chi2  = 2 * (log_lik_full - log_lik_null)
    p_val = stats.chi2.sf(chi2, df)
    return chi2, p_val


def wald_stats(model, X: np.ndarray, y: np.ndarray):
    """
    Compute Wald z-statistics, p-values, ORs, and 95% CIs for each predictor.
    Uses the observed Fisher information (Hessian of log-likelihood).
    """
    n, k    = X.shape
    p_hat   = model.predict_proba(X)[:, 1]
    W       = np.diag(p_hat * (1 - p_hat))

    # Design matrix including intercept
    X_full  = np.column_stack([np.ones(n), X])
    try:
        H       = X_full.T @ W @ X_full
        H_inv   = np.linalg.inv(H)
        se_all  = np.sqrt(np.diag(H_inv))
    except np.linalg.LinAlgError:
        se_all  = np.full(X_full.shape[1], np.nan)

    coefs     = np.concatenate([[model.intercept_[0]], model.coef_[0]])
    z_stats   = coefs / se_all
    p_vals    = 2 * stats.norm.sf(np.abs(z_stats))
    ors       = np.exp(coefs)
    ci_lower  = np.exp(coefs - 1.96 * se_all)
    ci_upper  = np.exp(coefs + 1.96 * se_all)

    return coefs, se_all, z_stats, p_vals, ors, ci_lower, ci_upper


def format_predictor_table(names, coefs, ses, zs, ps, ors, ci_lo, ci_hi) -> str:
    """Format a results table as a string."""
    lines = []
    header = (f"{'Predictor':<35} {'β':>8} {'SE':>8} {'z':>8} "
              f"{'p':>8} {'OR':>8} {'95% CI':>20}")
    lines.append(header)
    lines.append('-' * len(header))
    for name, b, se, z, p, o, lo, hi in zip(
            names, coefs, ses, zs, ps, ors, ci_lo, ci_hi):
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        ci  = f"[{lo:.2f}, {hi:.2f}]"
        lines.append(
            f"{name:<35} {b:>8.3f} {se:>8.3f} {z:>8.3f} "
            f"{p:>8.3f} {o:>8.3f} {ci:>20} {sig}"
        )
    return '\n'.join(lines)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to experimental trials only, compute derived columns,
    and add participant-mean-binarized bet_high.
    """
    # Experimental trials only
    df = df[df['phase'] == 'experimental'].copy()
    print(f"  Experimental trials: {len(df)}")

    # ----- bet_high: above personal mean (binary, paper replication) -----
    mean_bets = df.groupby('participant_id')['bet'].transform('mean')
    df['bet_high'] = (df['bet'] > mean_bets).astype(int)

    # ----- bet_magnitude: deviation from personal mean (continuous, new) -----
    # Captures how much MORE or LESS than their own baseline each participant bets.
    # A chronic high-roller and a cautious bettor are now on the same scale.
    df['bet_magnitude'] = df['bet'] - mean_bets

    # ----- choice_same: chose same color as previous wheel outcome -----
    # We need the previous wheel_outcome per participant, ordered by trial
    df = df.sort_values(['participant_id', 'global_trial'])
    df['prev_wheel_outcome'] = df.groupby('participant_id')['wheel_outcome'].shift(1)
    df['choice_same'] = (df['color_choice'] == df['prev_wheel_outcome']).astype(int)

    # ----- color_run_length: consecutive OUTCOME run length -----
    # Count how many consecutive identical wheel_outcomes end at the previous trial
    # (i.e. the run the participant SAW before placing this bet)
    # Compute color_run_length without groupby.apply (preserves all columns)
    # For each trial: count consecutive identical wheel_outcomes ending at the
    # PREVIOUS trial (i.e. the run the participant saw before placing this bet).
    df = df.sort_values(['participant_id', 'global_trial']).reset_index(drop=True)

    run_lengths = []
    for pid, group in df.groupby('participant_id', sort=False):
        outcomes = group['wheel_outcome'].tolist()
        for i in range(len(outcomes)):
            if i == 0:
                run_lengths.append((group.index[i], np.nan))
                continue
            length = 1
            for j in range(i - 1, 0, -1):
                if outcomes[j] == outcomes[j - 1]:
                    length += 1
                else:
                    break
            run_lengths.append((group.index[i], float(length)))

    idx_vals = pd.Series(
        {idx: val for idx, val in run_lengths}
    )
    df['color_run_length'] = idx_vals

    # ----- streak_length & prior_outcome from existing columns -----
    # streak_length is already in the CSV (raw, unclipped)
    # prior_outcome is already in the CSV
    df['prior_win'] = (df['prior_outcome'] == 'W').astype(int)

    # ----- baseline DVs -----
    # The original task used a fixed $5 midpoint of a $1-$9 scale. This task
    # uses a [1,2,4,8] scale whose neutral/default bet is $2. We provide TWO
    # baselines so they can be compared:
    #   personal-mean  (bet_high / bet_magnitude, computed above)
    #   fixed $2       (bet_high_2 / bet_mag_2) — "above the task default"
    # The _5-suffixed names drive the existing figures; point them at the $2
    # fixed baseline so the figures show the fixed-baseline view. (Set them to
    # bet_high/bet_magnitude instead if you prefer the personal-mean view.)
    FIXED_BASELINE = 2   # task default / neutral bet on the [1,2,4,8] scale
    df['bet_high_2'] = (df['bet'] > FIXED_BASELINE).astype(int)   # bet is 4 or 8
    df['bet_mag_2']  = df['bet'] - FIXED_BASELINE                 # $ above default

    return df


def exclude_invariant_participants(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """
    Exclude participants who show no variance on the DV.
    Paper excluded ~28% of participants who always chose the same color
    and/or always placed the same bet.
    """
    valid = df.groupby('participant_id')[dv].nunique()
    keep  = valid[valid > 1].index
    n_excl = (valid <= 1).sum()
    if n_excl > 0:
        pct = 100 * n_excl / len(valid)
        print(f"  Excluded {n_excl} participant(s) ({pct:.0f}%) with no variance on '{dv}'")
    return df[df['participant_id'].isin(keep)].copy()


# =============================================================================
# PARTICIPANT DUMMIES
# =============================================================================

def add_participant_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode participant_id (drop first = reference category)."""
    dummies = pd.get_dummies(df['participant_id'], prefix='pid', drop_first=True)
    return pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def get_pid_cols(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith('pid_')]


# =============================================================================
# MODEL 1 — Gambler's Fallacy on color choice
# =============================================================================

def run_model1(df: pd.DataFrame, report: list):
    report.append("\n" + "=" * 70)
    report.append("MODEL 1: Gambler's Fallacy — Color Run Length → Color Choice")
    report.append("=" * 70)
    report.append(
        "DV  : choice_same (1 = chose same color as previous outcome)\n"
        "IV  : color_run_length (1-5)\n"
        "Paper prediction: β < 0 (longer runs → less likely to choose same color)"
    )

    # Filter: need prior outcome (exclude trial 1) and run length 1-5
    d = df.dropna(subset=['color_run_length', 'choice_same']).copy()
    d = d[(d['color_run_length'] >= MIN_COLOR_RUN) &
          (d['color_run_length'] <= MAX_COLOR_RUN)]
    d = exclude_invariant_participants(d, 'choice_same')
    d = add_participant_dummies(d)

    pid_cols = get_pid_cols(d)
    y        = d['choice_same'].values

    # Null model (participant dummies only)
    X_null   = d[pid_cols].values.astype(float) if pid_cols else np.ones((len(d), 1))
    _, ll_null = fit_logistic(X_null, y)

    # Full model
    X_full   = np.column_stack([d['color_run_length'].values, X_null])
    model, ll_full = fit_logistic(X_full, y)

    chi2, p_chi2 = lr_chi2(ll_full, ll_null, df=1)
    r2           = mcfadden_r2(ll_full, ll_null)

    coefs, ses, zs, ps, ors, ci_lo, ci_hi = wald_stats(model, X_full, y)
    names = ['Intercept', 'color_run_length'] + pid_cols

    report.append(f"\nN trials (after exclusions): {len(d)}")
    report.append(f"McFadden pseudo-R²: {r2:.3f}")
    report.append(f"LR χ²(1) = {chi2:.2f}, p = {p_chi2:.4f}\n")
    report.append(format_predictor_table(
        ['Intercept', 'color_run_length'],
        coefs[:2], ses[:2], zs[:2], ps[:2], ors[:2], ci_lo[:2], ci_hi[:2]
    ))

    if ps[1] < .05:
        direction = "decreased" if coefs[1] < 0 else "increased"
        report.append(
            f"\n✓ SIGNIFICANT: Color choice {direction} as run length grew "
            f"(OR = {ors[1]:.2f}, p = {ps[1]:.4f}) — Gambler's Fallacy confirmed."
        )
    else:
        report.append(f"\n✗ Not significant (p = {ps[1]:.4f})")

    # --- Data for Figure 3A ---
    # Predicted probability of choosing same color at each run length
    run_lengths  = np.arange(MIN_COLOR_RUN, MAX_COLOR_RUN + 1)
    # Use mean of participant dummies (i.e. average participant) for prediction
    pid_mean     = X_null.mean(axis=0)
    pred_prob_3a = []
    for rl in run_lengths:
        x_row = np.concatenate([[rl], pid_mean])
        log_odds = model.intercept_[0] + model.coef_[0] @ x_row
        pred_prob_3a.append(1 / (1 + np.exp(-log_odds)))

    # Observed mean per run length
    obs_3a = (d.groupby('color_run_length')['choice_same']
               .mean()
               .reindex(run_lengths))

    plot_data_3a = {
        'run_lengths' : run_lengths,
        'predicted'   : np.array(pred_prob_3a),
        'observed'    : obs_3a.values,
        'obs_sem'     : (d.groupby('color_run_length')['choice_same']
                          .sem().reindex(run_lengths).values),
    }
    return model, plot_data_3a


# =============================================================================
# MODEL 2 — Run length × Color choice on BET
# =============================================================================

def run_model2(df: pd.DataFrame, report: list):
    report.append("\n" + "=" * 70)
    report.append("MODEL 2: Gambler's Fallacy × Run Length → Bet Size")
    report.append("=" * 70)
    report.append(
        "DV  : bet_high\n"
        "IVs : color_run_length, choice_same, run_length × choice_same\n"
        "Paper prediction: interaction significant — highest bets after long\n"
        "                  runs when participant predicts against the run."
    )

    d = df.dropna(subset=['color_run_length', 'choice_same']).copy()
    d = d[(d['color_run_length'] >= MIN_COLOR_RUN) &
          (d['color_run_length'] <= MAX_COLOR_RUN)]
    d = exclude_invariant_participants(d, 'bet_high')
    d = add_participant_dummies(d)

    pid_cols = get_pid_cols(d)
    y        = d['bet_high'].values
    X_null   = d[pid_cols].values.astype(float) if pid_cols else np.ones((len(d), 1))
    _, ll_null = fit_logistic(X_null, y)

    d['run_x_choice'] = d['color_run_length'] * d['choice_same']
    X_full = np.column_stack([
        d['color_run_length'].values,
        d['choice_same'].values,
        d['run_x_choice'].values,
        X_null
    ])
    model, ll_full = fit_logistic(X_full, y)
    chi2, p_chi2   = lr_chi2(ll_full, ll_null, df=3)
    r2             = mcfadden_r2(ll_full, ll_null)

    coefs, ses, zs, ps, ors, ci_lo, ci_hi = wald_stats(model, X_full, y)
    names_short = ['Intercept', 'color_run_length', 'choice_same', 'run × choice']

    report.append(f"\nN trials: {len(d)}")
    report.append(f"McFadden pseudo-R²: {r2:.3f}")
    report.append(f"LR χ²(3) = {chi2:.2f}, p = {p_chi2:.4f}\n")
    report.append(format_predictor_table(
        names_short,
        coefs[:4], ses[:4], zs[:4], ps[:4], ors[:4], ci_lo[:4], ci_hi[:4]
    ))

    interaction_p = ps[3]
    if interaction_p < .05:
        report.append(
            f"\n✓ SIGNIFICANT interaction (p = {interaction_p:.4f}): "
            "Bets were highest after long outcome runs when predicting against "
            "the run — Gambler's Fallacy extends to betting behavior."
        )
    else:
        report.append(f"\n✗ Interaction not significant (p = {interaction_p:.4f})")

    # --- Data for Figure 3B ---
    # Predicted P(bet_high) at each run length, separately for choice_same=1/0
    pid_mean    = X_null.mean(axis=0)
    run_lengths = np.arange(MIN_COLOR_RUN, MAX_COLOR_RUN + 1)
    pred_same, pred_diff = [], []
    for rl in run_lengths:
        for cs, store in [(1, pred_same), (0, pred_diff)]:
            x_row    = np.concatenate([[rl, cs, rl * cs], pid_mean])
            log_odds = model.intercept_[0] + model.coef_[0] @ x_row
            store.append(1 / (1 + np.exp(-log_odds)))

    obs_same = (d[d['choice_same'] == 1]
                .groupby('color_run_length')['bet_high']
                .agg(['mean', 'sem']).reindex(run_lengths))
    obs_diff = (d[d['choice_same'] == 0]
                .groupby('color_run_length')['bet_high']
                .agg(['mean', 'sem']).reindex(run_lengths))

    plot_data_3b = {
        'run_lengths' : run_lengths,
        'pred_same'   : np.array(pred_same),
        'pred_diff'   : np.array(pred_diff),
        'obs_same'    : obs_same['mean'].values,
        'obs_same_sem': obs_same['sem'].values,
        'obs_diff'    : obs_diff['mean'].values,
        'obs_diff_sem': obs_diff['sem'].values,
    }
    return plot_data_3b


# =============================================================================
# MODEL 3 — Feedback Streaks → Bet (the primary loss-chasing model)
# =============================================================================

def run_model3(df: pd.DataFrame, report: list):
    report.append("\n" + "=" * 70)
    report.append("MODEL 3: Feedback Streaks (W/L) → Bet Size  *** MAIN MODEL ***")
    report.append("=" * 70)
    report.append(
        "DV  : bet_high\n"
        "IVs : prior_win (W=1, L=0), streak_length (categorical 2 vs 1,\n"
        "      3 vs 1, 4 vs 1), prior_win × streak_length interaction.\n\n"
        "Paper finding (Exp 2):\n"
        "  - Bets INCREASED as LOSING streaks grew longer (loss chasing).\n"
        "  - Bets did NOT change reliably after WINNING streaks.\n"
        "  - Interaction significant at streak_length 3 and 4.\n\n"
        "Exclusions (matching paper):\n"
        "  - Trials with no prior history (streak_type = 'none')\n"
        "  - Streak length >= 5 (paper: 4.4% of trials)\n"
    )

    # Filter
    d = df[
        (df['streak_type'] != 'none') &
        (df['streak_type'] != 'practice') &
        (df['streak_length'] <= MAX_STREAK_FOR_MODEL) &
        (df['streak_length'] >= 1)
    ].copy()
    d = exclude_invariant_participants(d, 'bet_high')
    d = add_participant_dummies(d)

    pid_cols = get_pid_cols(d)
    y        = d['bet_high'].values
    X_null   = d[pid_cols].values.astype(float) if pid_cols else np.ones((len(d), 1))
    _, ll_null = fit_logistic(X_null, y)

    report.append(f"N trials (after exclusions): {len(d)}")
    report.append(
        f"  Streak length distribution:\n"
        f"{d['streak_length'].value_counts().sort_index().to_string()}\n"
    )

    # ---- Categorical streak_length dummies (reference = 1) ----
    for sl in [2, 3, 4]:
        d[f'sl_{sl}'] = (d['streak_length'] == sl).astype(int)

    sl_cols     = ['sl_2', 'sl_3', 'sl_4']
    interaction_cols = [f'pw_x_sl{sl}' for sl in [2, 3, 4]]
    for sl in [2, 3, 4]:
        d[f'pw_x_sl{sl}'] = d['prior_win'] * d[f'sl_{sl}']

    # Main model: prior_win + streak dummies + interactions
    X_full = np.column_stack([
        d['prior_win'].values,
        d[sl_cols].values,
        d[interaction_cols].values,
        X_null,
    ])
    model, ll_full = fit_logistic(X_full, y)
    chi2, p_chi2   = lr_chi2(ll_full, ll_null, df=7)
    r2             = mcfadden_r2(ll_full, ll_null)

    coefs, ses, zs, ps, ors, ci_lo, ci_hi = wald_stats(model, X_full, y)
    names_short = (
        ['Intercept', 'prior_win (W=1)', 'streak_len=2', 'streak_len=3', 'streak_len=4']
        + ['W×len=2', 'W×len=3', 'W×len=4']
    )

    report.append(f"McFadden pseudo-R²: {r2:.3f}")
    report.append(f"LR χ²(7) = {chi2:.2f}, p = {p_chi2:.4f}\n")
    report.append(format_predictor_table(
        names_short,
        coefs[:8], ses[:8], zs[:8], ps[:8], ors[:8], ci_lo[:8], ci_hi[:8]
    ))

    # ---- Decomposition: separate models for winning vs losing streaks ----
    # (paper ran these to decompose the interaction)
    report.append("\n" + "-" * 70)
    report.append("DECOMPOSITION: Separate models for winning and losing streaks")
    report.append("(Replicates the paper's decomposition analysis)")
    report.append("-" * 70)

    for outcome_label, win_val in [("LOSING streaks (prior_win = 0)", 0),
                                    ("WINNING streaks (prior_win = 1)", 1)]:
        sub = d[d['prior_win'] == win_val].copy()
        if len(sub) < 20:
            report.append(f"\n{outcome_label}: insufficient data (n={len(sub)})")
            continue

        y_sub    = sub['bet_high'].values
        if len(np.unique(y_sub)) < 2:
            report.append(f"\n{outcome_label}: only one class in DV — skipping.")
            continue
        pid_sub  = get_pid_cols(sub)
        X_null_s = sub[pid_sub].values.astype(float) if pid_sub else np.ones((len(sub), 1))
        _, ll_n  = fit_logistic(X_null_s, y_sub)

        X_s      = np.column_stack([sub['streak_length'].values, X_null_s])
        m_s, ll_s = fit_logistic(X_s, y_sub)
        c2, pc2   = lr_chi2(ll_s, ll_n, df=1)

        c, se, z, pv, o, lo, hi = wald_stats(m_s, X_s, y_sub)

        report.append(f"\n{outcome_label}  (n = {len(sub)} trials)")
        report.append(format_predictor_table(
            ['Intercept', 'streak_length'],
            c[:2], se[:2], z[:2], pv[:2], o[:2], lo[:2], hi[:2]
        ))
        report.append(f"LR χ²(1) = {c2:.2f}, p = {pc2:.4f}")

        if pv[1] < .05:
            direction = "INCREASED" if c[1] > 0 else "DECREASED"
            report.append(
                f"✓ SIGNIFICANT: Bet size {direction} as streak grew longer "
                f"(OR = {o[1]:.2f}, p = {pv[1]:.4f})"
            )
        else:
            report.append(f"✗ Not significant (p = {pv[1]:.4f})")

    # ---- Descriptive table: mean bet by streak type ----
    report.append("\n" + "-" * 70)
    report.append("DESCRIPTIVE: Mean bet and P(bet_high) by streak type")
    report.append("(Streak type = history carried INTO the trial)")
    report.append("-" * 70)

    streak_order = [f"{o}{n}" for o in ['L', 'W'] for n in [1, 2, 3, 4]]
    desc = (
        df[df['streak_type'].isin(streak_order)]
        .groupby('streak_type')
        .agg(
            n        = ('bet', 'count'),
            mean_bet = ('bet', 'mean'),
            sd_bet   = ('bet', 'std'),
            p_high   = ('bet_high', 'mean'),
        )
        .reindex(streak_order)
        .dropna(how='all')
    )
    report.append(desc.to_string(float_format='{:.3f}'.format))

    # --- Data for Figure 3C ---
    # Predicted P(bet_high) at streak_length 1-4 for wins and losses,
    # using a simple logistic regression on streak_length (continuous)
    # fitted separately for each outcome type — matches the paper's lines.
    streak_lengths = np.array([1, 2, 3, 4])
    plot_data_3c   = {}

    for label, win_val in [('loss', 0), ('win', 1)]:
        sub = d[d['prior_win'] == win_val].copy()
        if len(sub) < 10 or sub['bet_high'].nunique() < 2:
            plot_data_3c[label] = None
            continue

        pid_sub  = get_pid_cols(sub)
        X_pid    = sub[pid_sub].values.astype(float) if pid_sub else np.ones((len(sub), 1))
        X_s      = np.column_stack([sub['streak_length'].values, X_pid])
        m_s, _   = fit_logistic(X_s, sub['bet_high'].values)

        pid_mean_s = X_pid.mean(axis=0)
        pred_probs = []
        for sl in streak_lengths:
            x_row    = np.concatenate([[sl], pid_mean_s])
            log_odds = m_s.intercept_[0] + m_s.coef_[0] @ x_row
            pred_probs.append(1 / (1 + np.exp(-log_odds)))

        obs = (sub.groupby('streak_length')['bet_high']
                  .agg(['mean', 'sem'])
                  .reindex(streak_lengths))

        plot_data_3c[label] = {
            'streak_lengths': streak_lengths,
            'predicted'     : np.array(pred_probs),
            'observed'      : obs['mean'].values,
            'obs_sem'       : obs['sem'].values,
        }

    return d, plot_data_3c


# =============================================================================
# OLS STREAK ANALYSIS  (Models 4a / 4b)
# =============================================================================

def ols_streak_trends(df: pd.DataFrame, report: list) -> dict:
    """
    For each DV (bet_high, bet_magnitude) and each streak direction (W, L),
    fit an OLS model:

        DV ~ streak_length + participant_dummies

    Returns observed means/SEMs and OLS trend lines for plotting.

    The slope of streak_length tells you:
        Losing streaks, bet_magnitude slope > 0  →  loss chasing
        Winning streaks, bet_magnitude slope ≈ 0 →  no hot-hand escalation
    """
    report.append("\n" + "=" * 70)
    report.append("OLS STREAK TRENDS  (Models 4a & 4b)")
    report.append("=" * 70)
    report.append(
        "For each DV × streak direction, fit:\n"
        "  DV ~ streak_length (continuous 1-4) + participant_dummies\n\n"
        "bet_high      : 1 if bet > personal mean, else 0\n"
        "bet_magnitude : bet minus personal mean  (e.g. +2.0 = $2 above own baseline)\n\n"
        "Slope of streak_length = how much the DV changes per additional\n"
        "consecutive outcome in that direction."
    )

    streak_df = df[
        (df['streak_type'] != 'none') &
        (df['streak_type'] != 'practice') &
        (df['streak_length'] <= MAX_STREAK_FOR_MODEL) &
        (df['streak_length'] >= 1)
    ].copy()

    plot_data = {}

    for dv, dv_label, dv_unit in [
        ('bet_high',      'P(bet > personal mean)',  'proportion'),
        ('bet_magnitude', 'Bet − personal mean ($)', 'dollars'),
    ]:
        plot_data[dv] = {}

        for direction, win_val, dir_label in [
            ('win',  1, 'Winning streaks'),
            ('loss', 0, 'Losing streaks'),
        ]:
            sub = streak_df[streak_df['prior_win'] == win_val].copy()

            report.append(f"\n--- {dv_label}  |  {dir_label} (n={len(sub)} trials) ---")

            if len(sub) < 10 or sub[dv].nunique() < 2:
                report.append("  Insufficient data — skipping.")
                plot_data[dv][direction] = None
                continue

            # ---- OLS with participant dummies ----
            sub = add_participant_dummies(sub)
            pid_cols = get_pid_cols(sub)

            y  = sub[dv].values.astype(float)
            sl = sub['streak_length'].values.astype(float)
            X  = np.column_stack([
                np.ones(len(sub)),
                sl,
                sub[pid_cols].values.astype(float) if pid_cols else np.zeros((len(sub), 0))
            ])

            # OLS via least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            except Exception:
                report.append("  OLS failed — skipping.")
                plot_data[dv][direction] = None
                continue

            intercept = coeffs[0]
            slope     = coeffs[1]

            # Standard error of slope
            y_hat  = X @ coeffs
            resid  = y - y_hat
            n, p   = X.shape
            sigma2 = np.sum(resid ** 2) / max(n - p, 1)
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                se_slope = np.sqrt(sigma2 * XtX_inv[1, 1])
            except np.linalg.LinAlgError:
                se_slope = np.nan

            t_stat = slope / se_slope if se_slope > 0 else np.nan
            p_val  = 2 * stats.t.sf(abs(t_stat), df=max(n - p, 1)) if not np.isnan(t_stat) else np.nan

            # R²
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            sig = '***' if p_val < .001 else '**' if p_val < .01 else '*' if p_val < .05 else 'n.s.'
            report.append(
                f"  slope = {slope:+.4f}  SE = {se_slope:.4f}  "
                f"t = {t_stat:.3f}  p = {p_val:.4f} {sig}  R² = {r2:.3f}"
            )

            if not np.isnan(p_val) and p_val < .05:
                direction_word = "INCREASED" if slope > 0 else "DECREASED"
                report.append(
                    f"  ✓ {dv} {direction_word} as {dir_label.lower()} grew longer."
                )
            else:
                report.append(f"  ✗ No significant trend.")

            # ---- Observed means/SEMs per streak length ----
            streak_lengths = np.array([1, 2, 3, 4])
            obs  = (sub.groupby('streak_length')[dv]
                       .agg(['mean', 'sem'])
                       .reindex(streak_lengths))

            # OLS trend line (using group-mean participant dummies = average participant)
            pid_mean = (sub[pid_cols].values.astype(float).mean(axis=0)
                        if pid_cols else np.array([]))
            trend = []
            for sl_val in streak_lengths:
                x_row  = np.concatenate([[1.0, float(sl_val)], pid_mean])
                trend.append(x_row @ coeffs[:len(x_row)])

            plot_data[dv][direction] = {
                'streak_lengths': streak_lengths,
                'observed'      : obs['mean'].values,
                'obs_sem'       : obs['sem'].values,
                'trend'         : np.array(trend),
                'slope'         : slope,
                'p_val'         : p_val,
                'label'         : dir_label,
            }

    return plot_data


# =============================================================================
# FIGURE  —  Five panels, observed data + OLS trend, no predicted curves
# =============================================================================

def _panel(ax, x_vals, observed, obs_sem, trend,
           xlabel, ylabel, title, color, ref_line=None,
           bin_counts=None):
    """
    Draw one panel: OLS trend line + observed mean ± SEM dots.

    bin_counts : array of n per x_val. Bins below MIN_OBS_PER_BIN are shown
                 as open markers with their n printed below — excluded from OLS.
    """
    if bin_counts is None:
        bin_counts = np.full(len(x_vals), MIN_OBS_PER_BIN + 1)

    for i, (x, y, sem, n) in enumerate(zip(x_vals, observed, obs_sem, bin_counts)):
        if np.isnan(y):
            continue
        sparse = n < MIN_OBS_PER_BIN
        ax.errorbar(x, y, yerr=sem if not np.isnan(sem) else 0,
                    fmt='o', color=color if not sparse else 'white',
                    ms=7, lw=1.4, capsize=3, zorder=4,
                    markeredgecolor=color,
                    markeredgewidth=1.5 if sparse else 0.5)
        # n label below point
        ax.text(x, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.08,
                f'n={int(n)}', ha='center', va='top', fontsize=6.5,
                color='gray', transform=ax.get_xaxis_transform())

    # OLS trend — only over bins with sufficient observations
    valid_mask = bin_counts >= MIN_OBS_PER_BIN
    if valid_mask.sum() >= 2 and not np.all(np.isnan(trend[valid_mask])):
        valid_x = x_vals[valid_mask]
        valid_t = trend[valid_mask]
        ax.plot(valid_x, valid_t, '-', color=color, lw=2.0, zorder=3, alpha=0.85)

    if ref_line is not None:
        ax.axhline(ref_line, color='gray', lw=0.8, ls=':', zorder=1)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks(x_vals)


def _ols_trend(x: np.ndarray, y: np.ndarray,
               pid_dummies: np.ndarray, x_vals: np.ndarray,
               bin_counts: np.ndarray = None):
    """
    Fit OLS:  y ~ intercept + x + participant_dummies
    Returns (trend values at x_vals, slope, SE, t, p).

    If bin_counts is provided, trend values for sparse bins are set to NaN
    so the line is only drawn over well-sampled x positions.
    """
    X = np.column_stack([np.ones(len(x)), x, pid_dummies])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    intercept, slope = coeffs[0], coeffs[1]
    pid_mean = pid_dummies.mean(axis=0)

    trend = np.array([intercept + slope * v + pid_mean @ coeffs[2:]
                      for v in x_vals])

    # Mask trend values at sparse bins
    if bin_counts is not None:
        trend = np.where(bin_counts >= MIN_OBS_PER_BIN, trend, np.nan)

    # SE of slope
    y_hat  = X @ coeffs
    resid  = y - y_hat
    n, p   = X.shape
    sigma2 = np.sum(resid ** 2) / max(n - p, 1)
    try:
        se_slope = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    except np.linalg.LinAlgError:
        se_slope = np.nan

    t_stat = slope / se_slope if se_slope and se_slope > 0 else np.nan
    p_val  = (2 * stats.t.sf(abs(t_stat), df=max(n - p, 1))
              if not np.isnan(t_stat) else np.nan)

    return trend, slope, se_slope, t_stat, p_val


def plot_five_panels(df: pd.DataFrame, ols_data: dict,
                     out_path: Path, report: list):
    """
    Five-panel figure:

    Panel 1 — Color choice ~ color run length
        x: run length 1-5
        y: P(choose same color as previous)

    Panel 2 — Bet size (binary) ~ color run length
        x: run length 1-5
        y: P(bet > personal mean)

    Panel 3 — Bet magnitude ~ color run length
        x: run length 1-5
        y: bet − personal mean ($)

    Panel 4 — Bet size (binary) ~ win/loss streak length
        x: streak length 1-4
        y: P(bet > personal mean)
        Two series: winning (blue) and losing (red) streaks

    Panel 5 — Bet magnitude ~ win/loss streak length
        x: streak length 1-4
        y: bet − personal mean ($)
        Two series: winning (blue) and losing (red) streaks
    """
    plt.rcParams.update({
        'font.family'      : 'Arial',
        'font.size'        : 10,
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'axes.linewidth'   : 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    WIN_COLOR  = '#2166ac'   # blue
    LOSS_COLOR = '#d6604d'   # red
    RUN_COLOR  = '#4d4d4d'   # dark grey for run-length panels

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.subplots_adjust(wspace=0.45, left=0.05, right=0.98,
                        top=0.88, bottom=0.18)

    # ------------------------------------------------------------------
    # Shared data prep for panels 1-3  (color run length)
    # ------------------------------------------------------------------
    run_df = df.dropna(subset=['color_run_length']).copy()
    run_df = run_df[(run_df['color_run_length'] >= MIN_COLOR_RUN) &
                    (run_df['color_run_length'] <= MAX_COLOR_RUN)]
    run_df = add_participant_dummies(run_df)
    pid_cols = get_pid_cols(run_df)
    pid_mat  = run_df[pid_cols].values.astype(float) if pid_cols else np.zeros((len(run_df), 1))
    rl_vals  = np.arange(MIN_COLOR_RUN, MAX_COLOR_RUN + 1)

    # Per-bin observation counts — used to flag sparse bins
    bin_n_run = np.array([
        (run_df['color_run_length'] == rl).sum() for rl in rl_vals
    ])

    def run_obs(dv):
        return (run_df.groupby('color_run_length')[dv]
                      .agg(['mean', 'sem'])
                      .reindex(rl_vals))

    # OLS is fitted only on trials from bins that meet the minimum n threshold
    run_df_ols = run_df[run_df['color_run_length'].isin(
        rl_vals[bin_n_run >= MIN_OBS_PER_BIN]
    )]
    pid_mat_ols = (run_df_ols[pid_cols].values.astype(float)
                   if pid_cols else np.zeros((len(run_df_ols), 1)))

    # ------------------------------------------------------------------
    # Panel 1 — Color choice
    # ------------------------------------------------------------------
    obs1  = run_obs('choice_same')
    tr1, sl1, se1, t1, p1 = _ols_trend(
        run_df_ols['color_run_length'].values.astype(float),
        run_df_ols['choice_same'].values.astype(float),
        pid_mat_ols, rl_vals, bin_n_run)

    _panel(axes[0], rl_vals,
           obs1['mean'].values, obs1['sem'].values, tr1,
           'Color run length', 'P(choose same color)',
           '(1) Color choice', RUN_COLOR, ref_line=0.5,
           bin_counts=bin_n_run)
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    _annotate_slope(axes[0], sl1, p1, RUN_COLOR)

    # ------------------------------------------------------------------
    # Panel 2 — Bet size (binary) ~ run length
    # ------------------------------------------------------------------
    obs2  = run_obs('bet_high')
    tr2, sl2, se2, t2, p2 = _ols_trend(
        run_df_ols['color_run_length'].values.astype(float),
        run_df_ols['bet_high'].values.astype(float),
        pid_mat_ols, rl_vals, bin_n_run)

    _panel(axes[1], rl_vals,
           obs2['mean'].values, obs2['sem'].values, tr2,
           'Color run length', 'P(bet > personal mean)',
           '(2) Bet size ~ run', RUN_COLOR, ref_line=0.5,
           bin_counts=bin_n_run)
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    _annotate_slope(axes[1], sl2, p2, RUN_COLOR)

    # ------------------------------------------------------------------
    # Panel 3 — Bet magnitude ~ run length
    # ------------------------------------------------------------------
    obs3  = run_obs('bet_magnitude')
    tr3, sl3, se3, t3, p3 = _ols_trend(
        run_df_ols['color_run_length'].values.astype(float),
        run_df_ols['bet_magnitude'].values.astype(float),
        pid_mat_ols, rl_vals, bin_n_run)

    _panel(axes[2], rl_vals,
           obs3['mean'].values, obs3['sem'].values, tr3,
           'Color run length', 'Bet − personal mean ($)',
           '(3) Bet magnitude ~ run', RUN_COLOR, ref_line=0.0,
           bin_counts=bin_n_run)
    _annotate_slope(axes[2], sl3, p3, RUN_COLOR)

    # ------------------------------------------------------------------
    # Shared data prep for panels 4-5  (win/loss streak)
    # ------------------------------------------------------------------
    streak_df = df[
        (df['streak_type'] != 'none') &
        (df['streak_type'] != 'practice') &
        (df['streak_length'] <= MAX_STREAK_FOR_MODEL) &
        (df['streak_length'] >= 1)
    ].copy()
    sl_vals = np.array([1, 2, 3, 4])

    def streak_series(sub_df, dv):
        """Observed means/SEMs and OLS trend for one streak direction × DV."""
        sub_df = add_participant_dummies(sub_df)
        pid_c  = get_pid_cols(sub_df)
        pid_m  = (sub_df[pid_c].values.astype(float)
                  if pid_c else np.zeros((len(sub_df), 1)))
        obs    = (sub_df.groupby('streak_length')[dv]
                         .agg(['mean', 'sem'])
                         .reindex(sl_vals))
        trend, slope, se, t, p = _ols_trend(
            sub_df['streak_length'].values.astype(float),
            sub_df[dv].values.astype(float),
            pid_m, sl_vals)
        return obs['mean'].values, obs['sem'].values, trend, slope, p

    win_df  = streak_df[streak_df['prior_win'] == 1]
    loss_df = streak_df[streak_df['prior_win'] == 0]

    # ------------------------------------------------------------------
    # Panels 4 & 5 — two series each (win = blue, loss = red)
    # ------------------------------------------------------------------
    for ax_idx, dv, ylabel, title, ref in [
        (3, 'bet_high',      'P(bet > personal mean)',  '(4) Bet size ~ streak',      0.5),
        (4, 'bet_magnitude', 'Bet − personal mean ($)', '(5) Bet magnitude ~ streak', 0.0),
    ]:
        ax = axes[ax_idx]

        for sub_df, color, marker, label in [
            (win_df,  WIN_COLOR,  's', 'Win streak'),
            (loss_df, LOSS_COLOR, '^', 'Loss streak'),
        ]:
            if len(sub_df) < 5 or sub_df[dv].nunique() < 2:
                continue
            obs_m, obs_sem, trend, slope, p_val = streak_series(sub_df, dv)

            ax.plot(sl_vals, trend, '-', color=color, lw=2.0, zorder=3, alpha=0.85)
            ax.errorbar(sl_vals, obs_m, yerr=obs_sem,
                        fmt=marker, color=color, ms=7, lw=1.4,
                        capsize=3, zorder=4, label=label)

            # Slope annotation: win on top line, loss below
            y_pos = 0.93 if label == 'Win streak' else 0.82
            sig   = _sig_stars(p_val)
            sign  = '+' if slope >= 0 else ''
            ax.annotate(f"{label}: {sign}{slope:.3f} {sig}",
                        xy=(0.05, y_pos), xycoords='axes fraction',
                        fontsize=8, color=color, va='top')

        if ref is not None:
            ax.axhline(ref, color='gray', lw=0.8, ls=':', zorder=1)
        if dv == 'bet_high':
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

        ax.set_xlabel('Streak length', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xticks(sl_vals)
        ax.set_xticklabels(['1', '2', '3', '4'])
        ax.legend(frameon=False, fontsize=8, loc='lower right')

    # ------------------------------------------------------------------
    # Report OLS results for all five panels
    # ------------------------------------------------------------------
    report.append("\n" + "=" * 70)
    report.append("FIGURE PANEL OLS SLOPES")
    report.append("=" * 70)
    for label, slope, p_val in [
        ('Panel 1 — choice_same ~ run_length',    sl1, p1),
        ('Panel 2 — bet_high ~ run_length',        sl2, p2),
        ('Panel 3 — bet_magnitude ~ run_length',   sl3, p3),
    ]:
        sig = _sig_stars(p_val)
        report.append(f"  {label}: slope={slope:+.4f}  p={p_val:.4f} {sig}")

    for sub_df, dir_label, color in [
        (win_df,  'win streaks',  'Win'),
        (loss_df, 'loss streaks', 'Loss'),
    ]:
        for dv in ['bet_high', 'bet_magnitude']:
            if len(sub_df) >= 5 and sub_df[dv].nunique() >= 2:
                sub_df2 = add_participant_dummies(sub_df.copy())
                pid_c   = get_pid_cols(sub_df2)
                pid_m   = (sub_df2[pid_c].values.astype(float)
                           if pid_c else np.zeros((len(sub_df2), 1)))
                _, slope, _, _, p_val = _ols_trend(
                    sub_df2['streak_length'].values.astype(float),
                    sub_df2[dv].values.astype(float),
                    pid_m, sl_vals)
                sig = _sig_stars(p_val)
                report.append(
                    f"  Panel 4/5 — {dv} ~ {dir_label}: "
                    f"slope={slope:+.4f}  p={p_val:.4f} {sig}"
                )

    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def _sig_stars(p_val) -> str:
    if p_val is None or np.isnan(p_val): return ''
    return '***' if p_val < .001 else '**' if p_val < .01 else '*' if p_val < .05 else 'n.s.'


def _annotate_slope(ax, slope, p_val, color):
    sig  = _sig_stars(p_val)
    sign = '+' if slope >= 0 else ''
    ax.annotate(f"slope = {sign}{slope:.3f} {sig}",
                xy=(0.05, 0.93), xycoords='axes fraction',
                fontsize=8, color=color, va='top')
# =============================================================================
# FIGURE A  —  Trial-by-trial bet amount  (one subplot per participant)
# =============================================================================

def plot_trial_by_trial(raw: pd.DataFrame, out_path: Path):
    """
    One subplot per participant.
    x  : experimental trial number (1 … 72)
    y  : bet amount ($1–$9)
    Markers: green circle = Win outcome, red X = Loss outcome
    Horizontal dashed line at $5 (neutral baseline).
    Shaded region shows ±1 SD of that participant's bet distribution.
    """
    plt.rcParams.update({
        'font.family'      : 'Arial',
        'font.size'        : 9,
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'axes.linewidth'   : 0.8,
    })

    exp_df = raw[raw['phase'] == 'experimental'].copy()
    exp_df = exp_df.sort_values(['participant_id', 'global_trial'])
    participants = sorted(exp_df['participant_id'].unique())
    n = len(participants)

    # Layout: up to 3 columns
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.35,
                        left=0.07, right=0.97, top=0.92, bottom=0.10)

    for idx, pid in enumerate(participants):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub = exp_df[exp_df['participant_id'] == pid].reset_index(drop=True)

        # Renumber experimental trials 1…N within participant
        sub['exp_trial'] = range(1, len(sub) + 1)

        wins  = sub[sub['correct'] == True]
        losses = sub[sub['correct'] == False]

        # ── Bet line ─────────────────────────────────────────────────────────
        ax.plot(sub['exp_trial'], sub['bet'],
                color='#555555', lw=1.2, zorder=2, alpha=0.7)

        # ── ±1 SD shading ────────────────────────────────────────────────────
        mean_bet = sub['bet'].mean()
        sd_bet   = sub['bet'].std()
        ax.axhspan(mean_bet - sd_bet, mean_bet + sd_bet,
                   color='#cccccc', alpha=0.25, zorder=1)

        # ── Outcome markers ──────────────────────────────────────────────────
        ax.scatter(wins['exp_trial'],  wins['bet'],
                   marker='o', color='#2ecc71', s=40, zorder=4,
                   label='Win', edgecolors='none')
        ax.scatter(losses['exp_trial'], losses['bet'],
                   marker='x', color='#e74c3c', s=40, zorder=4,
                   label='Loss', linewidths=1.2)

        # ── $5 reference line ────────────────────────────────────────────────
        ax.axhline(5, color='#888888', lw=0.9, ls='--', zorder=1, alpha=0.8)

        ax.set_title(f"Participant {pid}\n"
                     f"mean={mean_bet:.1f}  SD={sd_bet:.1f}",
                     fontsize=9, pad=4)
        ax.set_xlabel('Trial', fontsize=8)
        ax.set_ylabel('Bet ($)', fontsize=8)
        ax.set_xlim(0, len(sub) + 1)
        ax.set_ylim(0.5, 9.5)
        ax.set_yticks([1, 3, 5, 7, 9])
        ax.tick_params(labelsize=8)

        if idx == 0:
            ax.legend(fontsize=7, frameon=False, loc='upper left',
                      markerscale=0.9, handletextpad=0.3)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Trial-by-trial bet amount\n"
                 "Green circle = Win  |  Red × = Loss  |  Dashed line = $5 baseline  |  "
                 "Shaded = ±1 SD",
                 fontsize=10, y=0.98)

    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure A saved: {out_path}")


# =============================================================================
# FIGURES B & C  —  Streak × bet  ($5 fixed baseline)
# =============================================================================

STREAK_ORDER = ['L1', 'L2', 'L3', 'L4', 'W1', 'W2', 'W3', 'W4']
#               ↑ ascending streak length in both arms
#               Loss arm: L1(len=1) → L4(len=4)   OLS streak_length maps directly to x 0→3
#               Win  arm: W1(len=1) → W4(len=4)   OLS streak_length maps directly to x 4→7
STREAK_X     = np.arange(len(STREAK_ORDER))   # 0 … 7
WIN_COLOR    = '#2166ac'
LOSS_COLOR   = '#d6604d'


def _streak_obs(df: pd.DataFrame, dv: str) -> tuple:
    """
    Observed group mean ± SEM per streak category (L4…W4).
    Returns (means, sems) aligned to STREAK_ORDER.
    """
    grp   = df.groupby('streak_type')[dv].agg(['mean', 'sem']).reindex(STREAK_ORDER)
    return grp['mean'].values, grp['sem'].values


def _streak_ols_trend(df: pd.DataFrame, dv: str,
                      streak_subset: list, x_positions: np.ndarray) -> tuple:
    """
    Fit OLS on one arm (win or loss streaks) with participant dummies.
    streak_subset : list of streak_type strings e.g. ['L1','L2','L3','L4']
    x_positions   : matching x positions for plotting the trend line

    Returns (trend_y, slope, p_val).
    """
    sub = df[df['streak_type'].isin(streak_subset)].copy()
    if len(sub) < 5 or sub[dv].nunique() < 2:
        return np.full(len(x_positions), np.nan), np.nan, np.nan

    sub = add_participant_dummies(sub)
    pid_cols = get_pid_cols(sub)
    pid_mat  = (sub[pid_cols].values.astype(float)
                if pid_cols else np.zeros((len(sub), 1)))

    # Use streak_length (1–4) as the continuous IV
    y  = sub[dv].values.astype(float)
    sl = sub['streak_length'].values.astype(float)
    X  = np.column_stack([np.ones(len(sub)), sl, pid_mat])

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = coeffs[0], coeffs[1]
    pid_mean = pid_mat.mean(axis=0)

    # Trend evaluated at streak lengths 1–4 (mapped to x_positions)
    trend = np.array([intercept + slope * sl_val + pid_mean @ coeffs[2:]
                      for sl_val in [1, 2, 3, 4]])

    # SE and p-value for slope
    y_hat  = X @ coeffs
    resid  = y - y_hat
    n, p   = X.shape
    sigma2 = np.sum(resid ** 2) / max(n - p, 1)
    try:
        se_slope = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
        t_stat   = slope / se_slope if se_slope > 0 else np.nan
        p_val    = 2 * stats.t.sf(abs(t_stat), df=max(n - p, 1))
    except Exception:
        p_val = np.nan

    return trend, slope, p_val


def _draw_streak_panel_collapsed(ax, df: pd.DataFrame, dv: str,
                                  ylabel: str, ref_line: float, title: str):
    """
    x-axis : streak length  1  2  3  4
    Two series:
        Red  triangles + line  →  losing streaks  (L1 … L4)
        Blue squares   + line  →  winning streaks (W1 … W4)

    OLS trend fitted separately for each arm.
    Loss chasing shows as a rising red line; hot-hand as a rising blue line.
    """
    sl_vals = np.array([1, 2, 3, 4])
    DODGE   = 0.08   # horizontal offset so overlapping markers stay visible

    for label, streaks, color, marker, arm_label in [
        ('loss', ['L1', 'L2', 'L3', 'L4'], LOSS_COLOR, '^', 'Loss streaks'),
        ('win',  ['W1', 'W2', 'W3', 'W4'], WIN_COLOR,  's', 'Win streaks'),
    ]:
        # Offset loss markers left, win markers right
        x_plot = sl_vals - DODGE if label == 'loss' else sl_vals + DODGE

        sub = df[df['streak_type'].isin(streaks)].copy()
        if len(sub) < 5:
            continue

        # Observed mean ± SEM at each streak length
        obs = (sub.groupby('streak_length')[dv]
                   .agg(['mean', 'sem'])
                   .reindex(sl_vals))

        ax.errorbar(x_plot, obs['mean'].values,
                    yerr=obs['sem'].values,
                    fmt=marker, color=color, ms=7, lw=1.4,
                    capsize=3, zorder=4, label=arm_label)

        # OLS trend drawn at true x positions (no dodge on the line)
        trend, slope, p_val = _streak_ols_trend(df, dv, streaks, sl_vals)
        if not np.any(np.isnan(trend)):
            ax.plot(sl_vals, trend, '-', color=color, lw=2.0,
                    zorder=3, alpha=0.85)

        # Slope annotation — always shown, even if slope is nan or ~0
        ypos = 0.90 if label == 'loss' else 0.80
        if np.isnan(slope):
            # Zero variance in this arm — slope undefined, show flat indicator
            ann_text = f"{arm_label}: slope ≈ 0.000 (flat)"
        else:
            sig  = _sig_stars(p_val)
            sign = '+' if slope >= 0 else ''
            ann_text = f"{arm_label}: slope = {sign}{slope:.3f} {sig}"
        ax.annotate(ann_text,
                    xy=(0.04, ypos), xycoords='axes fraction',
                    fontsize=8.5, color=color, va='top')

    ax.axhline(ref_line, color='gray', lw=0.9, ls=':', zorder=1)
    ax.set_xticks(sl_vals)
    ax.set_xticklabels(['1', '2', '3', '4'])
    ax.set_xlim(0.7, 4.3)   # padding so dodged markers at x=1 and x=4 aren't clipped
    ax.set_xlabel('Streak length', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    # Place legend in upper right, away from data points which cluster at bottom
    ax.legend(frameon=False, fontsize=9, loc='upper right')


def plot_streak_fixed_baseline(df: pd.DataFrame,
                               out_path_b: Path, out_path_c: Path):
    """
    Figure B : P(bet > $5) by streak length, win vs loss  ($5 fixed baseline)
    Figure C : Bet magnitude (bet − $5) by streak length  ($5 fixed baseline)
    """
    plt.rcParams.update({
        'font.family'      : 'Arial',
        'font.size'        : 10,
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'axes.linewidth'   : 0.8,
    })

    streak_df = df[
        (df['streak_type'].isin(STREAK_ORDER)) &
        (df['streak_length'] <= MAX_STREAK_FOR_MODEL)
    ].copy()

    # ── Figure B ─────────────────────────────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(6, 4.5))
    fig_b.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.16)

    _draw_streak_panel_collapsed(
        ax_b, streak_df, 'bet_high_2',
        ylabel='P(bet > $5)',
        ref_line=0.5,
        title='Bet size by streak length  (baseline = $5)',
    )
    ax_b.set_ylim(0, 1)
    ax_b.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    fig_b.text(0.5, 0.01,
               'Points = observed mean ± SEM  |  Lines = OLS trend  |  Baseline: $2',
               ha='center', va='bottom', fontsize=8, color='gray', style='italic')
    fig_b.savefig(out_path_b, dpi=180, bbox_inches='tight')
    plt.close(fig_b)
    print(f"Figure B saved: {out_path_b}")

    # ── Figure C ─────────────────────────────────────────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(6, 4.5))
    fig_c.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.16)

    _draw_streak_panel_collapsed(
        ax_c, streak_df, 'bet_mag_2',
        ylabel='Bet − $2',
        ref_line=0.0,
        title='Bet magnitude by streak length  (baseline = $5)',
    )
    fig_c.text(0.5, 0.01,
               'Points = observed mean ± SEM  |  Lines = OLS trend  |  Baseline: $2',
               ha='center', va='bottom', fontsize=8, color='gray', style='italic')
    fig_c.savefig(out_path_c, dpi=180, bbox_inches='tight')
    plt.close(fig_c)
    print(f"Figure C saved: {out_path_c}")


# =============================================================================
# ENDPOINT COMPARISON  —  L1 vs L4, L4 vs W4  (per participant + pooled)
# =============================================================================

def _ttest_two_groups(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Independent-samples Welch t-test (unequal variance).
    Returns (mean_a, mean_b, diff, t, df, p, cohen_d).
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean_a, mean_b = np.mean(a), np.mean(b)
    diff           = mean_b - mean_a        # positive = b > a
    t, p           = stats.ttest_ind(a, b, equal_var=False)

    # Cohen's d (pooled SD)
    pooled_sd = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d         = diff / pooled_sd if pooled_sd > 0 else np.nan
    df        = len(a) + len(b) - 2
    return mean_a, mean_b, diff, t, df, p, d


def plot_endpoint_comparison(df: pd.DataFrame, out_path: Path, report: list):
    """
    Two-panel figure addressing the PI's two specific questions:

    Panel 1 — L1 vs L4  (within-participant, loss arm only)
        Shows whether bets are higher after 4 consecutive losses
        compared to just 1 loss.  Tests loss chasing specifically.

    Panel 2 — L4 vs W4  (within-participant, loss vs win arm)
        Shows whether the worst losing streak produces higher bets
        than the best winning streak.  Tests the W/L asymmetry
        your PI is concerned about.

    Each panel:
        - One column per participant with jittered raw trial dots
        - Large coloured marker = participant mean
        - Connecting line shows direction of effect within person
        - t-test result annotated above each pair
        - Pooled estimate (fixed-effects) shown in rightmost column
    """
    plt.rcParams.update({
        'font.family'      : 'Arial',
        'font.size'        : 10,
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'axes.linewidth'   : 0.8,
    })

    streak_df    = df[df['streak_type'].isin(STREAK_ORDER)].copy()
    participants = sorted(df['participant_id'].unique())
    n_pids       = len(participants)
    DV           = 'bet'     # raw bet — most interpretable for the PI

    report.append("\n" + "=" * 70)
    report.append("ENDPOINT COMPARISONS  (L1 vs L4,  L4 vs W4)")
    report.append("=" * 70)
    report.append(
        "DV = raw bet amount ($1-$9)\n"
        "Test = independent-samples Welch t-test per participant, then pooled.\n"
        "Pooled = fixed-effects inverse-variance weighted mean of per-person diffs.\n"
    )

    fig, axes = plt.subplots(1, 2,
                             figsize=(5 * (n_pids + 1.5), 5.5),
                             sharey=False)
    fig.subplots_adjust(wspace=0.45, left=0.09, right=0.97,
                        top=0.85, bottom=0.14)

    comparisons = [
        # (ax, group_a_type, group_b_type, color_a, color_b, panel_title, question)
        (axes[0], 'L1', 'L4', '#f4a58a', '#d6604d',
         'L1  vs  L4\n(Does losing more = betting more?)',
         'Q1: L1 vs L4'),
        (axes[1], 'L4', 'W4', '#d6604d', '#2166ac',
         'L4  vs  W4\n(Is losing streak worse than winning streak?)' ,
         'Q2: L4 vs W4'),
    ]

    for ax, type_a, type_b, col_a, col_b, panel_title, q_label in comparisons:

        rng    = np.random.default_rng(42)
        x_cols = []   # x-centre per participant column

        all_diffs, all_ses = [], []   # for pooled estimate

        for col_idx, pid in enumerate(participants):
            sub    = streak_df[streak_df['participant_id'] == pid]
            bets_a = sub[sub['streak_type'] == type_a][DV].values.astype(float)
            bets_b = sub[sub['streak_type'] == type_b][DV].values.astype(float)

            x_centre = col_idx * 2.2   # spacing between participant columns
            x_cols.append(x_centre)

            for bets, x_offset, color in [
                (bets_a, -0.35, col_a),
                (bets_b, +0.35, col_b),
            ]:
                if len(bets) == 0:
                    continue
                jitter = rng.uniform(-0.12, 0.12, size=len(bets))
                ax.scatter(x_centre + x_offset + jitter, bets,
                           color=color, s=18, alpha=0.4, zorder=3,
                           edgecolors='none')
                ax.plot(x_centre + x_offset, np.mean(bets),
                        'o', color=color, ms=11, zorder=5,
                        markeredgecolor='white', markeredgewidth=1.2)

            # Connecting line between means
            if len(bets_a) > 0 and len(bets_b) > 0:
                ma, mb = np.mean(bets_a), np.mean(bets_b)
                line_color = '#555555'
                ax.plot([x_centre - 0.35, x_centre + 0.35], [ma, mb],
                        '-', color=line_color, lw=1.5, zorder=4, alpha=0.7)

            # t-test
            ma, mb, diff, t, df_t, p, d = _ttest_two_groups(bets_a, bets_b)
            na, nb = len(bets_a), len(bets_b)

            if not np.isnan(p):
                sig  = '***' if p < .001 else '**' if p < .01 else \
                       '*' if p < .05 else 'n.s.'
                sign = '+' if diff >= 0 else ''
                ax.text(x_centre, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 9.8,
                        f'{sign}{diff:.2f}\nt({df_t:.0f})={t:.2f}\np={p:.3f} {sig}',
                        ha='center', va='bottom', fontsize=7.5,
                        color='#333333')

                all_diffs.append({'diff': diff, 'se': np.sqrt(
                    np.var(bets_b, ddof=1)/nb + np.var(bets_a, ddof=1)/na
                ) if (na > 1 and nb > 1) else np.nan})

            report.append(
                f"  {q_label} | Participant {pid}:\n"
                f"    {type_a}: mean={ma:.2f} n={na}  "
                f"  {type_b}: mean={mb:.2f} n={nb}\n"
                f"    diff={sign if not np.isnan(diff) else ''}{diff:.3f}  "
                f"t({df_t:.0f})={t:.3f}  p={p:.4f}  d={d:.3f}"
            )

        # ── Pooled column ─────────────────────────────────────────────────────
        x_pool = (n_pids) * 2.2
        valid  = [r for r in all_diffs if not np.isnan(r['se']) and r['se'] > 0]

        if valid:
            weights = np.array([1 / r['se']**2 for r in valid])
            diffs   = np.array([r['diff'] for r in valid])
            p_diff  = np.sum(weights * diffs) / np.sum(weights)
            p_se    = np.sqrt(1 / np.sum(weights))
            z       = p_diff / p_se
            p_val   = 2 * stats.norm.sf(abs(z))
            ci95    = 1.96 * p_se

            sig  = '***' if p_val < .001 else '**' if p_val < .01 else \
                   '*' if p_val < .05 else 'n.s.'
            sign = '+' if p_diff >= 0 else ''

            # Error bar for pooled
            ax.errorbar(x_pool, p_diff + 5,    # offset to mid-scale for visual
                        yerr=ci95, fmt='D',
                        color='#333333', ms=10, lw=2.0,
                        capsize=5, zorder=6,
                        markeredgecolor='white', markeredgewidth=1.2,
                        label='Pooled')
            ax.text(x_pool,
                    ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 9.8,
                    f'{sign}{p_diff:.2f}\nz={z:.2f}\np={p_val:.3f} {sig}',
                    ha='center', va='bottom', fontsize=7.5,
                    color='#333333', fontweight='bold')

            report.append(
                f"\n  {q_label} | POOLED:\n"
                f"    diff = {sign}{p_diff:.4f}  SE = {p_se:.4f}  "
                f"z = {z:.3f}  p = {p_val:.4f} {sig}\n"
                f"    95% CI: [{p_diff - ci95:.4f}, {p_diff + ci95:.4f}]"
            )

        # ── Axes ──────────────────────────────────────────────────────────────
        all_x_ticks  = list(range(n_pids))
        tick_labels  = [f'P{pid}' if str(pid).isdigit() else str(pid)
                        for pid in participants]

        ax.set_xticks([i * 2.2 for i in range(n_pids)] + [x_pool])
        ax.set_xticklabels(tick_labels + ['Pooled'], fontsize=9)
        ax.set_ylim(0.5, 10.5)
        ax.set_yticks([1, 3, 5, 7, 9])
        ax.set_ylabel('Bet amount ($)', fontsize=10)
        ax.set_title(panel_title, fontsize=10, fontweight='bold', pad=10)
        ax.axhline(5, color='gray', lw=0.8, ls=':', alpha=0.7)

        # Legend
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], marker='o', color=col_a, ms=8, ls='none',
                   label=type_a),
            Line2D([0], [0], marker='o', color=col_b, ms=8, ls='none',
                   label=type_b),
        ], frameon=False, fontsize=9, loc='lower right')

    fig.suptitle(
        "Within-participant endpoint comparisons\n"
        "Large markers = participant mean  |  Small dots = individual trials  |  "
        "Line connects means  |  Dotted line = $5 baseline",
        fontsize=10, y=0.98,
    )

    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Endpoint comparison figure saved: {out_path}")


# =============================================================================
# FOREST PLOT  —  Individual slopes + pooled estimate
# =============================================================================

def _individual_ols_slope(sub: pd.DataFrame, dv: str) -> tuple:
    """
    Fit OLS for a single participant:  DV ~ streak_length  (no dummies needed)
    Returns (slope, se, t, p, n).
    """
    sub = sub.dropna(subset=[dv, 'streak_length'])
    if len(sub) < 4 or sub[dv].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, 0

    x = sub['streak_length'].values.astype(float)
    y = sub[dv].values.astype(float)
    X = np.column_stack([np.ones(len(x)), x])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        slope     = coeffs[1]
        y_hat     = X @ coeffs
        resid     = y - y_hat
        n, p      = X.shape
        sigma2    = np.sum(resid ** 2) / max(n - p, 1)
        se        = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
        t_stat    = slope / se if se > 0 else np.nan
        p_val     = 2 * stats.t.sf(abs(t_stat), df=max(n - p, 1))
        return slope, se, t_stat, p_val, len(sub)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, 0


def _pooled_slope(rows: list[dict]) -> tuple:
    """
    Fixed-effects pooled estimate: inverse-variance weighted mean of slopes.
    Returns (pooled_slope, pooled_se, z, p).
    """
    valid = [r for r in rows if not np.isnan(r['slope']) and r['se'] > 0]
    if not valid:
        return np.nan, np.nan, np.nan, np.nan

    weights    = np.array([1 / r['se'] ** 2 for r in valid])
    slopes     = np.array([r['slope'] for r in valid])
    pooled     = np.sum(weights * slopes) / np.sum(weights)
    pooled_se  = np.sqrt(1 / np.sum(weights))
    z          = pooled / pooled_se if pooled_se > 0 else np.nan
    p_val      = 2 * stats.norm.sf(abs(z)) if not np.isnan(z) else np.nan
    return pooled, pooled_se, z, p_val


def plot_forest(df: pd.DataFrame, out_path: Path, report: list):
    """
    Forest plot of individual-participant OLS slopes for:
        - Loss streak → bet_high    (left panel)
        - Loss streak → bet_mag_2   (right panel)
        - Win  streak → bet_high    (left panel, lighter colour)
        - Win  streak → bet_mag_2   (right panel, lighter colour)

    Each row = one participant.
    Marker size proportional to n (number of trials in that arm).
    Pooled fixed-effects estimate shown at the bottom with a diamond.
    Vertical dashed line at zero = no effect.

    Reading the plot
    ----------------
    - If individual CIs cross zero but all estimates sit on the same side,
      that is consistent directional signal despite individual non-significance.
    - The pooled estimate collapses individual uncertainty and shows whether
      the group-level effect is reliable.
    """
    plt.rcParams.update({
        'font.family'      : 'Arial',
        'font.size'        : 10,
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'axes.linewidth'   : 0.8,
    })

    participants = sorted(df['participant_id'].unique())
    streak_df    = df[
        (df['streak_type'].isin(STREAK_ORDER)) &
        (df['streak_length'] <= MAX_STREAK_FOR_MODEL) &
        (df['streak_length'] >= 1)
    ].copy()

    report.append("\n" + "=" * 70)
    report.append("FOREST PLOT — Individual OLS slopes (loss & win streaks)")
    report.append("=" * 70)
    report.append(
        "Each participant's slope for DV ~ streak_length fitted separately.\n"
        "Pooled estimate = fixed-effects inverse-variance weighted mean.\n"
        "Key question: are individual slopes consistently in the same direction\n"
        "even if no single participant reaches significance?"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 1.2 * (len(participants) + 3))))
    fig.subplots_adjust(wspace=0.55, left=0.18, right=0.97, top=0.90, bottom=0.12)

    configs = [
        # (ax, dv, title, loss_color, win_color)
        (axes[0], 'bet_high_2',
         'P(bet > $5)  ~  streak length',
         '#d6604d', '#f4a58a'),
        (axes[1], 'bet_mag_2',
         'Bet − $2  ~  streak length',
         '#d6604d', '#f4a58a'),
    ]

    for ax, dv, title, loss_col, win_col in configs:

        # y positions: participants top → bottom, then gap, then pooled
        y_positions = list(range(len(participants), 0, -1))
        y_pooled    = -0.5

        loss_rows, win_rows = [], []

        for pid, y_pos in zip(participants, y_positions):
            sub      = streak_df[streak_df['participant_id'] == pid]
            sub_loss = sub[sub['prior_win'] == 0]
            sub_win  = sub[sub['prior_win'] == 1]

            for sub_arm, rows, color, offset, arm_label in [
                (sub_loss, loss_rows, loss_col, -0.18, 'Loss'),
                (sub_win,  win_rows,  win_col,  +0.18, 'Win'),
            ]:
                slope, se, t_stat, p_val, n = _individual_ols_slope(sub_arm, dv)
                rows.append({'pid': pid, 'slope': slope, 'se': se,
                              'p_val': p_val, 'n': n})

                if np.isnan(slope):
                    ax.text(0, y_pos + offset, f'{pid}: insufficient data',
                            va='center', ha='center', fontsize=7, color='gray')
                    continue

                ci95 = 1.96 * se
                # CI line
                ax.plot([slope - ci95, slope + ci95],
                        [y_pos + offset, y_pos + offset],
                        '-', color=color, lw=1.5, zorder=3)
                # Marker — size proportional to n
                ms = 4 + min(n / 20, 6)
                sig_marker = '*' if (p_val is not None and p_val < .05) else ''
                ax.plot(slope, y_pos + offset,
                        'D', color=color, ms=ms, zorder=4,
                        markeredgecolor='white', markeredgewidth=0.5)

                # p-value label to the right
                p_str = f'p={p_val:.3f}{sig_marker}' if not np.isnan(p_val) else ''
                ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] != 1 else 0.5,
                        y_pos + offset, f'  {p_str}',
                        va='center', ha='left', fontsize=6.5, color=color)

        # ── Pooled estimates ──────────────────────────────────────────────────
        for rows, color, offset, arm_label in [
            (loss_rows, loss_col, -0.18, 'Loss pooled'),
            (win_rows,  win_col,  +0.18, 'Win pooled'),
        ]:
            p_slope, p_se, z, p_val = _pooled_slope(rows)
            if np.isnan(p_slope):
                continue

            ci95 = 1.96 * p_se
            # Diamond marker for pooled
            diamond_x = [p_slope - ci95, p_slope, p_slope + ci95, p_slope]
            diamond_y = [y_pooled + offset,
                         y_pooled + offset + 0.25,
                         y_pooled + offset,
                         y_pooled + offset - 0.25]
            ax.fill(diamond_x, diamond_y, color=color, zorder=5, alpha=0.9)
            ax.plot([p_slope - ci95, p_slope + ci95],
                    [y_pooled + offset, y_pooled + offset],
                    '-', color=color, lw=2.0, zorder=4)

            sig = '***' if p_val < .001 else '**' if p_val < .01 else \
                  '*' if p_val < .05 else 'n.s.' if not np.isnan(p_val) else ''
            ax.text(p_slope, y_pooled + offset - 0.35,
                    f'{arm_label}: slope={p_slope:+.3f}, p={p_val:.3f} {sig}',
                    va='top', ha='center', fontsize=7.5, color=color,
                    fontweight='bold')

            report.append(
                f"\n  {dv} | {arm_label}:\n"
                f"    pooled slope = {p_slope:+.4f}  SE = {p_se:.4f}  "
                f"z = {z:.3f}  p = {p_val:.4f} {sig}"
            )

        # ── Axes formatting ───────────────────────────────────────────────────
        ax.axvline(0, color='black', lw=0.9, ls='--', zorder=1, alpha=0.5)

        # Divider between participants and pooled
        ax.axhline(0.35, color='gray', lw=0.8, ls='-', alpha=0.4)

        ax.set_yticks(y_positions + [y_pooled])
        ax.set_yticklabels(
            [f'P{pid}' if str(pid).isdigit() else str(pid)
             for pid in participants] + ['Pooled'],
            fontsize=9
        )
        ax.set_xlabel('OLS slope  (per streak step)', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

        # Legend
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], marker='D', color=loss_col, ms=6, ls='none',
                   label='Loss streak'),
            Line2D([0], [0], marker='D', color=win_col,  ms=6, ls='none',
                   label='Win streak'),
        ], frameon=False, fontsize=8, loc='lower right')

    fig.suptitle(
        'Forest plot of individual OLS slopes  (95% CI)\n'
        'Diamond = pooled fixed-effects estimate  |  '
        '* p<.05  ** p<.01  *** p<.001',
        fontsize=10, y=0.98,
    )

    # Add interpretation note at bottom
    fig.text(
        0.5, 0.01,
        'Consistent direction across participants (even with wide CIs) indicates '
        'reliable signal despite individual non-significance.',
        ha='center', va='bottom', fontsize=8, color='gray', style='italic',
    )

    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Forest plot saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/'

    print("\n" + "=" * 70)
    print("LOSS CHASING ANALYSIS")
    print("=" * 70)

    raw = load_data(data_path)

    print("\nPreparing data...")
    df = prepare_data(raw)

    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    analysis_csv = out_dir / 'loss_chasing_analysis_data.csv'
    df.to_csv(analysis_csv, index=False)
    print(f"Analysis dataset saved: {analysis_csv}")

    report = []
    report.append("=" * 70)
    report.append("LOSS CHASING ANALYSIS")
    report.append("=" * 70)
    report.append(
        "\nAnalysis logic:\n"
        "  The STREAK is the IV (cause); the BET CHANGE is the DV (effect).\n"
        "  streak_type describes the history the participant carries INTO\n"
        "  each trial when placing their bet — always forward-looking:\n"
        "  'Given I just experienced [Streak X], how do I bet on the NEXT trial?'\n\n"
        "  W1 = 'I just won once; now deciding my bet.'\n"
        "  L4 = 'I just lost 4 times in a row; now deciding my bet.'\n"
        "       ^^^ PRIMARY measure of loss chasing ^^^"
    )

    report.append(f"\nParticipants: {df['participant_id'].nunique()}")
    report.append(f"Total experimental trials: {len(df)}")

    # Statistical models (kept for report)
    run_model1(df, report)
    run_model2(df, report)
    run_model3(df, report)
    ols_streak_trends(df, report)

    # Five-panel figure (personal-mean baseline, existing)
    fig_path = out_dir / 'figure_panels.png'
    plot_five_panels(df, {}, fig_path, report)

    # Figure A — trial-by-trial bet amount per participant
    fig_a_path = out_dir / 'figure_A_trial_by_trial.png'
    plot_trial_by_trial(raw, fig_a_path)

    # Figures B & C — streak × bet with $5 fixed baseline
    fig_b_path = out_dir / 'figure_B_streak_bet_high.png'
    fig_c_path = out_dir / 'figure_C_streak_bet_magnitude.png'
    plot_streak_fixed_baseline(df, fig_b_path, fig_c_path)

    # Endpoint comparisons — L1 vs L4, L4 vs W4
    fig_e_path = out_dir / 'figure_E_endpoint.png'
    plot_endpoint_comparison(df, fig_e_path, report)

    # Forest plot — individual slopes + pooled estimate
    fig_forest_path = out_dir / 'figure_D_forest.png'
    plot_forest(df, fig_forest_path, report)

    report_text = '\n'.join(report)
    report_path = out_dir / 'loss_chasing_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nReport saved: {report_path}")

    return report_path, analysis_csv, fig_path, fig_a_path, fig_b_path, fig_c_path


if __name__ == '__main__':
    main()
