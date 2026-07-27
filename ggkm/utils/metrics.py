import numpy as np


def uno_c_index_rmst(
    S_pred,
    t_eval,
    delta_eval,
    t_train,
    delta_train,
    t_grid,
    tau=None,
):

    if tau is None:
        tau = t_grid[-1]

    if not np.all(np.isfinite(S_pred)):
        return np.nan

    rmst = np.trapezoid(S_pred, t_grid, axis=1)
    risk_score = -rmst

    G = censoring_survival_function(
        t_train,
        delta_train,
    )

    G_t = np.maximum(G(t_eval), 1e-12)
    weights = 1.0 / (G_t**2)
    numerator = 0.0
    denominator = 0.0
    n = len(t_eval)

    for i in range(n):
        if delta_eval[i] != 1:
            continue
        if t_eval[i] >= tau:
            continue
        comparable = t_eval[i] < t_eval
        if not np.any(comparable):
            continue
        w = weights[i]
        denominator += w * np.sum(comparable)
        numerator += w * np.sum(risk_score[i] > risk_score[comparable])

    if denominator == 0:
        return np.nan

    return numerator / denominator


def kaplan_meier(t, delta):
    order = np.argsort(t)
    t_ord = t[order]
    d_ord = delta[order]
    unique_t = np.unique(t_ord)

    surv = 1.0
    km_t = [0.0]
    km_s = [1.0]

    for ti in unique_t:
        events = d_ord[t_ord == ti].sum()
        at_risk = (t_ord >= ti).sum()
        if at_risk > 0:
            surv *= 1.0 - events / at_risk

        km_t.append(ti)
        km_s.append(surv)

    return np.array(km_t), np.array(km_s)


def censoring_survival_function(t_train, delta_train):
    km_times, km_surv = kaplan_meier(t_train, 1 - delta_train)

    def G(t_query):
        t_query = np.asarray(t_query)
        idx = np.clip(
            np.searchsorted(km_times, t_query, side="right") - 1, 0, len(km_surv) - 1
        )

        return km_surv[idx]

    return G


def integrated_brier_score(
    S_pred,
    t_eval,
    delta_eval,
    t_train,
    delta_train,
    t_grid,
):

    if not np.all(np.isfinite(S_pred)):
        return np.inf

    G = censoring_survival_function(t_train, delta_train)
    bs_grid = np.zeros(len(t_grid))
    G_y = np.maximum(G(t_eval), 1e-12)

    for j, t in enumerate(t_grid):

        event_before = (t_eval <= t) & (delta_eval == 1)
        alive_after = t_eval > t
        G_t = max(float(G(t)), 1e-12)
        term1 = event_before * S_pred[:, j] ** 2 / G_y
        term2 = alive_after * (1.0 - S_pred[:, j]) ** 2 / G_t
        bs_grid[j] = np.mean(term1 + term2)

    tau = t_grid[-1] - t_grid[0]

    return np.trapezoid(bs_grid, t_grid) / tau


def cure_tpr_fpr(
    pi_hat,
    c_grid=None,
):
    pi_hat = np.asarray(pi_hat, dtype=float)

    if not np.all(np.isfinite(pi_hat)):
        return None, None, None

    pi_hat = np.clip(pi_hat, 0.0, 1.0)

    if c_grid is None:
        c_grid = np.linspace(0.0, 1.0, 101)
    else:
        c_grid = np.asarray(c_grid, dtype=float)

    w_pos = 1.0 - pi_hat
    w_neg = pi_hat

    denom_tpr = np.sum(w_pos)
    denom_fpr = np.sum(w_neg)

    if denom_tpr <= 0 or denom_fpr <= 0:
        return c_grid, np.full_like(c_grid, np.nan), np.full_like(c_grid, np.nan)

    tpr = np.empty_like(c_grid, dtype=float)
    fpr = np.empty_like(c_grid, dtype=float)

    for j, c in enumerate(c_grid):
        selected = pi_hat <= c
        tpr[j] = np.sum(selected * w_pos) / denom_tpr
        fpr[j] = np.sum(selected * w_neg) / denom_fpr

    return c_grid, tpr, fpr


def auc_cure(
    pi_hat,
    c_grid=None,
):
    c_grid, tpr, fpr = cure_tpr_fpr(pi_hat, c_grid=c_grid)

    if tpr is None or fpr is None:
        return np.nan, None, None, None

    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    auc = np.trapezoid(tpr_sorted, fpr_sorted)

    return auc, tpr_sorted, fpr_sorted, c_grid[order]
