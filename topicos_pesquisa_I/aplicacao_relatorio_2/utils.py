import numpy as np


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
