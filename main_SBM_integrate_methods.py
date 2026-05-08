import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import pandas as pd
import time

# ==================== PARAMETERS ====================

unified_M_BIG= 1e+4
M_BIG = 1e+7
M_SMALL = 1e-6
EPSILON = 1e-6

NUM_INPUTS = 10
NUM_OUTPUTS = 15
NUM_DMUS = 400
RANDOM_SEED = 42

SHOW_DETAILS = True
OUTPUT_FILE = 'dea_results.xlsx'
# ====================================================

def random_dea_data_set(m, s, n, seed):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(1, 100, size=(m, n))
    Y = np.random.uniform(1, 100, size=(s, n))
    return X, Y


################################# SBM-SSBM and Enhanced SBM-SSBM  ########################
      # SBM-SSBM andmy modification on  SBM-SSBM
#####################################################################################

def solve_SBM_scipy(X, Y, p):
    m, n = X.shape
    s = Y.shape[0]
    xp = X[:, p]
    yp = Y[:, p]
    
    n_vars = 1 + m + s + n
    c = np.zeros(n_vars)
    c[0] = 1
    c[1:1+m] = -1 / (m * xp)
    
    A_eq = []
    b_eq = []
    
    eq_row = np.zeros(n_vars)
    eq_row[0] = 1
    eq_row[1+m:1+m+s] = 1 / (s * yp)
    A_eq.append(eq_row)
    b_eq.append(1)
    
    for i in range(m):
        eq_row = np.zeros(n_vars)
        eq_row[0] = -xp[i]
        eq_row[1+i] = 1
        eq_row[1+m+s:] = X[i, :]
        A_eq.append(eq_row)
        b_eq.append(0)
    
    for r in range(s):
        eq_row = np.zeros(n_vars)
        eq_row[0] = -yp[r]
        eq_row[1+m+r] = -1
        eq_row[1+m+s:] = Y[r, :]
        A_eq.append(eq_row)
        b_eq.append(0)
    
    bounds = [(0, None)] * n_vars
    result = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method='highs')
    
    if result.success:
        return result.fun, result.x[1+m+s:]
    return np.nan, np.zeros(n)

def solve_SSBM_scipy(X, Y, p):
    m, n = X.shape
    s = Y.shape[0]
    xp = X[:, p]
    yp = Y[:, p]
    
    peer_idx = [i for i in range(n) if i != p]
    X_peer = X[:, peer_idx]
    Y_peer = Y[:, peer_idx]
    n_peer = len(peer_idx)
    
    n_vars = 1 + m + s + n_peer
    c = np.zeros(n_vars)
    c[0] = 1
    c[1:1+m] = 1 / (m * xp)
    
    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []
    
    eq_row = np.zeros(n_vars)
    eq_row[0] = 1
    eq_row[1+m:1+m+s] = -1 / (s * yp)
    A_eq.append(eq_row)
    b_eq.append(1)
    
    for i in range(m):
        ub_row = np.zeros(n_vars)
        ub_row[0] = -xp[i]
        ub_row[1+i] = -1
        ub_row[1+m+s:] = X_peer[i, :]
        A_ub.append(ub_row)
        b_ub.append(0)
    
    for r in range(s):
        ub_row = np.zeros(n_vars)
        ub_row[0] = yp[r]
        ub_row[1+m+r] = -1
        ub_row[1+m+s:] = -Y_peer[r, :]
        A_ub.append(ub_row)
        b_ub.append(0)
    
    bounds = [(0, None)] * n_vars
    result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                     bounds=bounds, method='highs')
    
    return max(result.fun, 1.0) if result.success else 1.0


 
def SBM_SSBM_algorithm(X, Y, epsilon, use_enhanced=True, tol=1e-8):
    X_full = X.copy()
    Y_full = Y.copy()

    n = X.shape[1]
    eff_scores = np.ones(n)
    IE = []

    # DMUهایی که هنوز باید evaluate شوند
    to_evaluate = list(range(n))

    # DMUهایی که هنوز در reference set مجازند
    reference_alive = np.ones(n, dtype=bool)

    start_total = time.time()

    while to_evaluate:
        p = to_evaluate.pop(0)

        # مرجع فعلی
        ref_idx = np.where(reference_alive)[0].tolist()

        # اگر p در reference set نیست، موقتاً برای ارزیابی خودش اضافه شود
        if p not in ref_idx:
            ref_idx = sorted(ref_idx + [p])

        X_ref = X_full[:, ref_idx]
        Y_ref = Y_full[:, ref_idx]
        p_local = ref_idx.index(p)

        delta_p, lambda_opt = solve_SBM_scipy(X_ref, Y_ref, p_local)
        eff_scores[p] = delta_p

        active_global = []
        if use_enhanced:
            active_local = np.where(lambda_opt > tol)[0].tolist()
            active_global = [ref_idx[j] for j in active_local]
            IE.extend(active_global)
        else:
            if delta_p >= 1 - epsilon:
                IE.append(p)

        # -----------------------------
        # 1) activeها را از evaluation حذف کن
        # -----------------------------
        if use_enhanced and active_global:
            active_set = set(active_global)
            to_evaluate = [j for j in to_evaluate if j not in active_set]

        # -----------------------------
        # 2) dominatedها را از reference set حذف کن
        # -----------------------------
        if delta_p >= 1 - epsilon:
            x_o = X_full[:, p]
            y_o = Y_full[:, p]

            for j in range(n):
                if j == p:
                    continue
                if not reference_alive[j]:
                    continue

                dominated = (
                    np.all(X_full[:, j] >= x_o - tol) and
                    np.all(Y_full[:, j] <= y_o + tol)
                )

                if dominated:
                    reference_alive[j] = False

    IE = sorted(set(IE))
    super_eff = eff_scores.copy()

    for i in IE:
        super_eff[i] = solve_SSBM_scipy(X_full, Y_full, i)

    total_time = time.time() - start_total
    return eff_scores, IE, super_eff, total_time





def SBM_SSBM_algorithm1(X, Y, epsilon, use_enhanced=True):
    n = X.shape[1]
    IE = []
    J = list(range(n))
    eff_scores = np.ones(n)
    
    start_total = time.time()
    
    while J:
        p = J[0]
        delta_p, lambda_opt = solve_SBM_scipy(X, Y, p)
        eff_scores[p] = delta_p
        
        if use_enhanced:
            active_idx = np.where(lambda_opt > 1e-8)[0].tolist()
            IE.extend(active_idx)
        else:
            if delta_p >= 1 - epsilon:
                IE.append(p)
        
        J = [j for j in J if j not in IE and j != p]
    
    super_eff = eff_scores.copy()
    IE = list(set(IE))
    
    for i in IE:
        super_eff[i] = solve_SSBM_scipy(X, Y, i)
    
    total_time = time.time() - start_total
    return eff_scores, IE, super_eff, total_time

def unified_SBM_SuperSBM_scipy(X, Y, p):
    m, n = X.shape
    s = Y.shape[0]
    x_p = X[:, p]
    y_p = Y[:, p]
    
    num_vars = 1 + m + s + (n-1) + 1
    c = np.zeros(num_vars)
    c[0] = 1
    c[1:1+m] = -1/(m * x_p)
    
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    eq1 = np.zeros(num_vars)
    eq1[0] = 1
    eq1[1+m:1+m+s] = 1/(s * y_p)
    A_eq.append(eq1)
    b_eq.append(1)
    
    X_excl = np.delete(X, p, axis=1)
    Y_excl = np.delete(Y, p, axis=1)
    
    for i in range(m):
        row = np.zeros(num_vars)
        row[0] = -x_p[i]
        row[1+i] = 1
        row[1+m+s:1+m+s+(n-1)] = X_excl[i, :]
        A_ub.append(row)
        b_ub.append(0)
    
    for r in range(s):
        row = np.zeros(num_vars)
        row[0] = y_p[r]
        row[1+m+r] = 1
        row[1+m+s:1+m+s+(n-1)] = -Y_excl[r, :]
        A_ub.append(row)
        b_ub.append(0)
    
    for i in range(m):
        row = np.zeros(num_vars)
        row[1+i] = -1
        row[-1] = -unified_M_BIG
        A_ub.append(row)
        b_ub.append(0)
    
    for i in range(m):
        row = np.zeros(num_vars)
        row[1+i] = 1
        row[-1] = unified_M_BIG
        A_ub.append(row)
        b_ub.append(unified_M_BIG)
    
    for r in range(s):
        row = np.zeros(num_vars)
        row[1+m+r] = -1
        row[-1] = -unified_M_BIG
        A_ub.append(row)
        b_ub.append(0)
    
    for r in range(s):
        row = np.zeros(num_vars)
        row[1+m+r] = 1
        row[-1] = unified_M_BIG
        A_ub.append(row)
        b_ub.append(unified_M_BIG)
    
    bounds = Bounds(lb=[-np.inf]*num_vars, ub=[np.inf]*num_vars)
    bounds.lb[0] = 0
    bounds.lb[1+m+s:1+m+s+(n-1)] = 0
    bounds.lb[-1] = 0
    bounds.ub[-1] = 1
    
    constraints = [LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub)),
                   LinearConstraint(np.array(A_eq), np.array(b_eq), np.array(b_eq))]
    
    integrality = np.zeros(num_vars)
    integrality[-1] = 1
    
    result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
    return result.fun if result.success else np.nan


################################# small unified binary MILP  ########################
      # my modification on 
      #A modified slacks-based measure of efficiency in data envelopment analysis
      #Kaoru Tone a , Mehdi Toloo b , ∗, Mohammad Izadikhah c
      #2021
#####################################################################################


def Unified_SBM_SuperSBM_algorithm(X, Y, epsilon):
    n = X.shape[1]
    scores = np.zeros(n)
    start_time = time.time()
    
    for p in range(n):
        scores[p] = unified_SBM_SuperSBM_scipy(X, Y, p)
    
    elapsed_time = time.time() - start_time
    n_efficient = np.sum(scores >= 1 - epsilon)
    return scores, n_efficient, elapsed_time

def solve_OneSupSBM_scipy(X, Y, k):
    m, n = X.shape
    s = Y.shape[0]
    
    # Variables: [alpha, t1, t2, lambda1(n), lambda2(n), s_minus(m), s_plus(s), 
    #             x_tilde(m), y_tilde(s), u(m), v(m), w]
    num_vars = 1 + 1 + 1 + n + n + m + s + m + s + m + m + 1
    
    c = np.zeros(num_vars)
    idx = 0
    alpha_idx = idx; idx += 1
    t1_idx = idx; idx += 1
    t2_idx = idx; idx += 1
    lambda1_idx = idx; idx += n
    lambda2_idx = idx; idx += n
    s_minus_idx = idx; idx += m
    s_plus_idx = idx; idx += s
    x_tilde_idx = idx; idx += m
    y_tilde_idx = idx; idx += s
    u_idx = idx; idx += m
    v_idx = idx; idx += m
    w_idx = idx
    
    # Objective
    c[u_idx:u_idx+m] = 1/(m * X[:, k])
    c[v_idx:v_idx+m] = 1/(m * X[:, k])
    c[w_idx] = -1
    c[t1_idx] = 1
    c[s_minus_idx:s_minus_idx+m] = -1/(m * X[:, k])
    
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    # sum(x_tilde[i]/X[i,k])/m - 1 <= alpha * M_BIG
    row = np.zeros(num_vars)
    row[alpha_idx] = -M_BIG
    row[x_tilde_idx:x_tilde_idx+m] = 1/(m * X[:, k])
    A_ub.append(row)
    b_ub.append(1)
    
    # 1 = t1 + sum(s_plus[r]/Y[r,k])/s
    row = np.zeros(num_vars)
    row[t1_idx] = 1
    row[s_plus_idx:s_plus_idx+s] = 1/(s * Y[:, k])
    A_eq.append(row)
    b_eq.append(1)
    
    # t1 * X[i,k] = sum(X[i,j]*lambda1[j]) + s_minus[i]
    for i in range(m):
        row = np.zeros(num_vars)
        row[t1_idx] = X[i, k]
        row[lambda1_idx:lambda1_idx+n] = -X[i, :]
        row[s_minus_idx+i] = -1
        A_eq.append(row)
        b_eq.append(0)
    
    # t1 * Y[r,k] = sum(Y[r,j]*lambda1[j]) - s_plus[r]
    for r in range(s):
        row = np.zeros(num_vars)
        row[t1_idx] = Y[r, k]
        row[lambda1_idx:lambda1_idx+n] = -Y[r, :]
        row[s_plus_idx+r] = 1
        A_eq.append(row)
        b_eq.append(0)
    
    # 1 = sum(y_tilde[r]/Y[r,k])/s
    row = np.zeros(num_vars)
    row[y_tilde_idx:y_tilde_idx+s] = 1/(s * Y[:, k])
    A_eq.append(row)
    b_eq.append(1)
    
    # x_tilde[i] >= sum(X[i,j]*lambda2[j] for j!=k)
    for i in range(m):
        row = np.zeros(num_vars)
        row[x_tilde_idx+i] = -1
        for j in range(n):
            if j != k:
                row[lambda2_idx+j] = X[i, j]
        A_ub.append(row)
        b_ub.append(0)
    
    # x_tilde[i] >= t2 * X[i,k]
    for i in range(m):
        row = np.zeros(num_vars)
        row[t2_idx] = X[i, k]
        row[x_tilde_idx+i] = -1
        A_ub.append(row)
        b_ub.append(0)
    
    # y_tilde[r] <= sum(Y[r,j]*lambda2[j] for j!=k)
    for r in range(s):
        row = np.zeros(num_vars)
        row[y_tilde_idx+r] = 1
        for j in range(n):
            if j != k:
                row[lambda2_idx+j] = -Y[r, j]
        A_ub.append(row)
        b_ub.append(0)
    
    # y_tilde[r] <= t2 * Y[r,k]
    for r in range(s):
        row = np.zeros(num_vars)
        row[t2_idx] = -Y[r, k]
        row[y_tilde_idx+r] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    # u[i] <= w * X[i,k]
    for i in range(m):
        row = np.zeros(num_vars)
        row[u_idx+i] = 1
        row[w_idx] = -X[i, k]
        A_ub.append(row)
        b_ub.append(0)
    
    # u[i] <= s_minus[i]
    for i in range(m):
        row = np.zeros(num_vars)
        row[u_idx+i] = 1
        row[s_minus_idx+i] = -1
        A_ub.append(row)
        b_ub.append(0)
    
    # u[i] >= s_minus[i] - (t1 - w) * X[i,k]
    for i in range(m):
        row = np.zeros(num_vars)
        row[u_idx+i] = -1
        row[s_minus_idx+i] = 1
        row[t1_idx] = -X[i, k]
        row[w_idx] = X[i, k]
        A_ub.append(row)
        b_ub.append(0)
    
    # v[i] >= X[i,k] * alpha
    for i in range(m):
        row = np.zeros(num_vars)
        row[v_idx+i] = -1
        row[alpha_idx] = X[i, k]
        A_ub.append(row)
        b_ub.append(0)
    
    # v[i] <= M_BIG * alpha
    for i in range(m):
        row = np.zeros(num_vars)
        row[v_idx+i] = 1
        row[alpha_idx] = -M_BIG
        A_ub.append(row)
        b_ub.append(0)
    
    # v[i] >= x_tilde[i] - (1 - alpha) * M_BIG
    for i in range(m):
        row = np.zeros(num_vars)
        row[v_idx+i] = -1
        row[x_tilde_idx+i] = 1
        row[alpha_idx] = M_BIG
        A_ub.append(row)
        b_ub.append(M_BIG)
    
    # v[i] <= x_tilde[i] - (1 - alpha) * X[i,k]
    for i in range(m):
        row = np.zeros(num_vars)
        row[v_idx+i] = 1
        row[x_tilde_idx+i] = -1
        row[alpha_idx] = -X[i, k]
        A_ub.append(row)
        b_ub.append(-X[i, k])
    
    # w >= M_SMALL * alpha
    row = np.zeros(num_vars)
    row[w_idx] = -1
    row[alpha_idx] = M_SMALL
    A_ub.append(row)
    b_ub.append(0)
    
    # w <= alpha
    row = np.zeros(num_vars)
    row[w_idx] = 1
    row[alpha_idx] = -1
    A_ub.append(row)
    b_ub.append(0)
    
    # w >= t1 - (1 - alpha)
    row = np.zeros(num_vars)
    row[w_idx] = -1
    row[t1_idx] = 1
    row[alpha_idx] = 1
    A_ub.append(row)
    b_ub.append(1)
    
    # w <= t1 - (1 - alpha) * M_SMALL
    row = np.zeros(num_vars)
    row[w_idx] = 1
    row[t1_idx] = -1
    row[alpha_idx] = -M_SMALL
    A_ub.append(row)
    b_ub.append(-M_SMALL)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    bounds = Bounds(lb=np.zeros(num_vars), ub=np.full(num_vars, np.inf))
    bounds.lb[t1_idx] = M_SMALL
    bounds.lb[t2_idx] = M_SMALL
    bounds.ub[alpha_idx] = 1
    
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub), LinearConstraint(A_eq, b_eq, b_eq)]
    
    integrality = np.zeros(num_vars)
    integrality[alpha_idx] = 1
    
    result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
    
    if result.success:
        return result.fun
    return np.nan
def OneSupSBM_algorithm(X, Y, epsilon):
    n = X.shape[1]
    scores = np.zeros(n)
    start_time = time.time()
    
    for k in range(n):
        scores[k] = solve_OneSupSBM_scipy(X, Y, k)
    
    elapsed_time = time.time() - start_time
    n_efficient = np.sum(scores >= 1 - epsilon)
    return scores, n_efficient, elapsed_time

################################# Integrated SBM-LP #################################

      #An integrated model for SBM and Super-SBM DEA models
      #Hsuan-Shih Lee 2021

#####################################################################################

def solve_IntegLP_scipy(X, Y, k):
    m, n = X.shape
    s = Y.shape[0]
    
    # Variables: [t1, t2, lambda1(n), lambda2(n), x1(m), y1(s), x2(m), y2(s)]
    num_vars = 2 + n + n + m + s + m + s
    
    c = np.zeros(num_vars)
    idx = 0
    t1_idx = idx; idx += 1
    t2_idx = idx; idx += 1
    lambda1_idx = idx; idx += n
    lambda2_idx = idx; idx += n
    x1_idx = idx; idx += m
    y1_idx = idx; idx += s
    x2_idx = idx; idx += m
    y2_idx = idx; idx += s
    
    # Objective: 1+(1/m)*sum(x1[i]/X[i,k]) + 1+(1/m)*sum(x2[i]/X[i,k]) - 1
    # Subtract 1 from the objective function


    c[x1_idx:x1_idx+m] = 1/(m * X[:, k])
    c[x2_idx:x2_idx+m] = 1/(m * X[:, k])
    
    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []
    
    # (1/s)*sum(y1[r]/Y[r,k]) = 1
    row = np.zeros(num_vars)
    row[y1_idx:y1_idx+s] = 1/(s * Y[:, k])
    A_eq.append(row)
    b_eq.append(1)
    
    # (1/s)*sum(y2[r]/Y[r,k]) = 1
    row = np.zeros(num_vars)
    row[y2_idx:y2_idx+s] = 1/(s * Y[:, k])
    A_eq.append(row)
    b_eq.append(1)
    
    # x1[i] >= sum(X[i,j]*lambda1[j])
    for i in range(m):
        row = np.zeros(num_vars)
        row[x1_idx+i] = -1
        row[lambda1_idx:lambda1_idx+n] = X[i, :]
        A_ub.append(row)
        b_ub.append(0)
    
    # y1[r] <= sum(Y[r,j]*lambda1[j])
    for r in range(s):
        row = np.zeros(num_vars)
        row[y1_idx+r] = 1
        row[lambda1_idx:lambda1_idx+n] = -Y[r, :]
        A_ub.append(row)
        b_ub.append(0)
    
    # x1[i] <= t1 * X[i,k]
    for i in range(m):
        row = np.zeros(num_vars)
        row[x1_idx+i] = 1
        row[t1_idx] = -X[i, k]
        A_ub.append(row)
        b_ub.append(0)
    
    # y1[r] >= t1 * Y[r,k]
    for r in range(s):
        row = np.zeros(num_vars)
        row[y1_idx+r] = -1
        row[t1_idx] = Y[r, k]
        A_ub.append(row)
        b_ub.append(0)
    
    # x2[i] >= sum(X[i,j]*lambda2[j] for j!=k)
    for i in range(m):
        row = np.zeros(num_vars)
        row[x2_idx+i] = -1
        for j in range(n):
            if j != k:
                row[lambda2_idx+j] = X[i, j]
        A_ub.append(row)
        b_ub.append(0)
    
    # y2[r] <= sum(Y[r,j]*lambda2[j] for j!=k)
    for r in range(s):
        row = np.zeros(num_vars)
        row[y2_idx+r] = 1
        for j in range(n):
            if j != k:
                row[lambda2_idx+j] = -Y[r, j]
        A_ub.append(row)
        b_ub.append(0)
    
    # x2[i] >= t2 * X[i,k]
    for i in range(m):
        row = np.zeros(num_vars)
        row[x2_idx+i] = -1
        row[t2_idx] = X[i, k]
        A_ub.append(row)
        b_ub.append(0)
    
    # y2[r] <= t2 * Y[r,k]
    for r in range(s):
        row = np.zeros(num_vars)
        row[y2_idx+r] = 1
        row[t2_idx] = -Y[r, k]
        A_ub.append(row)
        b_ub.append(0)
    
    bounds = [(0, None)] * num_vars
    result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                     bounds=bounds, method='highs')
    
    
    return result.fun -1 if result.success else np.nan

def IntegLP_algorithm(X, Y, epsilon):
    n = X.shape[1]
    scores = np.zeros(n)
    start_time = time.time()
    
    for k in range(n):
        scores[k] = solve_IntegLP_scipy(X, Y, k)
    
    elapsed_time = time.time() - start_time
    n_efficient = np.sum(scores >= 1 - epsilon)
    return scores, n_efficient, elapsed_time


#######################################

# Main

#######################################
Data = [1,2,3,4,5,6]

#Data = [1, 3, 5, 7,11]

data_num=6
NUM_INPUTS = 5
NUM_OUTPUTS = 8
NUM_DMUS = {1: 100, 2: 120, 3: 150, 4: 300, 5: 500, 6: 700}

METHODS = ['Enhanced-SupSBM', 'Simple-SupSBM', 'OneSupSBM', 'SBM-SSBM-LP']  # 'Unified' has been removed


all_times = {method: [] for method in METHODS}
all_results = {}

for d_num in range(1, data_num + 1):

    print(f'\n{"="*90}')
    print(f'Processing Data {d_num}'.center(90))
    print("="*90)
    
    np.random.seed(42)
    X = np.random.uniform(10, 100, size=(NUM_INPUTS, NUM_DMUS[d_num]))
    Y = np.random.uniform(10, 100, size=(NUM_OUTPUTS, NUM_DMUS[d_num]))
    
    # Store dimensions for this table
    all_results[d_num] = {
        'm': X.shape[0],
        'n': NUM_DMUS[d_num] if d_num >= 20 else 2,

        's': NUM_OUTPUTS if d_num >= 20 else 2
    }

    results = {}
    
    if 'Enhanced-SupSBM' in METHODS:
        eff_scores_enh, IE_enh, super_eff_enh, time_enh = SBM_SSBM_algorithm(X, Y, EPSILON, use_enhanced=True)
        results['Enhanced-SupSBM'] = {'scores': super_eff_enh, 'efficient': IE_enh, 'time': time_enh}
        all_times['Enhanced-SupSBM'].append(time_enh)
    
    if 'Simple-SupSBM' in METHODS:
        eff_scores_sim, IE_sim, super_eff_sim, time_sim = SBM_SSBM_algorithm(X, Y, EPSILON, use_enhanced=False)
        results['Simple-SupSBM'] = {'scores': super_eff_sim, 'efficient': IE_sim, 'time': time_sim}
        all_times['Simple-SupSBM'].append(time_sim)
    
    if 'Unified' in METHODS:
        scores_unified, n_eff_unified, time_unified = Unified_SBM_SuperSBM_algorithm(X, Y, EPSILON)
        results['Unified'] = {'scores': scores_unified, 'efficient': np.where(scores_unified >= 1 - EPSILON)[0].tolist(), 'time': time_unified}
        all_times['Unified'].append(time_unified)
    
    if 'OneSupSBM' in METHODS:
        scores_onesup, n_eff_onesup, time_onesup = OneSupSBM_algorithm(X, Y, EPSILON)
        results['OneSupSBM'] = {'scores': scores_onesup, 'efficient': np.where(scores_onesup >= 1 - EPSILON)[0].tolist(), 'time': time_onesup}
        all_times['OneSupSBM'].append(time_onesup)
    
    if 'SBM-SSBM-LP' in METHODS:
        scores_integlp, n_eff_integlp, time_integlp = IntegLP_algorithm(X, Y, EPSILON)
        results['SBM-SSBM-LP'] = {'scores': scores_integlp, 'efficient': np.where(scores_integlp >= 1 - EPSILON)[0].tolist(), 'time': time_integlp}
        all_times['SBM-SSBM-LP'].append(time_integlp)
    
    all_results[d_num].update(results)
    
    # Find fastest method for this table
    times = [(m, results[m]['time']) for m in METHODS if m in results]
    fastest = min(times, key=lambda x: x[1])
    
    # Display results table
    print('\nRESULTS SUMMARY')
    print('-'*90)
    
    table_data = []
    for method in METHODS:
        if method not in results:
            continue
        eff_count = np.sum(results[method]['scores'] >= 1 - EPSILON)
        max_diff = 0 if method == 'Simple-SupSBM' else np.max(np.abs(results[method]['scores'] - results['Simple-SupSBM']['scores']))
        
        table_data.append({
            'Method': method,
            'Time(s)': f"{results[method]['time']:.4f}",
            'Speedup': f"{fastest[1]/results[method]['time']:.2f}X" if method != fastest[0] else "1.00X",
            'Eff#': eff_count,
            'Max|Δ|': f"{max_diff:.6f}"
        })
    
    df_table = pd.DataFrame(table_data)
    print(df_table.to_string(index=False))
    
    # Detailed scores table
    if SHOW_DETAILS:
        print('\nDETAILED SCORES (First 10 DMUs)')
        print('-'*90)
        
        n_show = min(10, len(results['Simple-SupSBM']['scores']))
        scores_table = {'DMU': [f'DMU{i+1}' for i in range(n_show)]}
        for method in METHODS:
            if method in results:
                scores_table[method] = [f"{s:.4f}" for s in results[method]['scores'][:n_show]]
        
        df_scores = pd.DataFrame(scores_table)
        print(df_scores.to_string(index=False))
    
    # Save to Excel
    OUTPUT_FILE = f'dea_results_table{d_num}.xlsx'
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        pd.DataFrame(table_data).to_excel(writer, sheet_name='Summary', index=False)
        
        for method, data in results.items():
            df = pd.DataFrame({
                'DMU': [f'DMU{i+1}' for i in range(len(data['scores']))],
                'Score': data['scores'],
                'Efficient': data['scores'] >= 1 - EPSILON
            })
            df.to_excel(writer, sheet_name=method, index=False)
    
    print(f'\n✓ Results saved to {OUTPUT_FILE}')

# Final comparison table with dimensions
print('\n' + '='*90)
print('TIMING COMPARISON ACROSS ALL DATA'.center(90))
print('='*90)

time_comparison = {
    'Data #': Data,
    'm': [all_results[t]['m'] for t in Data],
    's': [all_results[t]['s'] for t in Data],
    'n': [all_results[t]['n'] for t in Data]
    
}
for method in METHODS:
    time_comparison[method] = [f"{t:.4f}" for t in all_times[method]]

df_time = pd.DataFrame(time_comparison)
print(df_time.to_string(index=False))
print('='*90)

# Save timing comparison
with pd.ExcelWriter('dea_timing_comparison.xlsx', engine='openpyxl') as writer:
    df_time.to_excel(writer, sheet_name='Timing', index=False)

print('\n✓ Timing comparison saved to dea_timing_comparison.xlsx')
