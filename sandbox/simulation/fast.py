"""Backend accelerato con Numba (opzionale).

Gli stessi schemi numerici delle classi in ``sandbox.models`` (che restano
l'implementazione di riferimento, leggibile e testata), riscritti come cicli
compilati JIT: ~20–150× più veloci, abbastanza per ricalcolare reti con
Hodgkin-Huxley a ogni movimento di uno slider. ``tests/test_backends.py``
verifica che i due backend producano gli stessi tracciati.

Lo stato della rete (potenziali, soglie, conduttanze, plasticità, linea di
ritardo e suo puntatore) arriva dall'esterno ed è aggiornato sul posto:
``engine.Stepper`` lo conserva fra una chiamata e l'altra per simulare a
blocchi (anello chiuso).

Convenzioni degli argomenti opzionali (per evitare array giganti inutili):
* ``g_ext``, ``ge_ext``, ``release_gain`` hanno forma (passi, n) oppure (1, n)
  se costanti (zeri per le conduttanze, uni per il guadagno);
* ``gap_on`` / ``stp_on`` saltano del tutto il lavoro quando la funzione è spenta.

Se Numba non è installato, ``AVAILABLE`` è False e il motore usa NumPy.
"""

import math

import numpy as np

try:
    from numba import njit
    AVAILABLE = True
except ImportError:  # pragma: no cover - dipende dall'ambiente
    AVAILABLE = False

    def njit(*args, **kwargs):
        return (lambda f: f) if not args or not callable(args[0]) else args[0]


@njit(cache=True)
def _release_and_deliver(raster_row, gain_row, buffer, ptr, g_e, g_i, decay_e, decay_i, w_exc, w_inh,
                         stp_on, stp_mask, stp_u, x, u, rec_decay, fac_decay):
    """Plasticità a breve termine, guadagno di rilascio, linea di ritardo, decadimento e consegna.

    Replica ShortTermPlasticity.efficacy + ExponentialSynapses.update. Restituisce il nuovo puntatore.
    """
    n = raster_row.size
    size = buffer.shape[0]
    for j in range(n):
        if stp_on:
            x[j] = 1.0 - (1.0 - x[j]) * rec_decay
            u[j] = stp_u + (u[j] - stp_u) * fac_decay
        eff = 0.0
        if raster_row[j]:
            eff = 1.0
            if stp_on and stp_mask[j]:
                release = u[j] * x[j]
                eff = release / stp_u
                x[j] -= release
                u[j] += stp_u * (1.0 - u[j])
            eff *= gain_row[j]
        buffer[ptr, j] = eff
    ptr = (ptr + 1) % size
    for j in range(n):
        g_e[j] *= decay_e
        g_i[j] *= decay_i
    for pre in range(n):
        a = buffer[ptr, pre]
        if a != 0.0:
            for post in range(n):
                g_e[post] += a * w_exc[pre, post]
                g_i[post] += a * w_inh[pre, post]
    return ptr


@njit(cache=True)
def _gap_rows(gap, gap_on):
    """Somma per riga e forma sparsa (CSR) delle gap junction: nei circuiti veri pochi neuroni ne hanno, e il
    termine Σ g·V a ogni passo costerebbe n² con la matrice piena. L'ordine delle somme non cambia."""
    n = gap.shape[0]
    rowsum = np.zeros(n)
    start = np.zeros(n + 1, dtype=np.int64)
    count = 0
    if gap_on:
        for i in range(n):
            for j in range(n):
                rowsum[i] += gap[i, j]
                if gap[i, j] != 0.0:
                    count += 1
    cols = np.empty(count, dtype=np.int64)
    vals = np.empty(count)
    k = 0
    for i in range(n):
        start[i] = k
        if gap_on:
            for j in range(n):
                if gap[i, j] != 0.0:
                    cols[k] = j
                    vals[k] = gap[i, j]
                    k += 1
    start[n] = k
    return rowsum, start, cols, vals


@njit(cache=True)
def _extra_terms(i, k_ext, g_ext, ge_ext, gap_on, gap_start, gap_cols, gap_vals, gap_rowsum, v_prev):
    """Σg e Σg·E delle conduttanze extra per il neurone i (stimoli + gap junction)."""
    g_x = g_ext[k_ext, i]
    num_x = ge_ext[k_ext, i]
    if gap_on:
        g_x += gap_rowsum[i]
        s = 0.0
        for k in range(gap_start[i], gap_start[i + 1]):
            s += gap_vals[k] * v_prev[gap_cols[k]]
        num_x += s
    return g_x, num_x


@njit(cache=True)
def lif_kernel(w_exc, w_inh, gap, gap_on, i_ext, g_ext, ge_ext, release_gain,
               stp_on, stp_mask, stp_u, rec_decay, fac_decay,
               dt, v_rest, v_thresh, v_reset, tau_m, r_m, t_ref, theta_jump, tau_theta, v_peak, spike_width,
               decay_e, decay_i, e_exc, e_inh, v, theta, refractory_left, g_e, g_i, x, u, buffer, ptr_ref):
    steps, n = i_ext.shape
    g_l = 1.0 / r_m
    c_m = tau_m / r_m
    i_leak = g_l * v_rest
    dt_over_c = dt / c_m
    theta_decay = math.exp(-dt / tau_theta) if tau_theta > 0 else 0.0
    ext_varies = g_ext.shape[0] == steps
    gain_varies = release_gain.shape[0] == steps

    v_prev = np.empty(n)
    gap_rowsum, gap_start, gap_cols, gap_vals = _gap_rows(gap, gap_on)
    ptr = ptr_ref[0]
    v_trace = np.empty((steps, n))
    g_e_trace = np.empty((steps, n), dtype=np.float32)
    g_i_trace = np.empty((steps, n), dtype=np.float32)
    raster = np.zeros((steps, n), dtype=np.bool_)

    for k in range(steps):
        k_ext = k if ext_varies else 0
        if gap_on:
            v_prev[:] = v
        any_spike = False
        for i in range(n):
            g_e_trace[k, i] = g_e[i]
            g_i_trace[k, i] = g_i[i]
            g_x, num_x = _extra_terms(i, k_ext, g_ext, ge_ext, gap_on, gap_start, gap_cols, gap_vals, gap_rowsum,
                                      v_prev)
            g_tot = g_l + g_e[i] + g_i[i] + g_x
            v_inf = (i_leak + g_e[i] * e_exc + g_i[i] * e_inh + num_x + i_ext[k, i]) / g_tot
            vi = v_inf + (v[i] - v_inf) * math.exp(-dt_over_c * g_tot)
            refractory = refractory_left[i] > 0
            if refractory:
                vi = v_reset
            refractory_left[i] -= dt
            theta[i] *= theta_decay
            if not refractory and vi >= v_thresh + theta[i]:
                raster[k, i] = True
                any_spike = True
                v_trace[k, i] = v_peak
                vi = v_reset
                theta[i] += theta_jump
                refractory_left[i] = t_ref
            else:
                v_trace[k, i] = vi
            v[i] = vi
        if gap_on and any_spike:  # spikelet trasmessi dalle gap junction (vedi AdaptiveLIF.spikelets)
            for i in range(n):
                if raster[k, i]:
                    continue
                coupling = 0.0
                for j in range(n):
                    if raster[k, j]:
                        coupling += gap[i, j]
                v[i] += coupling * (v_peak - v[i]) * spike_width / c_m
        ptr = _release_and_deliver(raster[k], release_gain[k if gain_varies else 0], buffer, ptr, g_e, g_i,
                                   decay_e, decay_i, w_exc, w_inh, stp_on, stp_mask, stp_u, x, u,
                                   rec_decay, fac_decay)
    ptr_ref[0] = ptr
    return v_trace, raster, g_e_trace, g_i_trace


@njit(cache=True)
def hh_kernel(w_exc, w_inh, gap, gap_on, i_ext, g_ext, ge_ext, release_gain,
              stp_on, stp_mask, stp_u, rec_decay, fac_decay,
              dt, x_inf, x_decay, v_min, v_max, inv_res,
              g_na_max, g_k_max, g_l, e_na, e_k, e_l, c_m, spike_detect,
              decay_e, decay_i, e_exc, e_inh, v, gates, g_e, g_i, x, u, buffer, ptr_ref):
    steps, n = i_ext.shape
    dt_over_c = dt / c_m
    i_leak = g_l * e_l
    ext_varies = g_ext.shape[0] == steps
    gain_varies = release_gain.shape[0] == steps

    v_prev = np.empty(n)
    gap_rowsum, gap_start, gap_cols, gap_vals = _gap_rows(gap, gap_on)
    ptr = ptr_ref[0]
    v_trace = np.empty((steps, n))
    g_e_trace = np.empty((steps, n), dtype=np.float32)
    g_i_trace = np.empty((steps, n), dtype=np.float32)
    raster = np.zeros((steps, n), dtype=np.bool_)

    for k in range(steps):
        k_ext = k if ext_varies else 0
        if gap_on:
            v_prev[:] = v
        for i in range(n):
            g_e_trace[k, i] = g_e[i]
            g_i_trace[k, i] = g_i[i]
            idx = int((min(max(v[i], v_min), v_max) - v_min) * inv_res + 0.5)
            for gidx in range(3):
                xi = x_inf[gidx, idx]
                gates[gidx, i] = xi + (gates[gidx, i] - xi) * x_decay[gidx, idx]
            m, h, nn = gates[0, i], gates[1, i], gates[2, i]
            g_na = g_na_max * m ** 3 * h
            g_k = g_k_max * nn ** 4
            g_x, num_x = _extra_terms(i, k_ext, g_ext, ge_ext, gap_on, gap_start, gap_cols, gap_vals, gap_rowsum,
                                      v_prev)
            g_tot = g_na + g_k + g_l + g_e[i] + g_i[i] + g_x
            v_inf = (g_na * e_na + g_k * e_k + i_leak
                     + g_e[i] * e_exc + g_i[i] * e_inh + num_x + i_ext[k, i]) / g_tot
            v_new = v_inf + (v[i] - v_inf) * math.exp(-dt_over_c * g_tot)
            if v[i] < spike_detect <= v_new:
                raster[k, i] = True
            v[i] = v_new
            v_trace[k, i] = v_new
        ptr = _release_and_deliver(raster[k], release_gain[k if gain_varies else 0], buffer, ptr, g_e, g_i,
                                   decay_e, decay_i, w_exc, w_inh, stp_on, stp_mask, stp_u, x, u,
                                   rec_decay, fac_decay)
    ptr_ref[0] = ptr
    return v_trace, raster, g_e_trace, g_i_trace
