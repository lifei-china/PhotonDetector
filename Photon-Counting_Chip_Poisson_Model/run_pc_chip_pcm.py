import yaml
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from pc_chip_pcm_utils import (
    load_yaml_config,   make_run_dir,   save_run_artifacts,
    init_frontend_shaper_params,
    load_spectrum_dict, compute_photon_rate_dict,
    generate_signal, 
    build_threshold_grid_dict,
    spectrum_from_signal,
    postprocess,
    plot_vint_prefilter,
    plot_and_save_dict,
    save_all_figures_and_plots
)



# -------------------------
#  主模型
# -------------------------

def simulate_runs_dict(bins, pdf, photon_rate, thr_grid, lsb_values,
                       chip_cfg: dict, run_cfg: dict,
                       tau_rc = 1):
    np.random.seed(run_cfg.get("seed", 0))
    n_runs = int(run_cfg.get("n_runs", 1))
    g_mat = 1000.0 / chip_cfg["W_eV"]

    T_total = float(chip_cfg["integral_time_s"])
    pulse   = float(chip_cfg["pulse_s"])
    dt      = float(chip_cfg["delta_t_s"])
    noise_ENC = float(chip_cfg["noise_ENC"])
    Cf = float(chip_cfg["Cf_F"])

    integrals, differentials, signals = [], [], []
    t_ref = None
    debug_figs = {}   # 用来返回图像

    for irun in range(n_runs):
        t, sig, times, Vint = generate_signal(
            bins, pdf, photon_rate,
            T_total,
            pulse,
            dt,
            g_mat=g_mat,
            noise_ENC=noise_ENC,
            Cf=Cf,
            tau_rc = tau_rc, 
        )
        if irun == 0:
            fig_vint = plot_vint_prefilter(t, Vint, t0=5e-6, t1=6.3e-6)
            debug_figs["Vint_prefilter"] = fig_vint
        thr_grid_local = np.linspace(0.0, np.percentile(sig, 99.9), 256)

        print("[THR] thr_local min/max min/max =", thr_grid_local.min(), thr_grid_local.max())
        print("[THR] sig min/max      =", sig.min(), sig.max())
        print("[THR] in-range count   =", np.sum((thr_grid >= sig.min()) & (thr_grid <= sig.max())))


        integ, diff = spectrum_from_signal(sig, thr_grid, x_axis=thr_grid)

        # print("thr_local min/max =", thr_grid_local.min(), thr_grid_local.max())
        print("thr min/max =", thr_grid.min(), thr_grid.max())
        print("sig p99/max =", np.percentile(sig, 99.9), sig.max())
        print("integ[0], integ[-1] =", integ[0], integ[-1])


        integrals.append(integ)
        differentials.append(diff)
        signals.append(sig)
        t_ref = t

    integral_avg = np.mean(integrals, axis=0)
    differential_avg = np.mean(differentials, axis=0)
    return t_ref, signals, integral_avg, differential_avg, debug_figs, lsb_values





# -------------------------
#  主程序
# -------------------------
def main():
    BASE_DIR = Path(__file__).resolve().parent

    cfg_path = BASE_DIR / "configs" / "gd_demo.yaml"
    cfg = load_yaml_config(cfg_path)  # 读取参数文件


    # 1) 创建本次运行输出目录
    results_root = BASE_DIR / cfg["run"]["results_root"]
    out = make_run_dir(results_root, cfg["run"]["exp_name"])

    # 2) 保存配置快照
    save_run_artifacts(out=out, cfg_dict=cfg, summary={})


    # (A) 光谱与滤片
    (energies, Phi_in, values, 
     photon_rate, bins, pdf, dE, 
     fig_spec) = load_spectrum_dict(cfg["spectrum"], cfg["source"], BASE_DIR)
    lambda_tau = photon_rate * cfg["chip"]["pulse_s"] 

 
    # (B) 初始化前端整形器参数 
    frontend = init_frontend_shaper_params(
        dt = float(cfg["chip"]["delta_t_s"]),
        pulse_width = float(cfg["chip"]["pulse_s"]),
        tau_decay=float(cfg["chip"]["tau_decay_s"]), 
    )
    tau_rc = frontend["tau_rc"]
    k_shaper = frontend["k_shaper"]
 


    # (C) 阈值轴 
    (thr_grid, lsb_values, 
     E_thr_keV) = build_threshold_grid_dict(cfg["chip"], 
                                            cfg["calib"], k_shaper)



    # (D) 仿真
    (t, signals, integral_avg, 
     differential_avg, debug_figs, lsb_values) = simulate_runs_dict(
        bins, pdf, photon_rate, thr_grid, E_thr_keV,
        cfg["chip"], cfg["run"],    tau_rc
    )

 

    # (E) 后处理
    energy_axis = lsb_values
    secd, secd_smooth = postprocess(differential_avg, energy_axis, cfg["run"]["smooth_sigma"])
    
    # 存图
    save_all_figures_and_plots(out, cfg, t=t, signals=signals,
        debug_figs=debug_figs,  fig_spec=fig_spec,)
    plot_and_save_dict(out, cfg, energy_axis, integral_avg, differential_avg, secd, secd_smooth, t, signals)

    # (G) 保存关键数组 
    np.save(out / "integral_avg.npy", integral_avg)
    np.save(out / "differential_avg.npy", differential_avg)
    np.save(out / "energy_axis_keV.npy", energy_axis)

    # 8) 写 summary
    summary = {
        "photon_rate_1ps": float(photon_rate),
        "lambda_tau": float(lambda_tau),
        "tau_rc_s": float(tau_rc),
        "k_shaper_V/V": float(k_shaper),
        "n_runs": int(cfg["run"]["n_runs"]),
        "seed": int(cfg["run"]["seed"]),
        "pulse_s": float(cfg["chip"]["pulse_s"]),
        "delta_t_s": float(cfg["chip"]["delta_t_s"]),
        "integral_time_s": float(cfg["chip"]["integral_time_s"]),
        "Cf_F": float(cfg["chip"]["Cf_F"]),
        "noise_ENC": float(cfg["chip"]["noise_ENC"]), 
        "thickness_mm": float(cfg["spectrum"]["thickness_mm_list"][cfg["spectrum"]["thickness_index"]]),
    }

    save_run_artifacts(out=out, cfg_dict=cfg, summary=summary)

    print(f"[OK] Results saved to: {out.resolve()}")


if __name__ == "__main__":
    main()