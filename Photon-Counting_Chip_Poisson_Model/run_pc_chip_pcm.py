import yaml
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from poisson_signal_generator_utils import (
    generate_signal,
    build_energy_bins_and_pdf,
    compute_filtered_spectrum,
    make_threshold_grid_from_LSB,
    spectrum_from_signal,
    plot_signals_over_time,
    plot_vint_prefilter,
)

# -------------------------
#  1） 参数配置
# -------------------------

def load_yaml_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found:\n{path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(results_root: str, exp_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(results_root) / exp_name / ts
    out.mkdir(parents=True, exist_ok=True)
    return out

def save_run_artifacts(out: Path, cfg_dict: dict, summary: dict):
    # 1) 配置快照
    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, allow_unicode=True, sort_keys=False)

    # 2) 摘要信息
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 3)  
    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

def postprocess(differential_avg, energy_axis, smooth_sigma: float):
    secd = -np.gradient(differential_avg, energy_axis)
    secd_smooth = gaussian_filter1d(secd, sigma=smooth_sigma)
    return secd, secd_smooth


# -------------------------
# 2) Utilities
# -------------------------
 
def load_spectrum_dict(spec_cfg: dict, base_dir: Path):
    spek_path = base_dir / spec_cfg["spek_path"]
    xcom_path = base_dir / spec_cfg["xcom_filter_path"]
 
    if not spek_path.exists():
        raise FileNotFoundError(f"spek_path not found: {spek_path}")
    if not xcom_path.exists():
        raise FileNotFoundError(f"xcom_filter_path not found: {xcom_path}")

    energies, Phi_in, Phi_out_list, fig_spec = compute_filtered_spectrum(
        spek_path=str(spek_path),
        xcom_filter_path=str(xcom_path),
        density=spec_cfg["density"],
        thickness_mm_list=spec_cfg["thickness_mm_list"],
        plot=spec_cfg.get("plot_filtered_spectrum", False),
    )

    values = Phi_out_list[spec_cfg["thickness_index"]]
    bins, pdf, dE = build_energy_bins_and_pdf(energies, values)
    return energies, Phi_in, values, bins, pdf, dE, fig_spec


def compute_photon_rate_dict(values, dE, src_cfg: dict):
    pixel_length_cm = src_cfg["pixel_length_m"] * 100.0
    A_cm2 = pixel_length_cm ** 2
    lam = np.sum(values * dE * src_cfg["current_mA"] * (1.0 / src_cfg["distance_m"]) ** 2 * A_cm2)
    return lam


def build_threshold_grid_dict(chip_cfg: dict, calib_cfg: dict):
    g_mat = 1000.0 / chip_cfg["W_eV"]
    thr_grid, lsb_values, E_thr_keV = make_threshold_grid_from_LSB(
        a_keV_per_LSB=calib_cfg["a_keV_per_LSB"],
        b_keV=calib_cfg["b_keV"],
        g_mat=g_mat,
        Cf=chip_cfg["Cf_F"],
        n_thr=calib_cfg.get("n_thr", 256),
    )
    return thr_grid, lsb_values, E_thr_keV, g_mat


def simulate_runs_dict(bins, pdf, photon_rate, thr_grid, lsb_values, chip_cfg: dict, run_cfg: dict, g_mat: float):
    np.random.seed(run_cfg.get("seed", 0))
    n_runs = int(run_cfg.get("n_runs", 1))

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
        )
        if irun == 0:
            fig_vint = plot_vint_prefilter(t, Vint, t0=5e-6, t1=6.3e-6)
            debug_figs["Vint_prefilter"] = fig_vint
        integ, diff = spectrum_from_signal(sig, thr_grid, lsb_values=lsb_values)
        integrals.append(integ)
        differentials.append(diff)
        signals.append(sig)
        t_ref = t

    integral_avg = np.mean(integrals, axis=0)
    differential_avg = np.mean(differentials, axis=0)
    return t_ref, signals, integral_avg, differential_avg, debug_figs


def save_or_show_fig(fig, save_path=None, show=True, dpi=150):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_and_save_dict(out: Path, cfg: dict, energy_axis, integral_avg, differential_avg, secd, secd_smooth, t, signals):
    # 信号叠加图（如果 utils 内部直接 show，那就只负责 show，不强制保存）
    if cfg["run"].get("plot_signals", True):
        plot_signals_over_time(
            t, signals,
            max_plot=cfg["run"].get("max_plot_signals", 5),
            zoom=tuple(cfg["run"].get("zoom", (5.0, 5.2)))
        )

    # 三联图保存
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    ax1.plot(energy_axis, integral_avg, lw=1.8)
    ax1.set_ylabel("Integral counts")
    ax1.set_title("Integral Spectrum")

    ax2.plot(energy_axis, differential_avg, lw=1.8, label="Differential (-dN/dT)")
    ax2.set_ylabel("Counts per keV")
    ax2.set_title("Differential Spectrum")
    ax2.set_ylim(-100, 100)
    ax2.legend()

    ax3.plot(energy_axis, secd, lw=1.0, label="raw")
    ax3.plot(energy_axis, secd_smooth, lw=1.8, label=f"Gaussian σ={cfg['run']['smooth_sigma']}")
    ax3.set_ylabel("2nd-derivative proxy")
    ax3.set_title("Second Derivative (smoothed)")
    ax3.set_ylim(-100, 30)
    ax3.legend()

    for ax in (ax1, ax2, ax3):
        ax.set_xticks(np.arange(0, 300, 20))
    ax3.set_xlabel("Threshold / Energy (keV)")

    fig.tight_layout()
    fig.savefig(out / "spectra.png", dpi=150, bbox_inches="tight")

    if cfg["run"].get("show_plots", True):
        plt.show()
    else:
        plt.close(fig)



# -------------------------
# 3) MAIN ENTRY
# -------------------------
def main():
    BASE_DIR = Path(__file__).resolve().parent

    cfg_path = BASE_DIR / "configs" / "gd_demo.yaml"
    cfg = load_yaml_config(cfg_path)  # cfg 是 dict
    print("chip.delta_t_s =", cfg["chip"]["delta_t_s"], type(cfg["chip"]["delta_t_s"]))
    print("chip.pulse_s   =", cfg["chip"]["pulse_s"], type(cfg["chip"]["pulse_s"]))
    print("chip.integral_time_s =", cfg["chip"]["integral_time_s"], type(cfg["chip"]["integral_time_s"]))
    print("Loaded yaml:", cfg_path)

    # 1) 创建本次运行输出目录（唯一的 out）
    results_root = BASE_DIR / cfg["run"]["results_root"]
    out = make_run_dir(
        results_root,
        cfg["run"]["exp_name"]
    )

    # 2) 保存配置快照（强烈建议）
    save_run_artifacts(
        out=out,
        cfg_dict=cfg,
        summary={}
    )

    # (A) 光谱与滤片
    energies, Phi_in, values, bins, pdf, dE, fig_spec = load_spectrum_dict(cfg["spectrum"], BASE_DIR)

    # (B) photon_rate（保持你原始逻辑：用选定的 values）
    photon_rate = compute_photon_rate_dict(values, dE, cfg["source"])
    lambda_tau = photon_rate * cfg["chip"]["pulse_s"]
    print(f"photon_rate (lambda) = {photon_rate:.6g} [1/s]")
    print(f"lambda*tau = {lambda_tau:.6g} (dimensionless)")

    # (C) 阈值轴
    thr_grid, lsb_values, E_thr_keV, g_mat = build_threshold_grid_dict(cfg["chip"], cfg["calib"])

    # (D) 仿真
    t, signals, integral_avg, differential_avg, debug_figs = simulate_runs_dict(
        bins, pdf, photon_rate,
        thr_grid, lsb_values,
        cfg["chip"], cfg["run"], g_mat
    )

    # (E) 后处理
    energy_axis = lsb_values
    secd, secd_smooth = postprocess(differential_avg, energy_axis, cfg["run"]["smooth_sigma"])

    fig_vint = debug_figs.get("Vint_prefilter")
    if fig_vint is not None:
        save_or_show_fig(
            fig_vint,
            save_path=out / "Vint_prefilter.png",
            show=cfg["run"]["show_plots"]
        )

    # (F) 作图 & 保存（全部保存到 out）
    if fig_spec is not None:
        save_or_show_fig(
            fig_spec,
            save_path=out / "filtered_spectrum.png",
            show=cfg["run"]["show_plots"]
        )

    plot_signals_over_time(
    t,
    signals,
    max_plot=cfg["run"]["max_plot_signals"],
    zoom=tuple(cfg["run"]["zoom"]),
    save_path=out / "signals_zoom.png",
    show=cfg["run"]["show_plots"]
    )

    plot_signals_over_time(
    t,
    signals,
    max_plot=cfg["run"]["max_plot_signals"],
    save_path=out / "signals_full.png",
    show=cfg["run"]["show_plots"]
    )
 
    plot_and_save_dict(out, cfg, energy_axis, integral_avg, differential_avg, secd, secd_smooth, t, signals)

    # (G) 保存关键数组（方便团队二次分析）
    np.save(out / "integral_avg.npy", integral_avg)
    np.save(out / "differential_avg.npy", differential_avg)
    np.save(out / "energy_axis_keV.npy", energy_axis)

    # 8) 写 summary（覆盖掉之前空的 summary）
    summary = {
        "photon_rate_1ps": float(photon_rate),
        "lambda_tau": float(lambda_tau),
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