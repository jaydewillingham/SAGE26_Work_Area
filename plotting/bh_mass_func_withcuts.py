#!/usr/bin/env python
"""
Black Hole Mass Function (BHMF) plotter for SAGE semi-analytic model output.

Compares model predictions against observational data across multiple redshifts.
Produces a publication-ready figure with a redshift colorbar and clean formatting.
Also produces diagnostic plots of M_BH vs M_* to show the effect of data cuts.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter

warnings.filterwarnings("ignore")


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class SimConfig:
    """Simulation and file configuration."""
    dir_name: str = '../output/my_mini_millennium/'
    #dir_name: str = '../output/mini_millenn_AGNefficiency/'
    file_name: str = 'model_0.hdf5'
    hubble_h: float = 0.678
    box_size: float = 62.5        # h^-1 Mpc
    volume_fraction: float = 1.0
    first_snap: int = 0
    last_snap: int = 63
    redshifts: List[float] = field(default_factory=lambda: [
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
        14.086,  12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
          7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
          3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
          2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
          1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
          0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
          0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
    ])

    @property
    def filepath(self) -> str:
        return os.path.join(self.dir_name, self.file_name)

    @property
    def volume(self) -> float:
        """Comoving volume in Mpc^3."""
        return (self.box_size / self.hubble_h) ** 3.0 * self.volume_fraction

    @property
    def n_snaps(self) -> int:
        return self.last_snap - self.first_snap + 1


@dataclass
class PlotConfig:
    """Plotting configuration."""
    output_format: str = '.pdf'
    data_dir: str = '../data/'
    # Redshifts at which to compare model vs. observations
    bhmf_redshifts: List[float] = field(
        default_factory=lambda: [0.1, 1.0, 2.0, 4.0, 6.0, 8.0]
    )
    # Observational data files keyed by redshift
    obs_files: Dict[float, str] = field(default_factory=lambda: {
        0.1: 'fig4_bhmf_z0.1.txt',
        1.0: 'fig4_bhmf_z1.0.txt',
        2.0: 'fig4_bhmf_z2.0.txt',
        4.0: 'fig4_bhmf_z4.0.txt',
        6.0: 'fig4_bhmf_z6.0.txt',
        8.0: 'fig4_bhmf_z8.0.txt',
    })
    # BHMF histogram parameters
    mass_bin_min: float = 5.0
    mass_bin_max: float = 11.5
    # CHANGED: Increased bin width to smooth jagged model lines as suggested.
    # 0.25 is a good start; 0.5 is another option for even smoother results.
    mass_bin_width: float = 0.25
    # Axis limits for BHMF
    xlim: tuple = (5.5, 10.0)
    ylim: tuple = (1e-5, 1e-2)
    # Axis limits for MBH-M* plot
    mstar_lim: tuple = (7.0, 12.0)
    mbh_lim: tuple = (4.0, 10.5)
    # Colormap
    cmap_name: str = 'plasma'


# ==============================================================================
# I/O utilities
# ==============================================================================

def read_hdf(filepath: str, snap_num: str, param: str) -> np.ndarray:
    """Read a single parameter array from an HDF5 snapshot."""
    with h5py.File(filepath, 'r') as f:
        if snap_num not in f:
            raise KeyError(f"Snapshot '{snap_num}' not found in {filepath}.")
        if param not in f[snap_num]:
            raise KeyError(f"Parameter '{param}' not found in {snap_num}.")
        return np.array(f[snap_num][param])


def snap_key(snap: int) -> str:
    """Return the HDF5 group key for a given snapshot index."""
    return f'Snap_{snap}'


def load_obs_data(filepath: str) -> Optional[np.ndarray]:
    """Load observational BHMF data."""
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(f"Expected >= 4 columns, got shape {data.shape}.")
        return data
    except Exception as exc:
        print(f"  [WARNING] Could not load {filepath}: {exc}")
        return None


# ==============================================================================
# Data loading
# ==============================================================================

def load_all_snapshots(cfg: SimConfig) -> Dict[str, np.ndarray]:
    """Load all required galaxy properties for every snapshot."""
    h = cfg.hubble_h
    M_unit = 1.0e10 / h   # Internal mass unit -> solar masses
    props = {name: {} for name in ['StellarMass', 'BlackHoleMass']}
    print(f"Reading galaxy properties from: {cfg.filepath}\n")
    for snap in range(cfg.first_snap, cfg.last_snap + 1):
        key = snap_key(snap)
        def M(param): return read_hdf(cfg.filepath, key, param) * M_unit
        props['StellarMass'][snap]  = M('StellarMass')
        props['BlackHoleMass'][snap] = M('BlackHoleMass')
    return props


# ==============================================================================
# BHMF computation
# ==============================================================================

def compute_bhmf(
    bh_mass_solar: np.ndarray,
    stellar_mass_solar: np.ndarray,
    mass_bins: np.ndarray,
    volume: float,
) -> tuple:
    """Compute the black hole mass function with a mass ratio cut."""
    bin_width = mass_bins[1] - mass_bins[0]
    bin_centers = mass_bins[:-1] + 0.5 * bin_width
    valid_stellar_mask = stellar_mass_solar > 0.0
    ratio = np.zeros_like(bh_mass_solar, dtype=float)
    np.divide(
        bh_mass_solar, stellar_mass_solar,
        out=ratio, where=valid_stellar_mask
    )
    mask = (bh_mass_solar > 0.0) & (ratio > 0.01)
    if not np.any(mask):
        return bin_centers, np.zeros(len(bin_centers))
    log_mass = np.log10(bh_mass_solar[mask])
    counts, _ = np.histogram(log_mass, bins=mass_bins)
    phi = np.where(counts > 0, counts / (volume * bin_width), 0.0)
    return bin_centers, phi


def find_nearest_snapshot(target_z: float, redshifts: List[float]) -> tuple:
    """Return (snap_index, actual_redshift) closest to target_z."""
    idx = int(np.argmin(np.abs(np.array(redshifts) - target_z)))
    return idx, redshifts[idx]

# ==============================================================================
# Plotting
# ==============================================================================

def apply_publication_style() -> None:
    """Apply rcParams for a clean, journal-ready style."""
    plt.rcParams.update({
        'figure.facecolor': 'white', 'figure.dpi': 150,
        'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.linewidth': 1.0,
        'axes.labelcolor': 'black', 'axes.labelsize': 13,
        'xtick.color': 'black', 'ytick.color': 'black',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'font.family': 'serif', 'font.size': 13, 'mathtext.fontset': 'stix',
        'legend.facecolor': 'white', 'legend.edgecolor': '0.7',
        'legend.fontsize': 11, 'legend.framealpha': 0.9,
        'lines.linewidth': 1.8, 'text.color': 'black',
    })


def plot_bhmf(
    props: Dict[str, np.ndarray],
    sim_cfg: SimConfig,
    plot_cfg: PlotConfig,
    output_dir: str,
) -> None:
    """Plot the Black Hole Mass Function at multiple redshifts."""
    print("Plotting black hole mass function...")
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    cmap = plt.get_cmap(plot_cfg.cmap_name)
    z_vals = np.array(plot_cfg.bhmf_redshifts)
    norm = mcolors.Normalize(vmin=z_vals.min(), vmax=z_vals.max())
    mass_bins = np.arange(
        plot_cfg.mass_bin_min,
        plot_cfg.mass_bin_max + plot_cfg.mass_bin_width,
        plot_cfg.mass_bin_width,
    )
    redshift_legend_handles = []

    for target_z in plot_cfg.bhmf_redshifts:
        snap_idx, actual_z = find_nearest_snapshot(target_z, sim_cfg.redshifts)
        color = cmap(norm(target_z))
        bin_centers, phi = compute_bhmf(
            props['BlackHoleMass'][snap_idx],
            props['StellarMass'][snap_idx],
            mass_bins,
            sim_cfg.volume,
        )
        valid = phi > 0
        if np.any(valid):
            ax.plot(bin_centers[valid], phi[valid], color=color, lw=2.0, ls='-', zorder=3)
            label = f'z = {target_z:.1f}'
            redshift_legend_handles.append(Line2D([0], [0], color=color, lw=2.0, ls='-', label=label))
        obs_filename = plot_cfg.obs_files.get(target_z)
        if obs_filename:
            obs_path = os.path.join(plot_cfg.data_dir, obs_filename)
            obs_data = load_obs_data(obs_path)
            if obs_data is not None:
                obs_mass, obs_phi, obs_phi_16, obs_phi_84 = obs_data.T
                ax.plot(obs_mass, obs_phi, color=color, lw=1.5, ls='--', alpha=0.85, zorder=2)
                ax.fill_between(obs_mass, obs_phi_16, obs_phi_84, color=color, alpha=0.15, zorder=1)

    style_handles = [
        Line2D([0], [0], color='0.3', lw=2.0, ls='-', label='SAGE (M$_{BH}$/M$_* > 0.01$)'),
        Line2D([0], [0], color='0.3', lw=1.5, ls='--', label=r'Zhang+23'),
    ]
    separator_handle = Line2D([0], [0], color='none', label='')
    all_handles = style_handles + [separator_handle] + redshift_legend_handles
    ax.legend(handles=all_handles, loc='upper right', fontsize=10, framealpha=0.9, edgecolor='0.7')
    ax.set_yscale('log')
    ax.set_xlim(*plot_cfg.xlim)
    ax.set_ylim(*plot_cfg.ylim)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{BH}\ /\ \mathrm{M}_\odot)$')
    ax.set_ylabel(r'$\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.45, color='grey', zorder=0)
    fig.tight_layout()
    output_path = os.path.join(output_dir, f'BlackHoleMassFunctionCUTS{plot_cfg.output_format}')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_path}\n")
    plt.close(fig)


# ==============================================================================
# NEW PLOTTING FUNCTION
# ==============================================================================

def plot_mbh_mstar_relation(
    props: Dict[str, np.ndarray],
    sim_cfg: SimConfig,
    plot_cfg: PlotConfig,
    output_dir: str,
) -> None:
    """
    Plot M_BH vs M_* to visualize the effect of the mass ratio cut.
    A separate plot is generated for each specified redshift.
    """
    print("Plotting M_BH vs M_* relation to show data cuts...")
    apply_publication_style()

    for target_z in plot_cfg.bhmf_redshifts:
        snap_idx, actual_z = find_nearest_snapshot(target_z, sim_cfg.redshifts)

        # --- Get data for this snapshot ---
        mbh_solar = props['BlackHoleMass'][snap_idx]
        mstar_solar = props['StellarMass'][snap_idx]

        # --- Filter for galaxies with both stellar and BH mass for plotting ---
        plot_mask = (mbh_solar > 0) & (mstar_solar > 0)
        if not np.any(plot_mask):
            print(f"  Skipping z={target_z:.1f}, no valid galaxies to plot.")
            continue

        mbh_plot = mbh_solar[plot_mask]
        mstar_plot = mstar_solar[plot_mask]

        # --- Determine which points are accepted vs rejected by the ratio cut ---
        ratio = mbh_plot / mstar_plot
        accepted_mask = ratio > 0.01
        rejected_mask = ~accepted_mask

        # --- Setup figure ---
        fig, ax = plt.subplots(figsize=(6, 6))

        # --- Plot rejected data (background) ---
        ax.scatter(
            np.log10(mstar_plot[rejected_mask]),
            np.log10(mbh_plot[rejected_mask]),
            s=5, c='0.7', alpha=0.3, zorder=1,
            label=r'Rejected (M$_\mathrm{BH}$/M$_* \leq 0.01$)'
        )

        # --- Plot accepted data (foreground) ---
        ax.scatter(
            np.log10(mstar_plot[accepted_mask]),
            np.log10(mbh_plot[accepted_mask]),
            s=15, c='crimson', alpha=0.7, zorder=2,
            label=r'Accepted (M$_\mathrm{BH}$/M$_* > 0.01$)'
        )

        # --- Plot the M_BH = 0.01 * M_* cut line ---
        mstar_line = np.logspace(plot_cfg.mstar_lim[0], plot_cfg.mstar_lim[1], 100)
        mbh_line = 0.01 * mstar_line
        ax.plot(
            np.log10(mstar_line), np.log10(mbh_line),
            'k--', lw=1.5, zorder=3, label=r'M$_\mathrm{BH} = 0.01 \times$ M$_*$'
        )

        # --- Formatting ---
        ax.set_title(f"SAGE Data at z = {actual_z:.2f}")
        ax.set_xlabel(r'$\log_{10}(M_*\ /\ \mathrm{M}_\odot)$')
        ax.set_ylabel(r'$\log_{10}(M_\mathrm{BH}\ /\ \mathrm{M}_\odot)$')
        ax.set_xlim(*plot_cfg.mstar_lim)
        ax.set_ylim(*plot_cfg.mbh_lim)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.45, color='grey', zorder=0)

        # --- Save ---
        fig.tight_layout()
        output_filename = f'MBH_vs_Mstar_z{target_z:.1f}{plot_cfg.output_format}'
        output_path = os.path.join(output_dir, output_filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved diagnostic plot to: {output_path}")
        plt.close(fig)
    print("") # Newline for cleaner output

# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    print("=" * 60)
    print(" Black Hole Mass Function — SAGE model")
    print("=" * 60 + "\n")

    sim_cfg  = SimConfig()
    plot_cfg = PlotConfig()

    output_dir = os.path.join(sim_cfg.dir_name, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Note: Reduced the number of loaded properties to only what's needed
        props = load_all_snapshots(sim_cfg)
    except (FileNotFoundError, KeyError) as e:
        print(f"\n[ERROR] Failed to load simulation data: {e}")
        print("Please check that 'SimConfig.dir_name' and 'SimConfig.file_name' are correct.")
        return

    # Plot the main BHMF figure
    plot_bhmf(props, sim_cfg, plot_cfg, output_dir)

    # Plot the new diagnostic M_BH vs M_* figure
    plot_mbh_mstar_relation(props, sim_cfg, plot_cfg, output_dir)

    print("Done.")


if __name__ == '__main__':
    main()