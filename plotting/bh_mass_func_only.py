#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os

from random import sample, seed

import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

# File details
DirName = '../output/my_mini_millennium/'
FileName = 'model_0.hdf5'

# Simulation details
Hubble_h = 0.678        # Hubble parameter
BoxSize = 62.5         # h-1 Mpc
VolumeFraction = 1.0   # Fraction of the full volume output by the model
FirstSnap = 0          # First snapshot to read
LastSnap = 63          # Last snapshot to read
redshifts = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 10.944, 10.073, 
             9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 
             2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 0.989, 0.905, 
             0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 
             0.116, 0.089, 0.064, 0.041, 0.020, 0.000]  # Redshift of each snapshot

#gadget_as=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#redshifts=1.0+1.0/np.array(gadget_as)

# Plotting options
#whichimf = 1        # 0=Slapeter; 1=Chabrier
#dilute = 7500       # Number of galaxies to plot in scatter plots
sSFRcut = -11.0     # Divide quiescent from star forming galaxies
SMFsnaps = [63, 37, 32, 27, 23, 20, 18, 16]  # Snapshots to plot the SMF

OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (8.34,6.25)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'


# ==================================================================

def read_hdf(filename = None, snap_num = None, param = None):

    property = h5.File(DirName+FileName,'r')
    return np.array(property[snap_num][param])


# ==================================================================

if __name__ == '__main__':

    print('Running Black Hole Mass Funtion Only\n')

    seed(2222)
    volume = (BoxSize/Hubble_h)**3.0 * VolumeFraction

    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir): os.makedirs(OutputDir)

    # Read galaxy properties
    print('Reading galaxy properties from', DirName+FileName, '\n')

    StellarMassFull = [0]*(LastSnap-FirstSnap+1)
    SfrDiskFull = [0]*(LastSnap-FirstSnap+1)
    SfrBulgeFull = [0]*(LastSnap-FirstSnap+1)
    BlackHoleMassFull = [0]*(LastSnap-FirstSnap+1)
    BulgeMassFull = [0]*(LastSnap-FirstSnap+1)
    HaloMassFull = [0]*(LastSnap-FirstSnap+1)
    cgmFull = [0]*(LastSnap-FirstSnap+1)
    hotgasFull = [0]*(LastSnap-FirstSnap+1)
    fullcgmFull = [0]*(LastSnap-FirstSnap+1)
    TypeFull = [0]*(LastSnap-FirstSnap+1)
    OutflowRateFull = [0]*(LastSnap-FirstSnap+1)
    coldgasFull = [0]*(LastSnap-FirstSnap+1)
    dT = [0]*(LastSnap-FirstSnap+1)
    RegimeFull = [0]*(LastSnap-FirstSnap+1)
    DiskRadiusFull = [0]*(LastSnap-FirstSnap+1)
    BulgeRadiusFull = [0]*(LastSnap-FirstSnap+1)
    RvirFull = [0]*(LastSnap-FirstSnap+1)
    FFBRegimeFull = [0]*(LastSnap-FirstSnap+1)

    for snap in range(FirstSnap,LastSnap+1):

        Snapshot = 'Snap_'+str(snap)

        StellarMassFull[snap] = read_hdf(snap_num = Snapshot, param = 'StellarMass') * 1.0e10 / Hubble_h
        SfrDiskFull[snap] = read_hdf(snap_num = Snapshot, param = 'SfrDisk')
        SfrBulgeFull[snap] = read_hdf(snap_num = Snapshot, param = 'SfrBulge')
        BlackHoleMassFull[snap] = read_hdf(snap_num = Snapshot, param = 'BlackHoleMass') * 1.0e10 / Hubble_h
        BulgeMassFull[snap] = read_hdf(snap_num = Snapshot, param = 'BulgeMass') * 1.0e10 / Hubble_h
        HaloMassFull[snap] = read_hdf(snap_num = Snapshot, param = 'Mvir') * 1.0e10 / Hubble_h
        cgmFull[snap] = read_hdf(snap_num = Snapshot, param = 'CGMgas') * 1.0e10 / Hubble_h
        hotgasFull[snap] = read_hdf(snap_num = Snapshot, param = 'HotGas') * 1.0e10 / Hubble_h
        fullcgmFull[snap] = (read_hdf(snap_num = Snapshot, param = 'CGMgas') + read_hdf(snap_num = Snapshot, param = 'HotGas')) * 1.0e10 / Hubble_h
        TypeFull[snap] = read_hdf(snap_num = Snapshot, param = 'Type')
        OutflowRateFull[snap] = read_hdf(snap_num = Snapshot, param = 'OutflowRate')
        coldgasFull[snap] = read_hdf(snap_num = Snapshot, param = 'ColdGas') * 1.0e10 / Hubble_h
        dT[snap] = read_hdf(snap_num = Snapshot, param = 'dT')
        RegimeFull[snap] = read_hdf(snap_num = Snapshot, param = 'Regime')
        FFBRegimeFull[snap] = read_hdf(snap_num = Snapshot, param = 'FFBRegime')
        DiskRadiusFull[snap] = read_hdf(snap_num = Snapshot, param = 'DiskRadius') / Hubble_h
        BulgeRadiusFull[snap] = read_hdf(snap_num = Snapshot, param = 'BulgeRadius') / Hubble_h
        RvirFull[snap] = read_hdf(snap_num = Snapshot, param = 'Rvir') / Hubble_h

#__________________________________________________
#--------------------------------------------------
# Here is the plotting part of the BH Mass Function
#__________________________________________________
#--------------------------------------------------

# Black Hole Mass Function at specific redshifts
    print('Plotting black hole mass function at specific redshifts')

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    # Define redshifts to plot (these correspond to the observation files)
    bhmf_redshifts = [0.1, 1.0, 2.0, 4.0, 6.0, 8.0]
    
    # Map redshifts to closest snapshots
    bhmf_snapshots = []
    actual_redshifts = []
    for target_z in bhmf_redshifts:
        snap_idx = np.argmin(np.abs(np.array(redshifts) - target_z))
        bhmf_snapshots.append(snap_idx)
        actual_redshifts.append(redshifts[snap_idx])
    
    # Define colormap - plasma from dark to light
    colors_bhmf = plt.cm.plasma(np.linspace(0.1, 0.9, len(bhmf_redshifts)))
    
    # Define mass bins for BHMF
    bhmf_mass_bins = np.arange(5.0, 11.5, 0.1)
    bhmf_mass_centers = bhmf_mass_bins[:-1] + 0.05
    bin_width = bhmf_mass_bins[1] - bhmf_mass_bins[0]
    
    # Plot SAGE model predictions for each redshift
    for i, (snap_idx, target_z, actual_z) in enumerate(zip(bhmf_snapshots, bhmf_redshifts, actual_redshifts)):
        # Filter for galaxies with black holes
        w = np.where(BlackHoleMassFull[snap_idx] > 0.0)[0]
        
        if len(w) > 0:
            bh_masses = np.log10(BlackHoleMassFull[snap_idx][w])
            counts, bin_edges = np.histogram(bh_masses, bins=bhmf_mass_bins)
            phi = counts / (volume * bin_width)
            
            # Only plot where we have data
            valid = phi > 0
            if np.any(valid):
                label = f'z = {actual_z:.1f} (SAGE)'
                ax.plot(bhmf_mass_centers[valid], phi[valid], 
                       color=colors_bhmf[i], linewidth=2, linestyle='-', label=label)
    
    # Load and plot observational data
    data_dir = './data/'
    obs_files = {
        0.1: 'fig4_bhmf_z0.1.txt',
        1.0: 'fig4_bhmf_z1.0.txt',
        2.0: 'fig4_bhmf_z2.0.txt',
        4.0: 'fig4_bhmf_z4.0.txt',
        6.0: 'fig4_bhmf_z6.0.txt',
        8.0: 'fig4_bhmf_z8.0.txt'
    }
    
    for i, target_z in enumerate(bhmf_redshifts):
        if target_z in obs_files:
            obs_file = data_dir + obs_files[target_z]
            try:
                # Load observation data (skip header lines starting with #)
                obs_data = np.loadtxt(obs_file)
                obs_mass = obs_data[:, 0]     # log10(Mbh [Msun])
                obs_phi = obs_data[:, 1]      # BHMF_best [Mpc^-3 dex^-1]
                obs_phi_16th = obs_data[:, 2] # BHMF_16th [Mpc^-3 dex^-1]
                obs_phi_84th = obs_data[:, 3] # BHMF_84th [Mpc^-3 dex^-1]
                
                # Plot observations with dashed line
                label = f'z = {target_z:.1f} (Obs)'
                ax.plot(obs_mass, obs_phi, color=colors_bhmf[i], 
                       linewidth=2, linestyle='--', label=label, alpha=0.8)
                
                # Add shaded error region for observations
                ax.fill_between(obs_mass, obs_phi_16th, obs_phi_84th,
                               color=colors_bhmf[i], alpha=0.2)
            except Exception as e:
                print(f'Warning: Could not load {obs_file}: {e}')
    
    # Set log scale and limits
    ax.set_yscale('log')
    ax.set_xlim(5.0, 10.0)
    ax.set_ylim(1e-5, 1e-2)
    
    # Labels and formatting
    ax.set_xlabel(r'$\log_{10} M_{\rm BH} [M_\odot]$', fontsize=14)
    ax.set_ylabel(r'$\phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=14)
    
    # Set minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    
    # Create legend
    leg = ax.legend(loc='upper right', fontsize=9, frameon=False, ncol=2)
    for text in leg.get_texts():
        text.set_fontsize(9)
    
    plt.tight_layout()
    
    # Save the plot
    outputFile = OutputDir + 'New_BlackHoleMassFunction' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print('Saved file to', outputFile, '\n')
    plt.close()