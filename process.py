import numpy as np
from modules import esp_ranging
from modules import acm_fig
import yaml
import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from pathlib import Path
import pandas as pd
import math
acm_fig.set_ieee()

# CONSTANTS

# function to calculate SSE of guess
def residuals(p, anchors, ranges, weights=None):
    dists = np.sqrt((anchors[:,0]-p[0])**2 + (anchors[:,1]-p[1])**2)
    r = dists - ranges
    if weights is not None:
        r = np.sqrt(weights) * r
    return r


MAX_FTM_TIMESTAMP_ERROR = 300

# SECTION 1: LOAD/COMPUTE variables
data_dir = Path("/home/bahor/WCSNG_RESEARCH/esp_ranging_6_node_fah")
host = "9C:13:9E:AA:E7:ED"

# MAC Addresses of Anchors
tgts = [
    "9C:13:9E:AB:A9:69",
    "68:B6:B3:2D:57:45",
    "DC:B4:D9:10:44:6D",
    "B8:F8:62:EC:78:C1",
    "B8:F8:62:EC:81:5D",
    "B8:F8:62:EC:73:01",
]

# Position in (X,Y) Plane of Anchors
tgt_pos = np.asarray([
    [-0.584,1.386],
    [-3.674,-0.909],
    [13.122,-7.769],
    [13.211,9.990],
    [11.126,12.303],
    [-2.247,15.155]
    ])

# initial guess for SSE calculations
init_guess = tgt_pos.mean(axis=0)

# Load in FTM/CSI Pairs from CSV
delta_z = 1.0
SYNC_METHOD = "CLOSEST"
N_MED = 3
BOT_DOWNSAMPLE = 1
t_ms = {}
t_fine = {}
H = {}
rssi = {}
for tgt in tgts:
    _t_ms, _t_fine, _H, _rssi = esp_ranging.generate_ftm_csi_pairs(data_dir / host.replace(':', '_'), tgt)
    t_ms[tgt] = _t_ms
    t_fine[tgt] = np.asarray(_t_fine) # t_fine[k][i] = FTM Timing value of measurement i from anchor k 
    H[tgt] = np.asarray(_H) # H[k][i] = CSI matrix of measurement i from anchor k 
    rssi[tgt] = np.asarray(_rssi) #rssi[k][i] = signal strength of measurement i from anchor k


t_esp = {k:np.asarray(t_ms[k])/1e3 for k in t_ms} # t_esp[k][i] = timestamp of measurement i from anchor k 



traj = pd.read_csv(data_dir / "trajectory_xy.csv")
t_bot = np.asarray(traj['t_sec'])[::BOT_DOWNSAMPLE]
xy_gnd = np.vstack((np.asarray(traj['x']), np.asarray(traj['y']))).T[::BOT_DOWNSAMPLE,:]  #xy_gnd[i] = ground truth positioning for sample i 


with open(data_dir/"map.yaml") as f:
    map_yaml = yaml.safe_load(f)
map_img = plt.imread(data_dir / "map.pgm")
map_los = plt.imread(data_dir / "map_sightlines.pgm")
map_orig = map_yaml['origin']
map_res = map_yaml['resolution']
map_shape = map_img.shape

los_bot = np.zeros((t_bot.shape[0])) #los_bot[i] = # of anchors with LOS at sample i 


# sync to bot positions
t_start = max([t_bot[0]] + [t_esp[k][0] for k in t_esp])
t_end = min([t_bot[-1]] + [t_esp[k][-1] for k in t_esp])
print(f"Experiment Duration: {t_end - t_start}")
in_bot = (t_bot > t_start) & (t_bot < t_end)
in_esp = {k:(t_esp[k] > t_start) & (t_esp[k] < t_end) for k in t_esp}
print(f"Bot data: {np.count_nonzero(in_bot)} ESP data: {[np.count_nonzero(in_esp[k]) for k in in_esp]}")
t_bot = t_bot[in_bot]
xy_gnd = xy_gnd[in_bot]
H = {k:H[k][in_esp[k], :] for k in H}
t = {k:t_fine[k][in_esp[k], :] for k in t_fine}
rssi = {k:rssi[k][in_esp[k]] for k in rssi}
t_esp = {k:t_esp[k][in_esp[k]] for k in t_esp}
sync_idx = {k:np.zeros(len(t_bot), dtype=int) for k in t_esp}
if SYNC_METHOD == "CLOSEST":
    for k in t_esp:
        for si in range(len(t_bot)):
            sync_idx[k][si] = np.argmin(np.abs(t_bot[si] - t_esp[k]))
elif SYNC_METHOD == "MOST_RECENT":
    for k in t_esp:
        for si in range(len(t_bot)):
            delta = t_bot[si] - t_esp[k]
            if np.all(delta < 0):
                sync_idx[k][si] = 0
            elif np.all(delta > 0):
                sync_idx[k][si] = delta.shape[0] - 1
            else:
                sync_idx[k][si] = np.argmax(delta < 0) - 1
elif SYNC_METHOD == "LAST_N_MED":
    for k in t_esp:
        for si in range(len(t_bot)):
            delta = t_bot[si] - t_esp[k]
            if np.all(delta < 0):
                sync_idx[k][si] = 0
            elif np.all(delta > 0):
                sync_idx[k][si] = delta.shape[0] - 1
            else:
                idx = np.argmax(delta < 0) - 1
                esp_t_fine = t[k][max(0,idx+1-N_MED):idx+1,:]
                delay = ((esp_t_fine[:,3] - esp_t_fine[:,0]) - (esp_t_fine[:,2] - esp_t_fine[:,1])) / 2e12
                try:
                    sync_idx[k][si] = max(0,idx+1-N_MED) + np.argsort(delay)[len(delay)//2]
                except:
                    breakpoint()
                    
                    

# SECTION 2 Localization Algorithms:
# Used Variables
# t[k][i] = FTM Timing values of measurement i from anchor k 
# H[k][i] = CSI matrix of measurement i from anchor k 
# rssi[k][i] = signal strength of measurement i from anchor k
# t_esp[k][i] = timestamp of measurement i from anchor k 
# t_bot[i] = timestamp of measurement i from robot 
# xy_gnd[i] = ground truth positioning for sample i
# sync_idx[k][si] = index of anchor k to use for sample i of robot location 



# Naive Trilateration Algorithm using Ground truth values (used to verify SSE function is accurate):
xy_est = np.zeros_like(xy_gnd)

for i in range(len(t_bot)):
    ground_truth_dist = []
    for j in range(len(tgts)):
        x_anc, y_anc = tgt_pos[j]
        x_true, y_true = xy_gnd[i]
        dist = math.sqrt((abs(x_true - x_anc) ** 2) + (abs(y_true - y_anc) ** 2))
        ground_truth_dist.append(dist)
    res = least_squares(residuals, xy_gnd[i], args=(tgt_pos, ground_truth_dist))
    est_x, est_y = res.x
    xy_est[i] = (est_x, est_y)

print("Error of Ground Truth Calculations")
err = np.linalg.norm(xy_est - xy_gnd, axis=1)  # meters

print(f"count={err.size}")
print(f"mean={np.mean(err):.2f} m, median={np.median(err):.2f} m")
print(f"p90={np.percentile(err,90):.2f} m, p95={np.percentile(err,95):.2f} m")
print(f"rmse={np.sqrt(np.mean(err**2)):.2f} m")

        
    


# Naive trilateration Algorithm using FTM measurements only:

xy_est = np.zeros_like(xy_gnd)

esp_timestamp_dist_from_ftm_timestamp = []  # mean of used anchors only
timestamp_dist_per_anchor = {anchor: [] for anchor in tgts}  # per-anchor used distances (NaN if skipped)

for i in range (len(t_bot)):
    # calculate ftm measured distance for each anchor 
    ftm_measured_dist = []
    anchors_used = []
    timestamp_used = []
    # calculate distance from anchor
    for anchor_idx, anchor in enumerate(tgts):
        idx = sync_idx[anchor][i]
        dist = esp_ranging.calculate_ftm_distance(t[anchor][idx])
        dist_from_ftm_timestamp = abs(t_esp[anchor][idx] - t_bot[i]) 
        # Skip this anchor if its timestamp is more than 3s away
        if dist_from_ftm_timestamp >= 3:
            timestamp_dist_per_anchor[anchor].append(np.nan)
            continue

        # Anchor is used
        anchors_used.append(tgt_pos[anchor_idx])
        ftm_measured_dist.append(dist)
        timestamp_used.append(dist_from_ftm_timestamp)
        timestamp_dist_per_anchor[anchor].append(dist_from_ftm_timestamp)

    mean_dist = np.mean(timestamp_used) if len(timestamp_used) else np.nan
    esp_timestamp_dist_from_ftm_timestamp.append(mean_dist)

    if len(ftm_measured_dist) == 0:
        xy_est[i] = xy_gnd[i - 1]  # no valid anchors; fall back to prev val 
        continue
        
        
        
    
    res = least_squares(residuals, xy_gnd[i], args=(np.asarray(anchors_used), ftm_measured_dist))
    est_x, est_y = res.x

    xy_est[i] = (est_x, est_y)
    
print("Error of Naive FTM calculations")

err = np.linalg.norm(xy_est - xy_gnd, axis=1)

print(f"count={err.size}")
print(f"mean={np.mean(err):.2f} m, median={np.median(err):.2f} m")
print(f"p90={np.percentile(err,90):.2f} m, p95={np.percentile(err,95):.2f} m")
print(f"rmse={np.sqrt(np.mean(err**2)):.2f} m")


# Weighted Naive Trilateration Algorithm:
xy_est = np.zeros_like(xy_gnd)

for i in range (len(t_bot)):
    # calculate ftm measured distance for each anchor 
    ftm_measured_dist = []
    weights = []
    anchors_used = []
    # calculate distance from anchor
    for anchor_idx, anchor in enumerate(tgts):
        
        idx = sync_idx[anchor][i]
        dist = esp_ranging.calculate_ftm_distance(t[anchor][idx])
        dist_from_ftm_timestamp = abs(t_esp[anchor][idx] - t_bot[i])

        # Skip this anchor if its timestamp is more than 3s away
        if dist_from_ftm_timestamp >= 3:
            continue

        anchors_used.append(tgt_pos[anchor_idx])
        ftm_measured_dist.append(dist)
        rssi_val = rssi[anchor][idx]
        weights.append(10 ** (rssi_val / 20))
        
    if len(ftm_measured_dist) == 0:
        xy_est[i] = xy_gnd[i]  # no valid anchors; fall back to prev val
        continue
    
    res = least_squares(residuals, xy_gnd[i], args=(np.asarray(anchors_used), ftm_measured_dist, weights))
    if(res.success is False):
        print("UNABLE TO SOLVE SSE")
    est_x, est_y = res.x
    real_x, real_y = xy_gnd[i]
    xy_est[i] = (est_x, est_y)
    
print("Error of Weighted FTM calculations")

err = np.linalg.norm(xy_est - xy_gnd, axis=1)  # meters

print(f"count={err.size}")
print(f"mean={np.mean(err):.2f} m, median={np.median(err):.2f} m")
print(f"p90={np.percentile(err,90):.2f} m, p95={np.percentile(err,95):.2f} m")
print(f"rmse={np.sqrt(np.mean(err**2)):.2f} m")

# Plot trajectory estimates
"""
plt.figure(figsize=(10, 8))
plt.plot(xy_gnd[:, 0], xy_gnd[:, 1], 'k-', linewidth=2, label='Ground Truth')
plt.plot(xy_est[:, 0], xy_est[:, 1], 'g.', markersize=4, label='Weighted FTM Estimate')
plt.scatter(tgt_pos[:, 0], tgt_pos[:, 1], c='r', s=100, marker='x', label='Anchors')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Weighted FTM Trilateration - Estimated vs Ground Truth Trajectory')
plt.legend()
plt.axis('equal')
plt.grid(True)

plt.show()
"""


# Plot predicted vs ground-truth coordinates over time (weighted FTM)
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(t_bot, xy_gnd[:, 0], 'k-', linewidth=1.2, label='x_true')
plt.plot(t_bot, xy_est[:, 0], 'g--', linewidth=1.0, label='x_pred')
plt.xlabel('Time (s)')
plt.ylabel('X (m)')
plt.title('X over Time: Predicted vs Ground Truth (Weighted FTM)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_bot, xy_gnd[:, 1], 'k-', linewidth=1.2, label='y_true')
plt.plot(t_bot, xy_est[:, 1], 'g--', linewidth=1.0, label='y_pred')
plt.xlabel('Time (s)')
plt.ylabel('Y (m)')
plt.title('Y over Time: Predicted vs Ground Truth (Weighted FTM)')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Plot timestamp distance over time
plt.figure(figsize=(14, 8))

