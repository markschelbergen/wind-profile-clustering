import numpy as np
import matplotlib.pyplot as plt
from dimensionality_reduction import log_law_wind_profile2, log_law_wind_profile3
from utils import add_panel_labels


def power_law(z, v_ref, z_ref, e=1./7):
    return v_ref*(z/z_ref)**e


heights = np.linspace(0, 200, 100)
roughness_lengths = [.1, .0002]
obukhov_lengths_mark = [-100, -350, 1e10, 350, 100]
lbls = ['VU', 'U', 'N', 'S', 'VS']

# plt.figure(figsize=(4, 3))
# plt.subplots_adjust(top=0.96, bottom=0.17, left=0.155, right=0.665)
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(6.5, 3))
plt.subplots_adjust(top=0.96, bottom=0.17, left=0.14, right=0.780, wspace=0.250)

rl = roughness_lengths[0]
# print(log_law_wind_profile3(np.e**.41*rl, rl, 1, 0))  # Verify

heights[0] = rl
for i, ol in enumerate(obukhov_lengths_mark):
    if ol == 1e10:
        v_log = log_law_wind_profile3(heights, rl, 1, 0)
    else:
        v_log = log_law_wind_profile3(heights, rl, 1, ol)
    ax[0].plot([0] + list(v_log), [0] + list(heights), color='C{}'.format(i), label=lbls[i])

    if ol == 1e10:
        v_log = log_law_wind_profile2(heights, rl, 1, 200, 0)
    else:
        v_log = log_law_wind_profile2(heights, rl, 1, 200, ol)
    ax[1].plot([0] + list(v_log), [0] + list(heights), color='C{}'.format(i), label=lbls[i])

rl = roughness_lengths[1]
# print(log_law_wind_profile3(np.e**.41*rl, rl, 1, 0))  # Verify

heights[0] = rl
v_log = log_law_wind_profile3(heights, rl, 1, 0)
ax[0].plot([0] + list(v_log), [0] + list(heights), '--', color='C2', label='RL=0.0002')
# plt.plot(v_log - log_law_wind_profile3(heights, roughness_lengths[0], 1, 0), heights, label='diff')

v_log = log_law_wind_profile2(heights, rl, 1, 200, 0)
ax[1].plot([0] + list(v_log), [0] + list(heights), '--', color='C2', label='RL=0.0002')

heights[0] = 0
ax[1].plot(power_law(heights, 1, 200), list(heights), '--', color='C5', label=r'$\alpha$=1/7')
# ax[1].plot(power_law(heights, 1, 200, 1/13), list(heights), '--', color='C6', label=r'$\alpha$=1/13')

ax[0].set_xlabel(r"$v_{\rm w}\,/\,v_*$ [-]")
ax[1].set_xlabel(r"$\tilde{v}_{\rm w}$ [-]")
ax[0].set_ylabel("Height [m]")
ax[0].set_xlim([0, None])
ax[1].set_xlim([0, 1.1])
ax[0].set_ylim([0, 200])
for a in ax:
    a.grid()
ax[1].legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
add_panel_labels(ax, [.48, .2])
plt.show()