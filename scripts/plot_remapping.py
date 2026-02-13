import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_input_chunks(chunks: list[tuple[float, float, float]], ax: plt.Axes):
    ax.set_xlabel('Real time')
    ax.set_ylabel('Inpout chunk number')
    for chunk_idx, (send_time, busy_start_time, busy_time) in enumerate(chunks):
        ax.scatter([send_time], [chunk_idx], color='black', s=30)
        ax.plot([busy_start_time, busy_start_time + busy_time], [chunk_idx, chunk_idx], color='red', lw=3, zorder=-1)

fig, axs = plt.subplots(ncols=3, figsize=(9, 3), gridspec_kw={'width_ratios': [2, 2, 1]})

axs[0].set_title('Actual streaming inference')
axs[0].set_xlim(-0.5, 7.5)
draw_input_chunks([
    (0, 0, 0.5),
    (0, 0.5, 2.5),
    (0, 3, 0.25),
    (0, 3.25, 0.25),
    (0, 3.5, 0.25),
    (0, 3.75, 0.25),
    (0, 4, 0.25),
    (0, 4.25, 0.25),
], axs[0])

axs[1].set_title('After time remapping')
axs[1].set_xlim(-0.5, 7.5)
draw_input_chunks([
    (0, 0, 0.5),
    (1, 1, 2.5),
    (2, 3.5, 0.25),
    (3, 3.75, 0.25),
    (4, 4, 0.25),
    (5, 5, 0.25),
    (6, 6, 0.25),
    (7, 7, 0.25),
], axs[1])
axs[1].axvspan(0.5, 1, color='blue', alpha=0.1)
axs[1].axvspan(4.25, 5, color='blue', alpha=0.1)
axs[1].axvspan(5.25, 6, color='blue', alpha=0.1)
axs[1].axvspan(6.25, 7, color='blue', alpha=0.1)

# legend
axs[2].axis('off')
axs[2].set_xlim(0, 1)
axs[2].set_ylim(0, 1)
shift = 0.8
axs[2].scatter([0.0], [shift], color='black', s=30, clip_on=False)
axs[2].text(0.2, shift, 'input chunk send time', va='center')
axs[2].plot([-0.1, 0.1], [shift - 0.15, shift - 0.15], color='red', lw=3, clip_on=False)
axs[2].text(0.2, shift - 0.15, 'input chunk busy time', va='center')
axs[2].add_patch(patches.Rectangle(
    (-0.1, shift - 0.35), 0.2, 0.1, clip_on=False, linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.1,
))
axs[2].text(0.2, shift - 0.3, 'artificial delays added', va='center')
plt.tight_layout()
plt.savefig('remapping.svg')