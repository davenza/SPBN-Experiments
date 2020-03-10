import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

r = pd.read_csv('cv_results.csv')
# r.set_index('Dataset', inplace=True)
print(r.shape[0])

folds = [10]
patience = [0, 5]

BAR_SEPARATION = 0.8
GROUP_SEPARATION = 0.2

COLOR1 = "#B5EA7F"
COLOR2 = "#739DF6"
COLOR3 = "#00000080"
COLOR4 = "#E3E971"

DATASETS_PER_ROW = 5

def draw_dfs(partial_df, image_name):
    n_datasets = min(partial_df.shape[0], DATASETS_PER_ROW)

    fig, axs = plt.subplots(1, n_datasets, sharex=True, squeeze=False)
    fig.set_figheight(5)
    fig.set_figwidth(16)

    for (idx, (i, row)) in enumerate(partial_df.iterrows()):
        n_validation_bars = len(folds) * len(patience)

        ckde_scores = [row['CKDE_Validation_' + str(f) + "_" + str(p)] for f in folds for p in patience]
        gaussian_scores = [row['Gaussian_Validation_' + str(f) + "_" + str(p)] for f in folds for p in patience]
        bic_score = row["BIC"]
        bge_score = row["BGe"]

        scores = ckde_scores + gaussian_scores + [bic_score] + [bge_score]

        color = [COLOR1]*n_validation_bars + [COLOR2]*n_validation_bars + [COLOR3] + [COLOR4]

        LENGTH_GROUP = n_validation_bars * BAR_SEPARATION

        xaxis = np.arange(n_validation_bars)*BAR_SEPARATION
        xaxis = np.hstack((xaxis, LENGTH_GROUP + np.arange(n_validation_bars)*BAR_SEPARATION + GROUP_SEPARATION))
        xaxis = np.hstack((xaxis, LENGTH_GROUP*2 + GROUP_SEPARATION*2, LENGTH_GROUP*2 + BAR_SEPARATION + GROUP_SEPARATION*3))

        bars = axs[0, idx].bar(xaxis, scores, color=color, linewidth=0.5, edgecolor="black")
        axs[0, idx].set_xticks([])
        axs[0, idx].set_title(row["Dataset"])

        ymin, ymax = axs[0,idx].get_ylim()
        ymax_abs = np.maximum(np.abs(ymin), np.abs(ymax))
        axs[0,idx].set_ylim(-ymax_abs, ymax_abs)
        axs[0,idx].axhline(0, linewidth=3, color='black')

        axs[0,idx].spines['top'].set_visible(False)
        axs[0,idx].spines['bottom'].set_visible(False)
        axs[0,idx].spines['right'].set_visible(False)
        # axs[0,idx].spines['left'].set_visible(False)

        # axs[0, idx].set_ylim(np.asarray(scores).min(), np.asarray(scores).max())

        patterns = (None, '//', None, '//', None, None)
        for bar, pattern in zip(bars, patterns):
            if pattern is not None:
                bar.set_hatch(pattern)

    COLOR1_patch = mpatches.Patch(color=COLOR1, label='SPBN')
    COLOR2_patch = mpatches.Patch(color=COLOR2, label='GBN Validation')
    COLOR3_patch = mpatches.Patch(color=COLOR3, label='GBN BIC')
    COLOR4_patch = mpatches.Patch(color=COLOR4, label='GBN BGe')

    PATIENCE0_PATCH = mpatches.Patch(fill=False, label="Patience 0")

    SPACE_PATCH = matplotlib.lines.Line2D([],[],linestyle='')
    PATIENCE5_PATCH = mpatches.Patch(hatch='//', fill=False,  label="Patience 5")
    # PATIENCE5_PATCH.set_hatch('////')
    plt.legend(handles=[COLOR1_patch, COLOR2_patch, COLOR3_patch, COLOR4_patch, SPACE_PATCH,
                        PATIENCE0_PATCH, PATIENCE5_PATCH])

    plt.subplots_adjust(wspace=0.4, hspace=0)
    plt.tight_layout()
    plt.savefig(image_name, dpi=300)
    # plt.show()


if __name__ == '__main__':
    for idx, partial_df in enumerate(r.loc[i:i + DATASETS_PER_ROW - 1, :] for i in range(0, r.shape[0], DATASETS_PER_ROW)):
        draw_dfs(partial_df, "results" + str(idx) + ".pdf")


# def

# fig, axs = plt.subplots(1, 5, sharex=True)
#
#
# for (i, row) in r.iterrows():
#     if i != 'Block':
#         continue
#     n_validation = len(folds) * len(patience)
#
#     ckde_scores = [row['CKDE_Validation_' + str(f) + "_" + str(p)] for f in folds for p in patience]
#     gaussian_scores = [row['Gaussian_Validation_' + str(f) + "_" + str(p)] for f in folds for p in patience]
#     bic_score = row["BIC"]
#     bge_score = row["BGe"]
#
#     scores = ckde_scores + gaussian_scores + [bic_score] + [bge_score]
#
#     color = [COLOR1]*n_validation + [COLOR2]*n_validation + [COLOR3] + [COLOR4]
#
#     LENGTH_GROUP = n_validation * BAR_SEPARATION
#
#     xaxis = np.arange(n_validation)*BAR_SEPARATION
#     xaxis = np.hstack((xaxis, LENGTH_GROUP + np.arange(n_validation)*BAR_SEPARATION + GROUP_SEPARATION))
#     xaxis = np.hstack((xaxis, LENGTH_GROUP*2 + GROUP_SEPARATION*2, LENGTH_GROUP*2 + BAR_SEPARATION + GROUP_SEPARATION*3))
#
#     print(xaxis)
#     bars = axs[0].bar(xaxis, scores, color=color, linewidth=0.5, edgecolor="black")
#     axs[0].set_xticks([])
#
#     patterns = (None, '//', None, '//', None, None)
#     for bar, pattern in zip(bars, patterns):
#         if pattern is not None:
#             bar.set_hatch(pattern)
#
# plt.subplots_adjust(wspace=0.2, hspace=0)
# plt.show()