import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle
from prettytable import PrettyTable
from sklearn.metrics import roc_curve



def Plot_Results():
    Terms = ['Training Time (Sec)', 'Mean Reward', 'Standard Deviation', 'Minimum Reward', 'Maximum Reward',
             'Optimum Action Taken (%)',
             'Average Success Rate (%)', 'Average Step Range']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7]
    Eval = np.load('Eval_Track.npy', allow_pickle=True)

    for m in range(len(Graph_Term)):
        Graph = Eval[Graph_Term[m], :, :]

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

        bar_width = 0.2  # Bar width
        methods = ['YOLOv3', 'YOLOv5', 'YOLOv9', 'FM-Yv9-NLF']
        colors_met = ['Deeppink', 'cyan', 'Green', 'Blue']

        # Introduce spacing by increasing X values
        num_groups = Graph.shape[0]
        X = np.arange(num_groups) * (bar_width * 6)  # Adding space after each group

        for j in range(4):  # Loop over methods
            bars = ax.bar(X + j * bar_width, Graph[:, 6 + j], color=colors_met[j], width=bar_width, label=methods[j])

        plt.xticks(X + (2 * bar_width), ['10', '20', '30', '40', '50'])  # Correct x-tick alignment
        plt.xlabel('No of Values')
        plt.ylabel(Terms[Graph_Term[m]])

        # Remove left and top spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(fontsize=9, frameon=False, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
        Path = f"./Results/{Terms[Graph_Term[m]]}_Method.png"
        plt.savefig(Path)
        plt.show(block=False)
        plt.pause(2)
        plt.close()

def Plot_Results_Act():
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
             'NPV',
             'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2]
    Eval = np.load('Eval_Seg_Act.npy', allow_pickle=True)
    Classifier = ['YOLOv3', 'YOLOv5', 'YOLOv9', 'FM-Yv9-NLF']

    colors_met = ['#7FFF00', '#929591', '#FF69B4', '#0000FF']

    for m in range(len(Graph_Term)):
        Graph = Eval[:, :, Graph_Term[m] + 4]


        bar_width = 0.15
        x = np.arange(Graph.shape[0])
        plt.figure(figsize=(7, 6))

        # Plot bars for each category
        for i in range(4):
            bars = plt.bar(x + (i) * bar_width, Graph[:, 6 + i], width=bar_width, label=Classifier[i],
                           color=colors_met[i])


        # Customizations
        plt.xticks(x + bar_width * 1.5, ['Linear', 'Relu', 'Tanh', 'Softmax', 'Sigmoid'], fontsize=10,
                   fontweight='bold')
        plt.xlabel('Activation Function', fontsize=12, fontweight='bold')
        plt.ylabel(Terms[Graph_Term[m]], fontsize=12, fontweight='bold')

        # Remove axes outline
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        # Custom Legend with Dot Markers, positioned at the top
        dot_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color
                       in colors_met]
        plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   fontsize=10, frameon=False, ncol=len(Classifier))

        # Add gridlines for y-axis only
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout for legend space outside the graph
        plt.tight_layout()

        # Save the figure with tight bounding box
        path = "./Results/ACT_%s_Method.png" % (Terms[Graph_Term[m]])
        plt.savefig(path, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def Plot_Results_Epoch():
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
             'NPV',
             'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2]
    Eval = np.load('Eval_Seg_epoch.npy', allow_pickle=True)

    for m in range(len(Graph_Term)):
        Graph = Eval[:, :, Graph_Term[m] + 4]

        sections = ['50', '100', '150', '200', '250']
        male_percent = Graph[:, 3]
        female_percent = Graph[:, 4]

        x = np.array([0, 2, 4, 6, 8])
        width = 0.45
        offset = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - offset, male_percent, width, label='Yolov9', color='blue')
        bars2 = ax.bar(x + offset, female_percent, width, label='FM-Yv9-NLF', color='darkgreen')

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='blue')

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='darkgreen')

        ax.set_ylabel(Terms[Graph_Term[m]])
        plt.xlabel('No of Epochs', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sections)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True,
                   prop={'family': 'serif', 'size': 10})

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

        # Show only bottom spine
        for spine_name, spine in ax.spines.items():
            spine.set_visible(spine_name == 'bottom')

        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        path = "./Results/Optimization_%s.png" % (Terms[Graph_Term[m]])
        plt.savefig(path, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(2)
        plt.close()



def Plot_ROC():
    lw = 2
    cls = ['YOLOv3', 'YOLOv5', 'YOLOv9', 'FM-Yv9-NLF']
    colors1 = cycle(["#65fe08", "#4e0550", "#f70ffa", "#004577", ])
    for i, color in zip(range(4), colors1):  # For all classifiers
        Predicted = np.load('roc_score.npy', allow_pickle=True)[i].astype('float')
        Actual = np.load('roc_act.npy', allow_pickle=True)[i].astype('int')
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label="{0}".format(cls[i]),
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/roc.png"
    plt.savefig(path)
    plt.show()


def plot_results_seg_table():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
             'NPV',
             'FDR', 'F1-Score', 'MCC']
    Dataset = ['Dataset 1', 'Dataset 2', 'Dataset 3']

    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']

    Full = ['TERMS', 'Unet', 'Unet++', 'FCN', 'DenseUnet', 'D-RNet-SE']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            Table = PrettyTable()
            Table.add_column(Full[0], Statistics)
            Table.add_column(Full[1], stats[i, 5, :])
            Table.add_column(Full[2], stats[i, 6, :])
            Table.add_column(Full[3], stats[i, 7, :])
            Table.add_column(Full[4], stats[i, 8, :])
            Table.add_column(Full[5], stats[i, 4, :])
            print('-------------------------------------------------- ', Terms[i - 4],
                  'Comparison for Segmentation -----------------------------------------------')
            print(Table)


if __name__ == '__main__':
    plot_results_seg_table()
    Plot_ROC()
    Plot_Results_Epoch()
    Plot_Results_Act()
    Plot_Results()
