import matplotlib.pyplot as plt
import os
import numpy as np
import os
from collections import Counter

MODEL_TITLE_DICT={"llama-2-7b": "LLaMA-2-7B", "mistral-7b": "Mistral-7B", 
        "llama-2-13b-chat": "LLaMA-2-13B-chat", "llama2-70b-chat": "LLaMA-2-70B-chat",
        "llama-2-7b-chat": "LLaMA-2-7B-chat", "llama-2-13b": "LLaMA-2-13B", "llama-2-70b": "LLaMA-2-70B", 
        "llama-3-8b": "LLaMA-3-8B","llama-3-70b": "LLaMA-3-70B","qwen-2-0.5b": "Qwen-2-0.5B","qwen-2-1.5b": "Qwen-2-1.5B",
        "llama-3-8b-instruct": "LLaMA-3-8B-Instruct","llama-3-70b-instruct": "LLaMA-3-70B-Instruct",
        "qwen-2-7b": "Qwen-2-7B","internlm-2.5-7b":"InternLM-2.5-7B", "phi-3-medium-instruct":"Phi-3-Medium-Instruct",
        "mistral-7b-v0.3":"Mistral-7B-V0.3","dclm-7b":"DCLM-7B", "llama-3.1-8b": "LLaMA-3.1-8B",
        "gemma-2-9b":"Gemma-2-9B"}

def plot_3D_tensor(layer_name, tensor, name):
    print(f"drawing {layer_name}")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(tensor.shape[1])
    Y = np.arange(tensor.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y,  tensor.cpu(), cmap='coolwarm', antialiased=False, shade=True, linewidth=0.5,rstride=1,cstride=1)
    plt.tight_layout()
    
    ax.set_xlabel('Channel', fontsize=14)
    ax.set_ylabel('Token', fontsize=14)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='z', labelsize=16)
    # ax.set_zlabel('Abs Magnitude')
    ax.view_init(elev=20., azim=-45)

    plt.tight_layout(pad=0.1)
    plt.savefig(f'{name}', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_layer_ax_input_sub(ax, mean, model_name, layer_name, show_ylabel=True):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey", "gold"]

    x_axis = np.arange(mean.shape[-1])+1
    for i in range(3):
        ax.plot(x_axis, mean[i], label=f"Top-{i+1}", color=colors[i], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)

    ax.plot(x_axis, mean[-2], label=f"Median", color=colors[-2], 
                     linestyle="-",  marker="v", markerfacecolor='none', markersize=5)
    ax.plot(x_axis, mean[-1], label=f"Min-1", color=colors[-1], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)

    if layer_name == 'q_proj':
        layer_name = 'q/k/v_proj'
    elif layer_name == 'up_proj':
        layer_name = 'up/gate_proj'
    title = f'{MODEL_TITLE_DICT[model_name]} {layer_name}'
    ax.set_title(title, fontsize=22, fontweight="bold")

    num_layers = mean.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label, xtick_label, fontsize=22)
    ax.set_xlabel('Layers', fontsize=22, labelpad=0.8)
    if show_ylabel:
        ax.set_ylabel("Maximum Value(token-wise)", fontsize=22, fontweight='bold')
    # ax.set_yticks(ax.get_yticks())
    ax.yaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_ticklabels(ax.get_yticklabels(), fontweight='bold')
    ax.grid(axis='x', color='0.75')
    ratio1 = (mean[0]/mean[-2]).max() # top-1/median
    if ratio1>10:
        color1 = 'red'
    else:
        color1 = 'green'
    ratio2 = (mean[-2]/mean[-1]).max() # median/min-1
    if ratio2>10:
        color2 = 'red'
    else:
        color2 = 'green'
    if ratio1>100:
        text1 = rf"max($\frac{{\text{{top-1}}}}{{\text{{median}}}}$)={ratio1:.0f} "
    else:
        text1 = rf"max($\frac{{\text{{top-1}}}}{{\text{{median}}}}$)={ratio1:.1f} "
    if ratio2 > 100:
        text2 = rf"max($\frac{{\text{{median}}}}{{\text{{min-1}}}}$)={ratio2:.0f}"
    else:
        text2 = rf"max($\frac{{\text{{median}}}}{{\text{{min-1}}}}$)={ratio2:.1f}"
    char_width = 0.019
    text1_length = len(text1) * char_width
    ax.text(0.5 - text1_length / 2 - char_width, 0.93, text1, 
            va='center', ha='left', fontsize=22, fontweight='bold', transform=ax.transAxes, color=color1)
    ax.text(0.5 + char_width, 0.93, text2, 
            va='center', ha='left', fontsize=22, fontweight='bold', transform=ax.transAxes, color=color2)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + (y_max-y_min)*0.1)

def plot_layer_ax_input(obj, model_name, savedir, layer_name, show_legend=True):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)
    mean = np.mean(obj,axis=0)
    plot_layer_ax_input_sub(axs, mean, model_name, layer_name)
    if show_legend:
        leg = axs.legend(
            loc='center', bbox_to_anchor=(0.5, -0.10),
            ncol=5, fancybox=True, prop={'size': 14}
        )
        leg.get_frame().set_edgecolor('silver')
        leg.get_frame().set_linewidth(1.0)
    plt.savefig(os.path.join(savedir,f"{model_name}-{layer_name}.png"), bbox_inches="tight", dpi=200)

def plot_combined_layer_ax_input(objs, model_name, savedir, layer_names, show_legend=True):
    fig, axs = plt.subplots(nrows=1, ncols=len(layer_names), figsize=(28, 4.5))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    all_handles_labels = []
    for index, (ax, stat, layer_name) in enumerate(zip(axs.flatten(), objs, layer_names)):
        show_ylabel = index==0
        mean = np.mean(stat, axis=0)
        plot_layer_ax_input_sub(ax, mean, model_name, layer_name, show_ylabel)
        handles, labels = ax.get_legend_handles_labels()
        all_handles_labels.append((handles, labels))

    handles, labels = all_handles_labels[0]
    if show_legend:
        fig.legend(handles, labels, loc='upper center', ncol=5, fancybox=True, prop={'size': 22}, bbox_to_anchor=(0.5, 1.23))

    plt.savefig(os.path.join(savedir, f"{model_name}-combined.png"), bbox_inches="tight", dpi=600)
    plt.savefig(os.path.join(savedir, f"{model_name}-combined.pdf"), bbox_inches="tight", dpi=600)
    plt.close(fig)
    
def plot_layer_ax_output_sub(ax, mean, model_name, layer_name, show_ylabel=True):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey", "gold"]

    x_axis = np.arange(mean.shape[-1])+1
    
    ax.plot(x_axis, mean[-1], label=f"Top-1", color=colors[-1], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)
    ax.plot(x_axis, mean[-2], label=f"Median", color=colors[-2], 
                     linestyle="-",  marker="v", markerfacecolor='none', markersize=5)
    for i in range(3):
        ax.plot(x_axis, mean[i], label=f"Min-{i+1}", color=colors[i], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)


    if layer_name == 'q_proj' or layer_name == 'apply_rotary_pos_emb_qk_rotation_wrapper.Q':
        layer_name = 'Q'
    elif layer_name == 'k_proj' or layer_name == 'apply_rotary_pos_emb_qk_rotation_wrapper.K':
        layer_name = 'K'
    elif layer_name == 'v_proj':
        layer_name = 'V'
        
    title = f'{MODEL_TITLE_DICT[model_name]} {layer_name}'
    ax.set_title(title, fontsize=22, fontweight="bold")

    num_layers = mean.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label, xtick_label, fontsize=22)
    ax.set_xlabel('Layers', fontsize=22, labelpad=0.8)
    if show_ylabel:
        ax.set_ylabel("Maximum Value(token-wise)", fontsize=22, fontweight='bold')
    ax.yaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_ticklabels(ax.get_yticklabels(), fontweight='bold')
    ax.grid(axis='x', color='0.75')
    ratio1 = (mean[-1]/mean[-2]).max() # top-1/median
    if ratio1>5:
        color1 = 'red'
    else:
        color1 = 'green'
    ratio2 = (mean[-2]/mean[0]).max() # median/min-1
    if ratio2>5:
        color2 = 'red'
    else:
        color2 = 'green'
    text1 = rf"max($\frac{{\text{{top-1}}}}{{\text{{median}}}}$)={ratio1:.1f} "
    text2 = rf"max($\frac{{\text{{median}}}}{{\text{{min-1}}}}$)={ratio2:.1f}"
    char_width = 0.019
    text1_length = len(text1) * char_width
    ax.text(0.5 - text1_length / 2 - char_width, 0.93, text1, 
            va='center', ha='left', fontsize=22, fontweight='bold', transform=ax.transAxes, color=color1)
    ax.text(0.5 + char_width, 0.93, text2, 
            va='center', ha='left', fontsize=22, fontweight='bold', transform=ax.transAxes, color=color2)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + (y_max-y_min)*0.1)

def plot_layer_ax_output(obj, model_name, savedir, layer_name, show_legend=True):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)
    mean = np.mean(obj,axis=0)
    plot_layer_ax_output_sub(axs, mean, model_name, layer_name)
    if show_legend:
        leg = axs.legend(
            loc='center', bbox_to_anchor=(0.5, -0.10),
            ncol=5, fancybox=True, prop={'size': 14}
        )
        leg.get_frame().set_edgecolor('silver')
        leg.get_frame().set_linewidth(1.0)
    plt.savefig(os.path.join(savedir,f"{model_name}-{layer_name}.png"), bbox_inches="tight", dpi=600)

def plot_combined_layer_ax_output(objs, model_name, savedir, layer_names, show_legend=True):
    fig, axs = plt.subplots(nrows=1, ncols=len(layer_names), figsize=(21, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)
    all_handles_labels = []
    for index, (ax, stat, layer_name) in enumerate(zip(axs.flatten(), objs, layer_names)):
        show_ylabel = index==0
        mean = np.mean(stat, axis=0)
        plot_layer_ax_output_sub(ax, mean, model_name, layer_name, show_ylabel)
        handles, labels = ax.get_legend_handles_labels()
        all_handles_labels.append((handles, labels))

    handles, labels = all_handles_labels[0]
    if show_legend:
        fig.legend(handles, labels, loc='upper center', ncol=5, fancybox=True, prop={'size': 22}, bbox_to_anchor=(0.5, 1.23))    

    plt.savefig(os.path.join(savedir,f"{model_name}-combined.png"), bbox_inches="tight", dpi=600)
    plt.savefig(os.path.join(savedir,f"{model_name}-combined.pdf"), bbox_inches="tight", dpi=600)

def plot_layer_outlier_token_num_sub(ax, mean, model_name):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey", "gold"]

    x_axis = np.arange(mean.shape[-1])+1

    ax.plot(x_axis, mean[0], color=colors[0], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)
    # ax.plot(x_axis, mean[0], label=f"# of outlier token", color=colors[0], 
    #                  linestyle="-",  marker="o", markerfacecolor='none', markersize=5)

    ax.set_title(MODEL_TITLE_DICT[model_name], fontsize=16, fontweight="bold")

    num_layers = mean.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label, xtick_label, fontsize=16)

    ax.set_xlabel('Layers', fontsize=16, labelpad=0.8)
    ax.set_ylabel("Avg. # of outlier tokens", fontsize=16)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.grid(axis='x', color='0.75')

    
def plot_layer_outlier_token_num(obj, model_name, savedir):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)

    mean = np.mean(obj,axis=0)
    plot_layer_outlier_token_num_sub(axs, mean, model_name)

    plt.savefig(os.path.join(savedir,f"{model_name}.png"), bbox_inches="tight", dpi=200)
    
def plot_outlier_token_position_sub(ax, labels, values, model_name):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey", "gold"]

    bars = ax.bar(labels, values, color=colors[0])
    ax.set_title(MODEL_TITLE_DICT[model_name], fontsize=20, fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom',fontsize=20, fontweight="bold")


    ax.set_xlabel('Position Index', fontsize=20, labelpad=0.8, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=20)
    plt.subplots_adjust(top=0.85)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    plt.xticks(fontsize=20, fontweight="bold")
    plt.yticks(fontsize=20)
    ax.set_ylim(0, max(values) * 1.13)
    
def plot_outlier_token_position(obj, model_name, savedir):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)

    data = Counter(obj)
    total = sum(data.values())
    percentages = {k: (v / total) * 100 for k, v in data.items()}
    sorted_percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True)[:5])
    sorted_percentages['Others'] = sum(v for k, v in percentages.items() if k not in sorted_percentages)
    labels = list(sorted_percentages.keys())
    labels = [str(k) for k in sorted_percentages.keys()]
    values = list(sorted_percentages.values())
    
    plot_outlier_token_position_sub(axs, labels, values, model_name)
    plt.savefig(os.path.join(savedir,f"{model_name}.png"), bbox_inches="tight", dpi=200)
    plt.savefig(os.path.join(savedir,f"{model_name}.pdf"), bbox_inches="tight", dpi=200)

def plot_outlier_token_sub(ax, labels, values, model_name):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey", "gold"]

    bars = ax.bar(labels, values, color=colors[0])
    ax.set_title(MODEL_TITLE_DICT[model_name], fontsize=20, fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom',fontsize=20, fontweight="bold")


    ax.set_xlabel('Token Content', fontsize=20, labelpad=0.8, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=20)
    plt.subplots_adjust(top=0.85)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    plt.xticks(fontsize=20, fontweight="bold")
    plt.yticks(fontsize=20)
    ax.set_ylim(0, max(values) * 1.13)
    
def plot_outlier_token(obj, model_name, savedir):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)

    data = Counter(obj)
    total = sum(data.values())
    percentages = {k: (v / total) * 100 for k, v in data.items()}
    sorted_percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True)[:5])
    sorted_percentages['Others'] = sum(v for k, v in percentages.items() if k not in sorted_percentages)
    labels = list(sorted_percentages.keys())
    labels = [str(k) for k in sorted_percentages.keys()]
    values = list(sorted_percentages.values())
    
    plot_outlier_token_sub(axs, labels, values, model_name)
    plt.savefig(os.path.join(savedir,f"{model_name}.png"), bbox_inches="tight", dpi=200)
    plt.savefig(os.path.join(savedir,f"{model_name}.pdf"), bbox_inches="tight", dpi=200)

def plot_outlier_token_test(obj, model_name, savedir):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)

    data = Counter(obj)
    total = sum(data.values())
    percentages = {k: (v / total) * 100 for k, v in data.items()}
    sorted_percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True)[:5])
    sorted_percentages['Others'] = sum(v for k, v in percentages.items() if k not in sorted_percentages)
    labels = list(sorted_percentages.keys())
    labels = [str(k) for k in sorted_percentages.keys()]
    values = list(sorted_percentages.values())
    
    plot_outlier_token_sub(axs, labels, values, model_name)
    plt.savefig(os.path.join(savedir,f"{model_name}_test.png"), bbox_inches="tight", dpi=200)