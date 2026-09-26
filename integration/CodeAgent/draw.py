import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path

# ，
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_combined_4plots_academic_style():
    """
     (2x2)，：
    1. fig.suptitle 
    2. ，
    3.  Y 
    4.  (a), (b), (c), (d) 
    5. 
    """
    # ---------------------------------------------------------
    # 1.  ()
    # ---------------------------------------------------------
    
    # ---  X  (Temperature  Top-p ) ---
    temp_x_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    top_p_x_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    
    # --- Y  (Pass@1 ) ---
    #  (a) ()
    y1_scores = np.array([34.88, 32.56, 33.72, 33.72, 35.47, 33.72, 32.56, 35.47, 34.30, 33.14, 34.30])
    #  (b) ()
    y2_scores = np.array([33.72, 34.30, 32.56, 36.05, 33.14, 34.30, 33.14, 36.05, 35.47, 36.63, 34.88])
    #  (c) ()
    y3_scores = np.array([34.29, 31.43, 31.43, 34.29, 28.57, 34.29, 31.43, 31.43, 25.71, 31.43, 28.57])
    #  (d) ()
    y4_scores = np.array([31.43, 31.43, 28.57, 31.43, 28.57, 25.71, 31.43, 34.29, 34.29, 34.29, 34.29])

    # 
    model_name = "HITHCoder"
    
    # ---------------------------------------------------------
    # 2.  2x2 
    # ---------------------------------------------------------
    # figsize ，dpi 
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5), dpi=100)
    
    # ---------------------------------------------------------
    # 3.  ()
    # ---------------------------------------------------------
    #  ()
    line_color = '#789999'   #  ()
    marker_color = '#E99695' #  ()
    
    # ，
    plt.rc('font', size=11)
    
    def plot_single_panel_academic(ax, x_vals, y_vals, x_label_prefix, y_lim_range, is_left_panel=False, is_bottom_panel=False):
        """，"""
        # ：
        ax.plot(x_vals, y_vals, 
                label=model_name,
                color=line_color,       # 
                linewidth=2.0,        # 
                marker='o',           # 
                markersize=8,         # 
                markerfacecolor=marker_color, # 
                markeredgecolor=marker_color) #  ()
        
        # ---  () ---
        
        # ，
        ax.set_title("") 
        
        # X： X 
        ax.set_xlabel(f"{x_label_prefix}", fontsize=12, labelpad=10, color='#444444')

        ax.set_xlim(-0.02, 1.05) # X， 0.0  1.0 
        ax.set_xticks(x_vals) # 
        
        # Y (Pass@1)
        if is_left_panel:
            ax.set_ylabel('Pass@1', fontsize=12, labelpad=10, color='#444444')
        else:
            ax.set_ylabel("") #  Y 
            ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=False) #  Y 

        ax.set_ylim(y_lim_range) # 
        ax.set_yticks(np.arange(y_lim_range[0], y_lim_range[1] + 1, 1)) # 
        
        # X ()
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        # ---  ---
        # 
        ax.grid(True, linestyle='--', color='#DDDDDD', linewidth=0.8, alpha=0.5)
        
        # ---  ---
        #  (L)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # 
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#333333', length=6)
        # （） dark gray 
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_color('#333333')

    # ---------------------------------------------------------
    # 4.  ()
    # ---------------------------------------------------------
    #  Y 
    
    #  (a) (, temp_x_values)
    plot_single_panel_academic(axs[0, 0], temp_x_values, y1_scores, "Temperature", [30.5, 36.5], is_left_panel=True, is_bottom_panel=False)
    
    #  (b) (, top_p_x_values)
    plot_single_panel_academic(axs[0, 1], top_p_x_values, y2_scores, "Top-p", [31.5, 37.5], is_left_panel=False, is_bottom_panel=False)
    
    #  (c) (, temp_x_values)
    plot_single_panel_academic(axs[1, 0], temp_x_values, y3_scores, "Temperature", [24.5, 35.5], is_left_panel=True, is_bottom_panel=True)
    
    #  (d) (, top_p_x_values)
    plot_single_panel_academic(axs[1, 1], top_p_x_values, y4_scores, "Top-p", [24.5, 35.5], is_left_panel=False, is_bottom_panel=True)

    # ---------------------------------------------------------
    # 5. 
    # ---------------------------------------------------------
    
    #  ()
    fig.suptitle("HITHCoder - DevEval Parameter Study", fontsize=18, fontweight='bold', color='#111111', y=0.96)
    
    #  (，)
    fig.legend(handles=[plt.Line2D([0], [0], color=line_color, marker='o', 
                                  markerfacecolor=marker_color, markeredgecolor=marker_color, markersize=8)],
               labels=[model_name],
               frameon=False, fontsize=12, loc='upper right', bbox_to_anchor=(0.95, 0.92))

    # ---------------------------------------------------------
    # 6.  (a), (b), (c), (d) 
    # ---------------------------------------------------------
    panel_titles = [
        "（a）DevEvalTemperature",
        "（b）DevEvalTop-p",
        "（c）CoderEvalTemperature",
        "（d）CoderEvalTop-p",
    ]
    for ax, panel_title in zip(axs.flat, panel_titles):
        ax.text(
            0.5, 1.04, panel_title,
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#222222'
        )

    # ，
    # rect=[left, bottom, right, top] ， (a)(b) 
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.94], pad=4.0) 

    # ，GUI
    output_path = Path(__file__).with_name("combined_4plots_academic_style.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f": {output_path}")

    # 
    plt.show()
    plt.close(fig)

# 
if __name__ == "__main__":
    draw_combined_4plots_academic_style()
