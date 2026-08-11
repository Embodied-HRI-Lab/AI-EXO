import time  
import datetime
import numpy as np
import csv   
import copy as cp   

from scipy.signal import butter, filtfilt, find_peaks  
from scipy.interpolate import interp1d    
from scipy.signal import resample  
from scipy.io import loadmat  
  
import matplotlib.pyplot as plt  
import argparse  
import matplotlib   
import seaborn as sns          
import pandas as pd   
from pathlib import Path

# from kmp.demo_GMR import GMR_pred, KMP_pred    
# from kmp.GMRbasedGP.utils.gmr import plot_gmm, Gmr    

# matplotlib.rcParams['pdf.fonttype'] = 42    
# matplotlib.rcParams['ps.fonttype']  = 42    

# ################################# 
# plt.rcParams['font.weight']      = 'bold'   
# plt.rcParams['axes.labelweight'] = 'bold'     

# sns.set(palette="muted", font_scale=1.4, color_codes=True)     
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}  
# sns.set_style("white")   

# font_size = 15  

custom_params = {"axes.spines.right": True, "axes.spines.top": False}    
sns.set_theme(style="ticks", font_scale=1.5, rc=custom_params)    
from scipy.signal import savgol_filter     
from matplotlib.patches import Ellipse     

matplotlib.rcParams['pdf.fonttype'] = 42    
matplotlib.rcParams['ps.fonttype']  = 42    

################################# 
plt.rcParams['font.weight']      = 'bold'   
plt.rcParams['axes.labelweight'] = 'bold'   
# from motor_control import wrist_control   

font_size = 15     
angle_to_radian = np.pi/180      
radian_to_angle = 180*np.pi    

# from kmp.demo_GMR import *    
# from kmp.demo_KMP import *     

def plot_comparison_negative_positive_power(
    power_data          = None, 
    mode_list           = None,   
    method_list         = None,  
    num_gait            = 5,    
    fig_path            = '',  
    fig_name            = ''  
):  
    plt.figure(figsize=(6.5, 5))            
    axes = plt.gca()    
    
    power_comparison_pd = pd.DataFrame(columns=['Power Percentage', 'Mode', 'Method', 'Gait'])       
    for method_index, method in enumerate(method_list):    
        for mode_index, mode in enumerate(mode_list):    
            for gait_index in range(num_gait):         
                power_comparison_pd.loc[len(power_comparison_pd)] = [power_data[method][mode][gait_index], mode, method, gait_index]      

    sns.barplot(data=power_comparison_pd, x='Method', y='Power Percentage', hue='Mode')     
    
    plt.tight_layout()   
    plt.legend(ncols=len(method_list), handletextpad=0.2, handlelength=0.8, labelspacing=0.3, loc="upper center", bbox_to_anchor=(0.5, 1.12))  
    plt.ylim([0, 1]) 
    
    # plt.ylabel('Force[N]', fontsize=font_size)             
    # plt.xlabel('Characters', fontsize=font_size)       
    sns.despine()   
    
    if fig_path is not None:       
        # plt.savefig(save_path + '/' + font_name + '.png', bbox_inches='tight', pad_inches=0.0)     
        plt.savefig(fig_path + '/' + fig_name + '.pdf', bbox_inches='tight', pad_inches=0.0, dpi=500)    
        plt.savefig(fig_path + '/' + fig_name + '.png', bbox_inches='tight', pad_inches=0.0, dpi=500)      
              
    plt.show()    
    
def plot_comparison_positive_power_participants(
    power_data          = None,  
    subject_list        = None,  
    method_list         = None,  
    num_gait            = 5,    
    fig_path            = '',  
    fig_name            = ''  
):  
    plt.figure(figsize=(6.5, 5))            
    axes = plt.gca()    
    
    power_comparison_pd = pd.DataFrame(columns=['Power Percentage', 'Subject', 'Method', 'Gait'])       
    for method_index, method in enumerate(method_list):    
        for subject_index, subject in enumerate(subject_list):       
            for gait_index in range(num_gait):         
                power_comparison_pd.loc[len(power_comparison_pd)] = [power_data[subject][method]['Positive'][gait_index], subject, method, gait_index]        

    sns.barplot(data=power_comparison_pd, x='Subject', y='Power Percentage', hue='Method')     
    
    plt.tight_layout()   
    plt.legend(ncols=len(method_list), handletextpad=0.2, handlelength=0.8, labelspacing=0.3, loc="upper center", bbox_to_anchor=(0.5, 1.12))  
    plt.ylim([0, 1]) 
    
    # plt.ylabel('Force[N]', fontsize=font_size)             
    # plt.xlabel('Characters', fontsize=font_size)       
    sns.despine()   
    
    if fig_path is not None:       
        # plt.savefig(save_path + '/' + font_name + '.png', bbox_inches='tight', pad_inches=0.0)     
        plt.savefig(fig_path + '/' + fig_name + '.pdf', bbox_inches='tight', pad_inches=0.0, dpi=500)    
        plt.savefig(fig_path + '/' + fig_name + '.png', bbox_inches='tight', pad_inches=0.0, dpi=500)      
              
    plt.show()    

def lower_filter(pre_vec=None, current_vec=None, alpha=0.05):    
    return (1 - alpha) * pre_vec + alpha * current_vec     

def compute_gait_average_profile(input_data):    
    gait_head_list         = input_data['gaitHeadList']    
    gait_tail_list         = input_data['gaitTailList']    
    data_seq               = input_data['dataSeq']    
    data_out_names         = input_data['dataOutNames']    
    data_shift             = input_data['dataShift']   
    normalized_gait_length = input_data['normalizedGaitLength']     
    
    # Assertions
    assert len(data_seq) == len(data_out_names), 'Error: dataSeq and dataOutNames need to have the same length!'

    # Initialize output    
    result = {}   

    # Compute   
    for idx in range(len(gait_head_list)):  
        gait_head = gait_head_list[idx]   
        gait_tail = gait_tail_list[idx]   
        
        for data_idx in range(len(data_seq)):   
            data = data_seq[data_idx]
            data_out_name = data_out_names[data_idx]
            gait_data = data[gait_head:gait_tail]
            # print("gait data :", gait_data.shape) 
            
            # Normalize
            x = np.arange(1, len(gait_data) + 1)
            interp_func = interp1d(x, gait_data, kind='linear', fill_value='extrapolate')
            gait_data_normalized = interp_func(np.linspace(1, len(gait_data), normalized_gait_length))

            # Compensate for the added delay
            gait_data_normalized = np.roll(gait_data_normalized, int(round(data_shift)))

            print("gait data normalized :", gait_data_normalized.shape)         
            # # Gather result
            # if f'gait{data_out_name}NormalizedList' not in result:  
            #     result[f'gait{data_out_name}NormalizedList'] = gait_data_normalized[np.newaxis, :]
            # else:
            #     result[f'gait{data_out_name}NormalizedList'] = np.vstack(
            #         [result[f'gait{data_out_name}NormalizedList'], gait_data_normalized[np.newaxis, :]]
            #     )

    return result

def compute_head_tail_list(
    hip_angle         = None,
    fc                = None, 
    fs                = None, 
    min_peak_height   = None, 
    min_gait_duration = None,
    max_gait_duration = None, 
    num_cycles        = 5    
):  
    b, a = butter(2, fc / (fs / 2), btype='low')     
    hip_angle_filtered = filtfilt(b, a, hip_angle)       
    
    gait_head_list = []     
    gait_tail_list = []     
    peaks, _ = find_peaks(hip_angle_filtered, height=min_peak_height, distance=min_gait_duration)   
    
    if peaks.size > 0: 
        for gait_idx in range(num_cycles):    
            gait_head = peaks[gait_idx]   
            gait_tail = peaks[gait_idx + 1]     
            if gait_tail - gait_head <= max_gait_duration:   
                gait_head_list.append(gait_head)
                gait_tail_list.append(gait_tail)    
                
        return gait_head_list, gait_tail_list   
    else:
        print("No peaks found in the hip angle data.")
        return [], []  

def compute_gait_average_profile(input_data):    
    gait_head_list         = input_data['gaitHeadList']    
    gait_tail_list         = input_data['gaitTailList']    
    data                   = input_data['data']     
    normalized_gait_length = input_data['normalizedGaitLength']     
    result = np.zeros((len(gait_head_list), normalized_gait_length))       
    for idx in range(len(gait_head_list)):  
        gait_head = gait_head_list[idx]   
        gait_tail = gait_tail_list[idx]     
        gait_data = data[gait_head:gait_tail]    
        
        # Normalize
        x = np.arange(1, len(gait_data) + 1)  
        
        # print(x.shape, gait_data.shape)  
        interp_func = interp1d(x, gait_data, kind='linear', fill_value='extrapolate')
        
        gait_data_normalized = interp_func(np.linspace(1, len(gait_data), normalized_gait_length))

        # Compensate for the added delay
        gait_data_normalized = np.roll(gait_data_normalized, 0)   

        result[idx, :] = gait_data_normalized[np.newaxis, :]  
    
    print("results :", result.shape)  
    avg_result = np.mean(result, axis=0)  # Average across all cycles
    std_result = np.std(result, axis=0)  # Standard deviation across all cycles
    return avg_result, std_result, result    

def load_gait_data(
        csv_file, 
        num_gait_cycles=6, 
        sample_rate=100, 
        start_index=0,
        end_index=5000  
    ): 
    # with open(csv_file, 'r') as f:  
    #     reader = csv.reader(f)  
    #     data = [list(map(float, row)) for row in reader if row]
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)[start_index:end_index, :]  # Load data from CSV file  
    print("data shape :", data.shape)  
    
    left_gait_head_list, left_gait_tail_list = compute_head_tail_list(
        hip_angle         = data[:, 1],
        fc                = 5,  
        fs                = 100,   
        min_peak_height   = 0, 
        min_gait_duration = 60,
        max_gait_duration = 2000,   
        num_cycles        = num_gait_cycles      
    )   
    print("left :", left_gait_head_list, left_gait_tail_list)   
    print("avg length :", np.mean(np.array(left_gait_tail_list) - np.array(left_gait_head_list)))
    
    right_gait_head_list, right_gait_tail_list = compute_head_tail_list( 
        hip_angle         = data[:, 2],  
        fc                = 5,  
        fs                = 100,   
        min_peak_height   = 0, 
        min_gait_duration = 60,
        max_gait_duration = 2000,   
        num_cycles        = num_gait_cycles      
    )
    print("right :", right_gait_head_list, right_gait_tail_list)  
    
    return np.array(data), left_gait_head_list, left_gait_tail_list, right_gait_head_list, right_gait_tail_list 

def segement_gait_data(data, num_gait=5): 
    segement_data = [] 
    
    return np.mean(segement_data, axis=0)  

def save_mean_gait_data(mean_data, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(mean_data.tolist())  

def process_gait_data(input_csv, output_csv):
    data = load_gait_data(input_csv)  
    mean_data = np.mean(data, axis=0)    
    save_mean_gait_data(mean_data, output_csv)    

def plot_comparison_results(
    time_list=None, 
    data_actual=None,   
    data_reference=None,   
    left_gait_head_list=None, 
    left_gait_tail_list=None,
    right_gait_head_list=None,
    right_gait_tail_list=None,  
    label_list=['Actual', 'Reference'], 
    save_path='comparison_of_' 
):  
    fig, axs = plt.subplots(1, 2, figsize=(15, 3.5))     
    
    axs[0].plot(time_list, data_actual[:, 0], label=label_list[0], color='blue')  
    axs[0].plot(time_list, data_reference[:, 0], label=label_list[1], color='black')    
    axs[0].scatter(
        time_list[left_gait_head_list], 
        data_actual[left_gait_head_list, 0], 
        color='red', label='Left Gait Head'
    )   
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5)  
    axs[0].set_xlabel('Time (s)')   
    axs[0].set_ylabel('Left Hip')     
    
    axs[1].plot(time_list, data_actual[:, 1], label=label_list[0], color='blue')    
    axs[1].plot(time_list, data_reference[:, 1], label=label_list[1], color='black')     
    axs[1].scatter(
        time_list[right_gait_head_list], 
        data_actual[right_gait_head_list, 0], 
        color='red', label='Left Gait Head'
    )   
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=5)   
    axs[1].set_xlabel('Time (s)')   
    axs[1].set_ylabel('Right Hip')      
     
    plt.tight_layout()   
    plt.savefig(save_path)    
    plt.show()   


def plot_exoskeleton_log(csv_path, output_path=None, start_time=None,
                         end_time=None, show=False):
    """Plot bilateral angle, angular velocity, and torque from an exo log."""
    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path, encoding="utf-8-sig")

    required = [
        "elapsed_s",
        "left_angle_x_deg", "right_angle_x_deg",
        "left_angular_velocity_x_dps", "right_angular_velocity_x_dps",
        "left_actual_torque_nm", "right_actual_torque_nm",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    if start_time is not None:
        data = data[data["elapsed_s"] >= start_time]
    if end_time is not None:
        data = data[data["elapsed_s"] <= end_time]
    if data.empty:
        raise ValueError("No samples remain in the requested time interval.")

    time_s = data["elapsed_s"].to_numpy()
    signal_rows = [
        ("left_angle_x_deg", "right_angle_x_deg", "Hip angle (deg)"),
        ("left_angular_velocity_x_dps", "right_angular_velocity_x_dps",
         "Angular velocity (deg/s)"),
        ("left_actual_torque_nm", "right_actual_torque_nm", "Torque (Nm)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    for axis, (left_column, right_column, ylabel) in zip(axes, signal_rows):
        axis.plot(time_s, data[left_column], color="#0072B2", linewidth=1.25,
                  label="Left")
        axis.plot(time_s, data[right_column], color="#D55E00", linewidth=1.25,
                  label="Right")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25, linewidth=0.6)

    axes[0].legend(loc="upper right", ncols=2, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(csv_path.stem, fontweight="bold")
    fig.tight_layout()

    if output_path is None:
        output_path = csv_path.with_name(f"{csv_path.stem}_plot.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
    
def plot_comparison_kinematics_kinetic_results(
    time_list=None, 
    data_actual=None,   
    data_reference=None,   
    data_velocity=None, 
    data_force_ours=None,   
    data_force_baseline=None, 
    data_power_ours=None, 
    data_power_baseline=None,  
    left_gait_head_list=None, 
    left_gait_tail_list=None,
    right_gait_head_list=None,
    right_gait_tail_list=None,  
    label_list=['Actual', 'Reference'], 
    save_path='comparison_of_' 
):  
    blue_colors = ['#ADD8E6', '#87CEEB', '#4682B4', '#4169E1', '#0000CD', '#00008B']
    # fig, axs = plt.subplots(1, 2, figsize=(15, 4))     
    fig, axs = plt.subplots(2, 1, figsize=(8, 7))    
    
    axs[0].plot(time_list, data_actual[:, 0], label=label_list[0], color='black')  
    # axs[0].plot(time_list, data_reference[:, 0], label=label_list[1], color='black')    
    # axs[0].scatter(
    #     time_list[left_gait_head_list], 
    #     data_actual[left_gait_head_list, 0], 
    #     color='red'
    # )   
    
    axs_0_mirror = axs[0].twinx()     
    if data_velocity is not None:  
        # axs_0_mirror.plot(time_list, data_velocity[:, 0], color='black', linestyle='--')    
        axs[0].plot(time_list, data_velocity[:, 0], label=label_list[1], color='black', linestyle='--')       
    
    axs[0].legend(
        loc='upper left', bbox_to_anchor=(0.0, 1.22), 
        handlelength=0.7, framealpha=0.8, columnspacing=0.5, handletextpad=0.4, 
        ncol=5, frameon=False
    )  
    axs[0].set_xlabel('Time (s)\n (a)')     
    axs[0].set_ylabel('Left Hip Joint')      
    
    if data_force_ours is not None:     
        axs_0_mirror.plot(time_list, data_force_ours[:, 0], label='Ours', color=blue_colors[1])     
        axs_0_mirror.plot(time_list, data_force_baseline[:, 0], label='Samsung', color=blue_colors[4])         
        axs_0_mirror.legend(loc='upper center', bbox_to_anchor=(0.8, 1.22), ncol=5, frameon=False)     
        axs_0_mirror.set_ylabel('Assistive Torque(Nm)', color='green')   
        
    if data_power_ours is not None:    
        axs_0_mirror.plot(time_list, data_power_ours[:, 0], label='Ours', color='green')     
        axs_0_mirror.plot(time_list, data_power_baseline[:, 0], label='Samsung', color='cyan')       
        axs_0_mirror.set_ylabel('Assistive Power')      
        
    axs_0_mirror.legend(
        loc='upper right', bbox_to_anchor=(1.0, 1.22), 
        handlelength=0.7, framealpha=0.8, columnspacing=0.5, handletextpad=0.4, 
        ncol=5, frameon=False
    )  
    
    axs[1].plot(time_list, data_actual[:, 1], label=label_list[0], color='black')    
    # axs[1].plot(time_list, data_reference[:, 1], label=label_list[1], color='black')     
    # axs[1].scatter(
    #     time_list[right_gait_head_list], 
    #     data_actual[right_gait_head_list, 0], 
    #     color='red' 
    # )   
    
    axs_1_mirror = axs[1].twinx()     
    if data_velocity is not None:    
        # axs_1_mirror.plot(time_list, data_velocity[:, 1], color='black', linestyle='--')     
        axs[1].plot(time_list, data_velocity[:, 1], label=label_list[1], color='black', linestyle='--')      
    
    axs[1].legend(
        loc='upper left', bbox_to_anchor=(0.0, 1.22), 
        handlelength=0.7, framealpha=0.8, columnspacing=0.5, handletextpad=0.4, 
        ncol=5, frameon=False
    )   
    axs[1].set_xlabel('Time (s)\n (b)')   
    axs[1].set_ylabel('Right Hip Joint')     
      
    if data_force_ours is not None:    
        axs_1_mirror.plot(time_list, data_force_ours[:, 1]*-1, label='Ours', color=blue_colors[1])     
        axs_1_mirror.plot(time_list, data_force_baseline[:, 1], label='Samsung', color=blue_colors[4])           
        axs_1_mirror.legend(loc='upper center', bbox_to_anchor=(0.8, 1.22), ncol=5, frameon=False)      
        axs_1_mirror.set_ylabel('Assistive Torque(Nm)', color='green')   
    
    if data_power_ours is not None:   
        # axs_1_mirror = axs[1].twinx()     
        axs_1_mirror.plot(time_list, data_power_ours[:, 1], label='Ours', color='green')     
        axs_1_mirror.plot(time_list, data_power_baseline[:, 1], label='Samsung', color='cyan')      
        axs_1_mirror.set_ylabel('Assistive Power')      
        
    axs_1_mirror.legend(
        loc='upper right', bbox_to_anchor=(1.0, 1.22), 
        handlelength=0.7, framealpha=0.8, columnspacing=0.5, handletextpad=0.4,
        ncol=5, frameon=False
    )  
    
    plt.tight_layout()    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=500)     
    plt.show()    
    
def plot_two_lines(
    time_list=None, 
    data_actual=None,     
    data_reference=None,     
    data_force=None, 
    label_list=['Actual', 'Reference'], 
    save_path='comparison_of_' 
):  
    fig, axs = plt.subplots(1, 2, figsize=(15, 3.5))    
    
    if time_list is not None:  
        axs[0].plot(time_list, data_reference[:, 0], label=label_list[0], linewidth=5, color='blue', alpha=0.2)   
        axs[0].plot(time_list, data_actual[:, 0], label=label_list[1], color='black')      
        # axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2)  
        axs[0].set_xlabel('Time (s)')   
        axs[0].set_ylabel('Hip Joint Position (deg)')       
        
        if data_force is not None:   
            axs_0_mirror = axs[0].twinx()   
            axs_0_mirror.plot(time_list, -1 * data_force[:, 0], label=label_list[1], linestyle='--', color='red')   
            axs_0_mirror.set_ylabel('Assistive Torque (Nm)')         
        
        axs[1].plot(time_list, data_reference[:, 1], label=label_list[0], linewidth=5, color='blue', alpha=0.2)    
        axs[1].plot(time_list, data_actual[:, 1], label=label_list[1], color='black')     
        # axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2)   
        axs[1].set_xlabel('Time (s)')   
        axs[1].set_ylabel('Hip Joint Postion (deg)')       
        
        if data_force is not None:   
            axs_1_mirror = axs[1].twinx()   
            axs_1_mirror.plot(time_list, data_force[:, 1], label=label_list[1], linestyle='--', color='red')   
            axs_1_mirror.set_ylabel('Assistive Torque (Nm)')        
    else:     
        axs[0].plot(data_reference[:, 0], label=label_list[0], linewidth=5, color='blue', alpha=0.2)    
        axs[0].plot(data_actual[:, 0], label=label_list[1], color='black')    
        # axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2)   
        axs[0].set_xlabel('Time (s)\n (a) Left')     
        axs[0].set_ylabel('Hip Joint Postion (deg)')         
        
        if data_force is not None:   
            axs_0_mirror = axs[0].twinx()    
            axs_0_mirror.plot(-1 * data_force[:, 0], label=label_list[1], color='green')  
            axs_0_mirror.set_ylabel('Assistive Torque (Nm)', color='green')          
        
        axs[1].plot(data_reference[:, 1], label=label_list[0], linewidth=5, color='blue', alpha=0.2)      
        axs[1].plot(data_actual[:, 1], label=label_list[1], color='black')     
        # axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2)   
        axs[1].set_xlabel('Time (s)\n (b) Right')    
        axs[1].set_ylabel('Hip Joint Postion (deg)')     
        
        if data_force is not None:   
            axs_1_mirror = axs[1].twinx()     
            axs_1_mirror.plot(data_force[:, 1], label=label_list[1], color='green')     
            axs_1_mirror.set_ylabel('Assistive Torque (Nm)', color='green')        
     
    plt.tight_layout()   
    # plt.savefig(save_path)      
    plt.show()  

def plot_gait_mean_std(
    gait_cycle_list= None,
    gait_data_mean = None,   
    gait_data_std  = None,   
    save_path      = 'comparison_of_' 
):  
    fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))         
    
    axs.fill_between(
        gait_cycle_list, 
        gait_data_mean - gait_data_std, 
        gait_data_mean + gait_data_std, 
        color='blue',    
        alpha=0.1  
    )    
    axs.plot(gait_cycle_list, gait_data_mean, linewidth=3, color='black', label="Actual Trajectory")         
    axs.set_xlabel('Normalized Gait Cycle (%)')     
    axs.set_ylabel('Hip Joint Position (deg)')     
    
    plt.legend(ncols=2, handletextpad=0.2, handlelength=0.8, labelspacing=0.1, loc="upper center", bbox_to_anchor=(0.5, 1.20), fontsize=15) 
    plt.tight_layout()    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=500)    
    plt.show()     

def plot_gait_actual_target_trajectory(
    nb_samples        = None, 
    Xt                = None, 
    Y                 = None, 
    nb_data           = None, 
    gait_cycle_list   = None,
    gait_data_actual  = None,   
    gait_data_target  = None,   
    scatter_list      = None, 
    save_path         = 'comparison_of_' 
):  
    fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))         
    
    # axs.fill_between(
    #     gait_cycle_list, 
    #     gait_data_mean - gait_data_std, 
    #     gait_data_mean + gait_data_std, 
    #     color='blue',    
    #     alpha=0.1  
    # )    
    for p in range(nb_samples):    
        plt.plot(Xt, Y[p*nb_data:(p+1)*nb_data, 0], color=[.55, .55, .55])      
        
    axs.plot(gait_cycle_list, gait_data_actual, linewidth=3, 
             color='black', label="Average Trajectory")     
    
    # axs.plot(gait_cycle_list, gait_data_target, linewidth=2, 
    #          color='blue', label="Reference Trajectory")   
       
    # axs.scatter(gait_cycle_list[scatter_list], gait_data_target[scatter_list], color='red')  
    # axs.scatter(gait_cycle_list[scatter_list], gait_data_actual[scatter_list], color='red')  
     
    axs.set_xlabel('Normalized Gait Cycle (%)')      
    axs.set_ylabel('Hip Joint Position (deg)')     
    
    plt.legend(
        ncols=2, loc="upper center", 
        # handletextpad=0.2, 
        # handlelength=1.1, labelspacing=0.1, 
        bbox_to_anchor=(0.5, 1.20), 
        fontsize=15, frameon=False
    )   
    plt.tight_layout()    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=500)    
    plt.show()      

def obtain_gmr_mean_variance(args=None, input_data=None):   
    nb_data    = args.nb_data   
    nb_samples = args.nb_samples      
    
    # demos = data.reshape((nb_data, nb_samples, 3))  
    demos = input_data     
    # print("input data shape :", demos.shape)     

    nb_data_sup = 0      
    dt = 0.01     
    demodura = dt * nb_data      
    # print("demodura :", demodura)     
    
    # model parameter 
    input_dim  = 1   
    output_dim = 1    
    # output_dim = 3    
    
    # Create time data     
    demos_t = [np.arange(nb_data)[:, None] for i in range(nb_samples)]     
    print("demos_t :", np.array(demos_t[0]).shape)     
    
    # Stack time and position data  
    demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    #  demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    print("demos_tx :", np.array(demos_tx).shape)    

    # Stack demos    
    demos_np = demos_tx[0]   
    print("demos_np :", demos_np.shape)    

    for i in range(1, nb_samples):     
        demos_np = np.vstack([demos_np, demos_tx[i]])     
    print("demos_np :", demos_np.shape)      
    
    X = demos_np[:, 0][:, None]    
    Y = demos_np[:, 1:]     
    print('X shape: ', X.shape, 'Y shape: ', Y.shape)    

    # Test data   
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]   
    # print("Xt :", Xt)  
    mu_gmr, sigma_gmr, gmr_model = GMR_pred(
        demos_np=demos_np,    
        X=X,   
        Xt=Xt,  
        Y=Y,    
        nb_data=args.nb_data,    
        nb_samples=args.nb_samples,     
        nb_states=args.nb_states,     
        input_dim=input_dim,    
        output_dim=output_dim
    )   
     
    # new via points  
    via_index    = [0, 100]   
    via_num      = len(via_index)   
    via_time     = dt * np.array(via_index)      
    via_flag     = np.ones(via_num)    
    via_points   = 1.8 * Y[via_index, :]        
    via_var_list = [0.0001, 0.0001]      
    ori_refTraj, mu_kmp, kmp_traj = KMP_pred(
        Xt=Xt,   
        mu_gmr=mu_gmr,   
        sigma_gmr=sigma_gmr,        
        viaNum=via_num,    
        viaFlag=via_flag,        
        via_time=via_time,         
        via_points=via_points,     
        via_var_list=via_var_list,    
        dt=0.01,    
        len=None,     
        lamda_1=0.01,      
        lamda_2=0.6,       
        kh=6, 
        output_dim=1,     
        dim=1      
    )  
    return mu_gmr, sigma_gmr, gmr_model, mu_kmp, kmp_traj     

def plot_path_mean_var(
    nb_samples=5, 
    nb_data   =200, 
    Xt        =None, 
    Y         =None, 
    mu_gmr    =None, 
    sigma_gmr =None, 
    save_path =None, 
    save_fig  =False
): 
    plt.figure(figsize=(6.5, 4))    
    
    for p in range(nb_samples):    
        plt.plot(Xt, Y[p*nb_data:(p+1)*nb_data, 0], color=[.55, .55, .55])      
        # plt.scatter(Xt, Y[p*nb_data, 0], color=[.55, .55, .55], marker='X', s=80)     
    
    # print("mu_gmr :", mu_gmr[:, :2].shape, "sigma_gmr :", sigma_gmr[:, :2, :2].shape)   
    plt.plot(Xt, mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3)    
    # plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)    
    
    plt.fill_between(
        Xt, 
        mu_gmr[:, 0] - sigma_gmr[:, 0, 0],  
        mu_gmr[:, 0] + sigma_gmr[:, 0, 0],   
        color='blue',    
        alpha=0.2  
    )   
    
    plt.xlabel("Normalized Gait Cycle (%)", fontsize=font_size)  
    plt.ylabel("Hip Joint Position (deg)", fontsize=font_size)   
    
    plt.legend(
        ncols=2, loc="upper center", 
        # handletextpad=0.2, 
        # handlelength=1.1, labelspacing=0.1, 
        bbox_to_anchor=(0.5, 1.20), 
        fontsize=font_size, 
        frameon=False
    )   
    plt.locator_params(nbins=3)       
    plt.tick_params(labelsize=font_size)         
    plt.tight_layout()      
    
    if save_fig:   
        plt.tight_layout()    
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=500)    
           
    plt.show()   
    
def obtain_samsung_control_force(
        data_angle    = None, 
        data_velocity = None,  
        K             = 20, 
        delay         = 30, 
        gait_length   = 150  
    ):  
    force_list = []  
    power_list = []  
    index      = 0
    
    delta_theta_vector = np.zeros(gait_length)    
    torque_vector      = np.zeros(gait_length)  
    
    left_angle  = data_angle[0, 0]      
    right_angle = data_angle[0, 1]     
    for index in range(data_angle.shape[0]):       
        current_index = index 
        index = index % gait_length     
        left_angle  = lower_filter(left_angle, data_angle[index, 0], alpha=0.05)  
        right_angle = lower_filter(right_angle, data_angle[index, 1], alpha=0.05)    
        
        # delta_theta = data[index, 1] - data[index, 0]  
        # torque      = K * (np.sin(data[index, 1]*np.pi/180.0) - np.sin(data[index, 0])*np.pi/180.0)  
        
        delta_theta = right_angle - left_angle  
        torque      = K * (np.sin(right_angle*np.pi/180.0) - np.sin(left_angle*np.pi/180.0))    
        # print("index :", index, ", torque :", torque)  
          
        # left_angle_old  = left_angle
        # right_angle_old = right_angle
        delta_theta_vector[index] = delta_theta  
        torque_vector[index]      = torque  
        
        delay_index = index - delay  
        if delay_index < 0:  
            delay_index = delay_index + gait_length 
        elif delay_index >= gait_length: 
            delay_index = delay_index - gait_length   
        else:  
            pass     
            
        if delta_theta_vector[delay_index] >= 0 and delta_theta_vector[delay_index] < 120: 
            left_force  = -1 * torque_vector[delay_index]    
            right_force = torque_vector[delay_index]    
            
            left_power  = left_force * data_velocity[current_index, 0]    
            right_power = right_force * data_velocity[current_index, 1]       
        elif delta_theta_vector[delay_index] < 0 and delta_theta_vector[delay_index] > -120: 
            left_force  = -1 * torque_vector[delay_index]  
            right_force = torque_vector[delay_index]  
            
            left_power  = left_force * data_velocity[current_index, 0]     
            right_power = right_force * data_velocity[current_index, 1]           
        else: 
            left_force  = 0.0 
            right_force = 0.0 
            
            left_power  = 0.0   
            right_power = 0.0   
            
        force_list.append([left_force, right_force])    
        power_list.append([left_power, right_power])   
        index += 1  
        
    return np.array(force_list), np.array(power_list)    

def obtain_our_assistive_power(
    data_force   =None, 
    data_velocity=None, 
    left_ratio   = 1, 
    right_ratio  = -1
):  
    power_list = []  
    for index in range(data_force.shape[0]):  
        left_power  = data_velocity[index, 0] * data_force[index, 0] * left_ratio  
        right_power = data_velocity[index, 1] * data_force[index, 1] * right_ratio 
        
        power_list.append([left_power, right_power])    
    return np.array(power_list)  

def calculate_positive_negative_perception(
    gait_vector_list= None, 
    num_gait        = None
):  
    percentage = np.zeros((2, num_gait))  
    for gait_index in range(num_gait):   
        gait_vector = gait_vector_list["gait_"+str(gait_index)]   
        length      = gait_vector.shape[0]   
        num_positive = 0  
        num_negative = 0   
        for index in range(length): 
            if gait_vector[index] > 0: 
                num_positive += 1
            else: 
                num_negative += 1 
        percentage[:, gait_index] = np.array([num_positive/length, num_negative/length])
    return percentage  
    
# tele-operation/tele-rehabilitation tasks 
def paper_ral_figure_7(  
    data_path   = None,  
    start_index = 0,  
    end_index   = 300,         
 	save_fig    = False,        
 	save_root   = '', 
    args        = None       
):  
    data = np.loadtxt(data_path,  delimiter=',', skiprows=1) 
    fig = plt.figure(figsize=(10, 4))  
    
    ax_1 = plt.subplot(1, 1, 1)       

    t_list = data[:, 0]    
   
    ax_1.plot(t_list, data[:, 3], label=r"$q_d^s$", linewidth=2.5, color='blue')   
    ax_1.plot(t_list, data[:, 4], label=r"$q_t^s$", linewidth=2.5, color='black')   
    
    ax_1.annotate(
        'Peak', 
        xy=(4, -0.2),         # 箭头指向的点 (x, y)
        xytext=(5, -0.1),           # 文本放置的位置
        arrowprops=dict(arrowstyle='->', color='red'),  # 箭头样式
        fontsize=12, color='red'
    )   
    ax_1.set_ylabel(r"Joint Angle $[\circ]$", color='black')     
    ax_1.set_xlabel("Time[s]")  
    # ax_1.set_xlim(-10, 50)  
    
    # ax_1.legend(loc="upper left", handlelength=1, framealpha=0.8, ncol=len(label_list))     
    ax_1.legend(loc="upper left", 
                bbox_to_anchor=(0.0, 1.2), handlelength=0.7, framealpha=0.8, ncol=2, columnspacing=0.5, handletextpad=0.4,
                frameon=False
            )              
    # ax_1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(label_list))              
    
    ax_2 = ax_1.twinx()      
    ax_2.plot(t_list, data[:, 5], label=r"$\tau_{q,d}^m$", linewidth=2.5, color='red')    
    ax_2.plot(t_list, data[:, 6], label=r"$\tau_{q,t}^m$", linewidth=2.5, color='green')    
    
    ax_2.set_ylabel(r"Interaction Force[Nm]", color='green')          
    ax_2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.2), 
                handlelength=0.7, framealpha=0.8, ncol=2, columnspacing=0.5, handletextpad=0.4, 
                frameon=False
            )               
     
    plt.tight_layout()       
 
    if save_fig:     
        print("save figure :", args.fig_path + '/' + args.fig_name)     
        plt.savefig(args.fig_path + '/' + args.fig_name + '.pdf', bbox_inches='tight', dpi=500, pad_inches=0.0)       
        # plt.savefig('./figures/' + flag + '.png', bbox_inches='tight', pad_inches=0.0)      
        # plt.savefig(save_root + '/angle_torque' + flag + '.png',bbox_inches='tight',pad_inches=0.0)      
         
    plt.show()    

def paper_ral_figure_8(  
    data_path   = None,  
    start_index = 0,  
    end_index   = 300,         
 	save_fig    = False,        
 	save_root   = '', 
    args        = None       
):  
    data = np.loadtxt(data_path,  delimiter=',', skiprows=1)   
    
    fig = plt.figure(figsize=(10, 4))  

    ax_1 = plt.subplot(1, 1, 1)    
    t_list = data[:, 0]    
   
    ax_1.plot(t_list, data[:, 3], label=r"$q_d^s$", linewidth=4.0, color='blue', alpha=0.1)   
    ax_1.plot(t_list, data[:, 4], label=r"$q_t^m$", linewidth=2.5, color='black')   
    
    ax_1.set_ylabel(r"Joint Angle $[\circ]$", color='black')     
    ax_1.set_xlabel("Time[s]")  
    # ax_1.set_xlim(-10, 50)  
     
    ax_1.legend(
        loc="upper left", 
        bbox_to_anchor=(0.0, 1.2), handlelength=0.7, framealpha=0.8, ncol=2, 
        columnspacing=0.8, handletextpad=0.8,
        frameon=False
    )            
    
    ax_2 = ax_1.twinx()      
    ax_2.plot(t_list, data[:, 5], label=r"$\tau_{q,d}^m$", linewidth=2.5, color='red')    
    ax_2.plot(t_list, data[:, 6], label=r"$\tau_{q,t}^s$", linewidth=2.5, color='green')    

    ax_2.set_ylabel(r"Interaction Force[Nm]", color='black')          
    ax_2.legend(
        loc="upper right", 
        bbox_to_anchor=(1.0, 1.2), handlelength=0.7, framealpha=0.8, ncol=2, 
        columnspacing=0.8, handletextpad=0.8, 
        frameon=False
    )               
     
    plt.tight_layout()       
 
    if save_fig:     
        print("save figure :", args.fig_path + '/' + args.fig_name)     
        plt.savefig(args.fig_path + '/' + args.fig_name + '.pdf', bbox_inches='tight', dpi=500, pad_inches=0.0)       
        # plt.savefig('./figures/' + flag + '.png', bbox_inches='tight', pad_inches=0.0)      
        # plt.savefig(save_root + '/angle_torque' + flag + '.png',bbox_inches='tight',pad_inches=0.0)      
         
    plt.show()   
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()   
    parser.add_argument(
        "csv", nargs="?", default=str(Path(__file__).parent / "exo_logs/yth10.csv"),
        help="Exoskeleton CSV log to plot (default: exo_logs/yth10.csv)",
    )
    parser.add_argument("--output", help="Output image path (default: beside the CSV)")
    parser.add_argument("--start", type=float, help="First time to include, in seconds")
    parser.add_argument("--end", type=float, help="Last time to include, in seconds")
    parser.add_argument("--show", action="store_true", help="Open the plot window")
    
    parser.add_argument('--save_fig', type=bool, default=True, help='choose index first !!!!')  
    parser.add_argument('--data_name', type=str, default="tracking_epi_circ", help='data name!!!!')  
    parser.add_argument('--file_path', type=str, default='wrist', help='choose index first !!!!')  
    parser.add_argument('--file_name', type=str, default='wrist', help='choose index first !!!!')  
    parser.add_argument('--root_path', type=str, default='data/', help='choose index first !!!!')   
    parser.add_argument('--fig_path', type=str, default='wrist', help='choose index first !!!!')  
    parser.add_argument('--fig_name', type=str, default='wrist', help='choose index first !!!!')  

    parser.add_argument('--nb_samples', type=int, default=5, help='choose mode first !!!!')  
    parser.add_argument('--nb_states', type=int, default=15, help='choose mode first !!!!')   
    parser.add_argument('--nb_data', type=int, default=101, help='choose mode first !!!!')  

    args = parser.parse_args()    
    plot_exoskeleton_log(
        args.csv,
        output_path=args.output,
        start_time=args.start,
        end_time=args.end,
        show=args.show,
    )
    raise SystemExit(0)
    
    
    # ##########################################################################
    
    # paper_ral_figure_7(  
    #     data_path   = '../GUI/0-RL/0-data/therapist_in_the_loop_training.csv',  
    #     start_index = 0,  
    #     end_index   = 300,         
    #     save_fig    = True,        
    #     save_root   = '', 
    #     args        = args       
    # )  
    
    # paper_ral_figure_8(  
    #     data_path   = '../GUI/0-RL/0-data/therapist_in_the_loop_training.csv',  
    #     start_index = 200,  
    #     end_index   = 400,         
    #     save_fig    = True,        
    #     save_root   = '', 
    #     args        = args       
    # )   

    # def scaled_sigmoid_with_deadzone(x, deadzone=0.5, slope=10):   
    #     y = np.zeros_like(x)
        
    #     # Left side
    #     left_mask = x < -deadzone  
    #     y[left_mask] = 2 / (1 + np.exp(-slope * (x[left_mask] + 0.5))) - 1
        
    #     # Right side
    #     right_mask = x > deadzone
    #     y[right_mask] = 2 / (1 + np.exp(-slope * (x[right_mask] - 0.5))) - 1
    #     return y  

    # # Input range
    # x_vals = np.linspace(-5, 5, 500)
    # y_vals = scaled_sigmoid_with_deadzone(x_vals, deadzone=0.5, slope=1.5)

    # # Plot
    # plt.plot(x_vals, y_vals, label='Scaled Sigmoid with Dead Zone')
    # plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # plt.axvline(-0.5, color='red', linestyle='--', linewidth=0.8, label='Dead Zone Boundaries')
    # plt.axvline(0.5, color='red', linestyle='--', linewidth=0.8)
    # plt.title('Sigmoid Function with Dead Zone [-0.5, 0.5]')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.grid(True)
    # plt.legend()
    # plt.show()  
    
    # ##########################################################################
    
    
    # ##########################################################################
    # data_path = 'RefTrajFouier125.mat'  
    # data = loadmat(data_path)  
    # print(data.keys())  
    # print("data shape :", data['E'].shape, data['F'].shape)    
    # print(data['E'])    
    # print(data['F'])    
    
    # data_path = 'SimGain.mat'  
    # data = loadmat(data_path)  
    # print(data.keys())  
    # print("data shape :", data['Kstar'], data['Ustar'], data['Xstar'].reshape(1, -1))  
       
    # print(data['E'])    
    # print(data['F'])    
    
    # n = 5 
    # omega = 1.21748757
    # v = [1, np.sin(omega )]    
    
    # #########################################################################
    # # input_csv = 'random_input_nyu.csv'  # Replace with your input CSV file
    # # input_csv = './IEEE_RAL_NYU/second_time_experiment_slow_2.csv'  
    # input_csv = './0-data/0-NYU-Zhemin/experiment_speed_175.csv'   
    # # # output_csv = 'random_input_nyu.csv'     # Replace with your desired output CSV file   
    
    # input_csv   = '0-data/0-NYU-Zhemin/experiment_speed_175.csv'
    # start_index = 2500   
    # end_index   = 4500  
    
    # input_csv     = './0-data/0-NYU-Zhemin/validation_experiment_speed_075.csv'   
    # start_index   = 5000   
    # end_index     = 6000    
    
    # # input_csv   = './0-data/0-NYU-Zhemin/validation_experiment_speed_125.csv'   
    # # start_index = 4000   
    # # end_index   = 5000    
      
    # # input_csv   = './0-data/0-NYU-Zhemin/validation_experiment_speed_175.csv'           
    # # start_index = 4500   
    # # end_index   = 5000     
    
    # ori_data = np.loadtxt(input_csv, delimiter=',', skiprows=1)
    # print("data shape :", ori_data.shape)   
    # data = ori_data[start_index:end_index, :]  # Load data from CSV file   
    
    # time_list      = data[:, 0]  
    # data_actual    = data[:, 1:3]  
    # data_reference = data_actual    
    # data_velocity  = data[:, 3:5] 
    # data_force     = data[:, 5:7]    
    
    # data_power_ours   = obtain_our_assistive_power(
    #     data_force    = data_force, 
    #     data_velocity = data_velocity 
    # )   
    # data_force_samsung, data_power_samsung = obtain_samsung_control_force(
    #     data_angle    = data_actual,  
    #     data_velocity = data_velocity,  
    #     K             = 10, 
    #     delay         = 60, 
    #     gait_length   = 160  
    # )   
    
    # plot_two_lines(
    #     time_list=None,  
    #     data_actual=data,    
    #     data_reference=data,     
    #     data_force=None,  
    #     label_list=['Actual', 'Reference'],  
    #     save_path='./1-figure/validation_speed_175.pdf'   
    # )   
    
    # plot_two_lines(
    #     time_list     =None,  
    #     data_actual   =data_actual,     
    #     data_reference=data_actual,       
    #     data_force    =None,  
    #     label_list=['Actual', 'Reference'],  
    #     save_path='./1-figure/validation_speed_175.png'   
    # )   
    
    ############################ Reference Data  #############################
    ##########################################################################    
    # reference_csv = './0-data/0-NYU-Zhemin/reference_speed_075_Ivan.csv'    
    # ref_data = np.loadtxt(reference_csv, delimiter=',', skiprows=1)       
    # reference_csv = './0-data/0-NYU-Zhemin/experiment_speed_125.csv'  
    # start_index   = 2000   
    # end_index     = 3000    
    # ref_data = np.loadtxt(reference_csv, delimiter=',', skiprows=1)[start_index:end_index, :]     
    
    # print("ref_data :", ref_data.shape)     
    
    # plot_two_lines(
    #     time_list=None,  
    #     data_actual=ref_data[:, 1:3],    
    #     data_reference=ref_data[:, 1:3],         
    #     data_force=None,  
    #     label_list=['Actual', 'Reference'],   
    #     save_path='./1-figure/comparison_of_hip_angles_two_lines'   
    # )   
    
    ####################################################################### 
    # data, left_gait_head_list_old, left_gait_tail_list_old, right_gait_head_list, right_gait_tail_list = load_gait_data(
    #     input_csv,         
    #     start_index=start_index, 
    #     end_index=end_index,  
    #     num_gait_cycles=5   
    # )       
    # left_gait_head_list  = [value + 15 for value in left_gait_head_list_old]  
    # left_gait_tail_list = [value + 15 for value in left_gait_tail_list_old]   
    
    # power_data      = {}    
    # percentage_data = {}    
    # both_side_data  = {}   
    # left_side_data  = {} 
    # right_side_data = {}  
    # for gait_index in range(5): 
    #     left_side_data['gait_'+str(gait_index)]  = data_power_ours[left_gait_head_list[gait_index]:left_gait_tail_list[gait_index], 0]  
    #     right_side_data['gait_'+str(gait_index)] = data_power_ours[right_gait_head_list[gait_index]:right_gait_tail_list[gait_index], 1]   
    
    # both_side_data['left']  = left_side_data   
    # both_side_data['right'] = right_side_data  
    # power_data['Ours']      = both_side_data    
    
    # percentage = calculate_positive_negative_perception(
    #     gait_vector_list= power_data['Ours']['left'],   
    #     num_gait        = 5  
    # )   
    # pos_neg_data = {} 
    # pos_neg_data['Positive']  = percentage[0, :]  
    # pos_neg_data['Negative'] = percentage[1, :]   
     
    # percentage_data['Ours']  = pos_neg_data   
    
    # both_side_data  = {}   
    # left_side_data  = {} 
    # right_side_data = {}  
    # for gait_index in range(5): 
    #     left_side_data['gait_'+str(gait_index)]  = data_power_samsung[left_gait_head_list[gait_index]:left_gait_tail_list[gait_index], 0]  
    #     right_side_data['gait_'+str(gait_index)] = data_power_samsung[right_gait_head_list[gait_index]:right_gait_tail_list[gait_index], 1]   
    
    # both_side_data['left']  = left_side_data
    # both_side_data['right'] = right_side_data
    # power_data['Samsung']   = both_side_data    
    
    # percentage = calculate_positive_negative_perception(
    #     gait_vector_list= power_data['Samsung']['left'],     
    #     num_gait        = 5  
    # )   
    # pos_neg_data = {} 
    # pos_neg_data['Positive']    = percentage[0, :]  
    # pos_neg_data['Negative']   = percentage[1, :]   
     
    # percentage_data['Samsung'] = pos_neg_data   
    
    # plot_comparison_negative_positive_power(
    #     power_data          = percentage_data,   
    #     mode_list           = ['Positive', 'Negative'],   
    #     method_list         = ['Ours', 'Samsung'],  
    #     num_gait            = 5,        
    #     fig_path            = './1-figure',    
    #     fig_name            = 'comparison_power_175'    
    # )   
    #############################################################################  
    # # subject_list    = ['Subject #1', 'Subject #2', 'Subject #3']  
    # subject_list    = ['Subject #3']  
    # index_list      = {'Subject #1': [3000, 5000], 'Subject #2': [4000, 6000], 'Subject #3': [4000, 6000]}   
    # # power_data      = {}  
    # percentage_data = {}  
    # method_list     = ['Ours', 'Samsung'] 
    
    # # input_csv     = './0-data/0-NYU-Zhemin/validation_experiment_speed_075.csv' 
    # input_csv     = './0-data/0-NYU-Zhemin/validation_experiment_speed_125.csv' 
    # # input_csv     = './0-data/0-NYU-Zhemin/validation_experiment_speed_175.csv' 
    
    # for subject_index, subject in enumerate(subject_list):  
    #     subject_data = {}
    #     # for method_index, method in enumerate(method_list):  
        
    #     start_index   = index_list[subject][0] 
    #     end_index     = index_list[subject][1]   
        
    #     ori_data = np.loadtxt(input_csv, delimiter=',', skiprows=1)  
    #     # print("data shape :", ori_data.shape)   
    #     data     = ori_data[start_index:end_index, :]  # Load data from CSV file   
        
    #     time_list      = data[:, 0]  
    #     data_actual    = data[:, 1:3]  
    #     data_reference = data_actual    
    #     data_velocity  = data[:, 3:5] 
    #     data_force     = data[:, 5:7]    
        
    #     data_power_ours   = obtain_our_assistive_power(
    #         data_force    = data_force, 
    #         data_velocity = data_velocity 
    #     )   
    #     data_force_samsung, data_power_samsung = obtain_samsung_control_force(
    #         data_angle    = data_actual,  
    #         data_velocity = data_velocity,  
    #         K             = 2, 
    #         delay         = 60, 
    #         gait_length   = 100 
    #     )   
        
    #     data, left_gait_head_list, left_gait_tail_list, right_gait_head_list, right_gait_tail_list = load_gait_data(
    #         input_csv,         
    #         start_index=start_index, 
    #         end_index=end_index,  
    #         num_gait_cycles=5   
    #     )   
        
    #     power_data      = {}    
    #     # percentage_data = {}    
    #     both_side_data  = {}   
    #     left_side_data  = {} 
    #     right_side_data = {}  
    #     for gait_index in range(5): 
    #         left_side_data['gait_'+str(gait_index)]  = data_power_ours[left_gait_head_list[gait_index]:left_gait_tail_list[gait_index], 0]  
    #         right_side_data['gait_'+str(gait_index)] = data_power_ours[right_gait_head_list[gait_index]:right_gait_tail_list[gait_index], 1]   

    #     both_side_data['left']  = left_side_data   
    #     both_side_data['right'] = right_side_data  
    #     power_data['Ours']      = both_side_data    

    #     percentage = calculate_positive_negative_perception(
    #         gait_vector_list= power_data['Ours']['left'],   
    #         num_gait        = 5  
    #     )   
    #     pos_neg_data = {} 
    #     pos_neg_data['Positive'] = percentage[0, :]  
    #     pos_neg_data['Negative'] = percentage[1, :]   
            
    #     subject_data['Ours']  = pos_neg_data   

    #     both_side_data  = {}   
    #     left_side_data  = {} 
    #     right_side_data = {}  
    #     for gait_index in range(5): 
    #         left_side_data['gait_'+str(gait_index)]  = data_power_samsung[left_gait_head_list[gait_index]:left_gait_tail_list[gait_index], 0]  
    #         right_side_data['gait_'+str(gait_index)] = data_power_samsung[right_gait_head_list[gait_index]:right_gait_tail_list[gait_index], 1]   

    #     both_side_data['left']  = left_side_data
    #     both_side_data['right'] = right_side_data
    #     power_data['Samsung']   = both_side_data    

    #     percentage = calculate_positive_negative_perception(
    #         gait_vector_list= power_data['Samsung']['left'],     
    #         num_gait        = 5  
    #     )   
    #     pos_neg_data = {} 
    #     pos_neg_data['Positive'] = percentage[0, :]  
    #     pos_neg_data['Negative'] = percentage[1, :]   
            
    #     subject_data['Samsung']  = pos_neg_data   
        
    #     print("subject :", subject)  
    #     percentage_data[subject] = subject_data  
     
    
    # plot_comparison_positive_power_participants(
    #     power_data          = percentage_data,  
    #     subject_list        = subject_list,  
    #     method_list         = method_list,  
    #     num_gait            = 5,    
    #     fig_path            = './1-figure',   
    #     fig_name            = 'comparison_power_075_multiple_subjects'   
    # )   
    
    #############################################################################
    # reference_csv = './0-data/0-NYU-Zhemin/reference_speed_075_Ivan.csv'    
    reference_csv = './0-data/0-NYU-Zhemin/reference_speed_075_Ivan.csv'   
    start_index   = 500   
    end_index     = 2000    
    ref_data = np.loadtxt(reference_csv, delimiter=',', skiprows=1)  
    print("ref_data :", ref_data.shape)     
    ref_data = ref_data[start_index:end_index, :]     
    
    plot_two_lines(
        time_list=None,  
        data_actual=ref_data[:, 1:3],    
        data_reference=ref_data[:, 1:3],         
        data_force=None,  
        label_list=['Actual', 'Reference'],   
        save_path='./1-figure/comparison_of_hip_angles_two_lines'   
    )   
    
    # data, left_gait_head_list, left_gait_tail_list, right_gait_head_list, right_gait_tail_list = load_gait_data(
    #     reference_csv,         
    #     start_index=start_index, 
    #     end_index=end_index,  
    #     num_gait_cycles=5   
    # )   
    
    # input_data                 = {}   
    # normalized_gait_length     = 101    
    # normalized_gait_cycle_list = np.arange(normalized_gait_length)    
    # fc = 5     
    # fs = 100        
    # b, a = butter(2, fc / (fs / 2), btype='low')    
    
    # input_data['gaitHeadList']         = left_gait_head_list   
    # input_data['gaitTailList']         = left_gait_tail_list   
    # input_data['data']                 = filtfilt(b, a, data[:, 1])    
    # input_data['normalizedGaitLength'] = normalized_gait_length    
    
    # avg_result, std_result, result_list = compute_gait_average_profile(input_data)  
    # print("avg_result :", avg_result.shape, result_list.shape)    
    
    # mu_gmr, sigma_gmr, gmr_model, mu_kmp, kmp_traj = obtain_gmr_mean_variance(args=args, input_data=np.array(result_list).reshape(-1, 1))   
    # print(mu_gmr.shape, sigma_gmr.shape)      
    
    # # plot_path_mean_var( 
    # #     nb_samples= args.nb_samples, 
    # #     nb_data   = args.nb_data, 
    # #     Xt        = normalized_gait_cycle_list, 
    # #     Y         = np.array(result_list).reshape(-1, 1),  
    # #     mu_gmr    = mu_gmr,   
    # #     # mu_gmr    = kmp_traj['mu'],     
    # #     sigma_gmr = sigma_gmr,   
    # #     save_path = './1-figure/gmr_path_mean_var',  
    # #     save_fig  = True  
    # # )   
    
    # plot_gait_mean_std(
    #     gait_cycle_list=normalized_gait_cycle_list,
    #     gait_data_mean =avg_result,   
    #     gait_data_std  =std_result,   
    #     save_path      ='./1-figure/comparison_of_one_gait' 
    # )   
    
    # # plot_gait_actual_target_trajectory(
    # #     nb_samples       = args.nb_samples, 
    # #     nb_data          = args.nb_data, 
    # #     Xt               = normalized_gait_cycle_list, 
    # #     Y                = np.array(result_list).reshape(-1, 1),  
    # #     gait_cycle_list  = normalized_gait_cycle_list, 
    # #     gait_data_actual = avg_result,    
    # #     gait_data_target = 1.25 * avg_result,   
    # #     scatter_list     = [0, 50, 100],  
    # #     save_path        = './1-figure/comparison_of_example_075.png'  
    # # )   
    
    # ########################################################
    # fc = 5     
    # fs = 100        
    # b, a = butter(2, fc / (fs / 2), btype='low')   
      
    # left_hip_angle = filtfilt(b, a, data[:, 1])    
    # print(left_hip_angle.shape)   
    # # data_reference[:, 0] = left_hip_angle    
    # right_hip_angle = filtfilt(b, a, data[:, 2])         
    # # data_reference[:, 1] = right_hip_angle    
    # print(right_hip_angle.shape)     
    
    # plot_comparison_results(
    #     time_list=data[:, 0],  
    #     data_actual=data[:, 1:3],    
    #     data_reference=np.hstack((left_hip_angle[:, np.newaxis], right_hip_angle[:, np.newaxis])),         
    #     left_gait_head_list=left_gait_head_list, 
    #     left_gait_tail_list=left_gait_tail_list, 
    #     right_gait_head_list=right_gait_head_list, 
    #     right_gait_tail_list=right_gait_tail_list,  
    #     label_list=['Actual', 'Reference'],  
    #     save_path='./1-figure/comparison_of_hip_angles'   
    # )   
    
    # # plot_start_index = 80
    # # plot_end_index   = 600
    # # plot_comparison_kinematics_kinetic_results(
    # #     time_list=data[plot_start_index:plot_end_index, 0],  
    # #     data_actual=np.hstack((left_hip_angle[plot_start_index:plot_end_index, np.newaxis], right_hip_angle[plot_start_index:plot_end_index, np.newaxis])),    
    # #     data_reference=np.hstack((left_hip_angle[plot_start_index:plot_end_index, np.newaxis], right_hip_angle[plot_start_index:plot_end_index, np.newaxis])),    
    # #     data_velocity=data_velocity[plot_start_index:plot_end_index, :]*1/10.0,  
    # #     data_force_ours=data_force[plot_start_index:plot_end_index, :],   
    # #     data_force_baseline=data_force_samsung[plot_start_index:plot_end_index, :],      
    # #     data_power_ours=None,     
    # #     data_power_baseline=None,        
    # #     left_gait_head_list=left_gait_head_list,      
    # #     left_gait_tail_list=left_gait_tail_list,     
    # #     right_gait_head_list=right_gait_head_list,      
    # #     right_gait_tail_list=right_gait_tail_list,      
    # #     label_list=['Angle', 'Velocity'],   
    # #     save_path='./1-figure/comparison_of_angles_force_speed_125_new.pdf'     
    # # )   
        
    # # plot_comparison_kinematics_kinetic_results(
    # #     time_list=data[:, 0],  
    # #     data_actual=np.hstack((left_hip_angle[:, np.newaxis], right_hip_angle[:, np.newaxis])),    
    # #     data_reference=np.hstack((left_hip_angle[:, np.newaxis], right_hip_angle[:, np.newaxis])),    
    # #     data_velocity=None,  
    # #     data_force_ours=None,   
    # #     data_force_baseline=None,     
    # #     data_power_ours=data_power_ours,    
    # #     data_power_baseline=data_power_samsung,       
    # #     left_gait_head_list=left_gait_head_list,      
    # #     left_gait_tail_list=left_gait_tail_list,     
    # #     right_gait_head_list=right_gait_head_list,    
    # #     right_gait_tail_list=right_gait_tail_list,      
    # #     label_list=['Actual', 'Reference'],   
    # #     save_path='./1-figure/comparison_of_angles_force_speed_075_new.png'    
    # # )   
    
    # # plot_comparison_negative_positive_power(
    # #     power_data        = None, 
    # #     mode_list         = ['Positive', 'Negative'],
    # #     method_list       = ['Ours', 'Samsung'],    
    # #     fig_path          = args.fig_path,   
    # #     fig_name          = args.fig_name   
    # # )   
    
    # # start_index = 47
    # # end_index   = 142  
    
    # input_data['gaitHeadList']         = left_gait_head_list   
    # input_data['gaitTailList']         = left_gait_tail_list   
    # input_data['data']                 = filtfilt(b, a, data[:, 1])    
    # input_data['normalizedGaitLength'] = normalized_gait_length    
    
    # avg_result_left, std_result_left, result_list_left = compute_gait_average_profile(input_data)  
    # print("avg_result :", avg_result_left.shape, result_list_left.shape)    
    
    # # mu_gmr, sigma_gmr, gmr_model, mu_kmp, kmp_traj = obtain_gmr_mean_variance(args=args, input_data=np.array(result_list).reshape(-1, 1))   
    # # print(mu_gmr.shape, sigma_gmr.shape)      
    
    # input_data['gaitHeadList']         = left_gait_head_list   
    # input_data['gaitTailList']         = left_gait_tail_list   
    # input_data['data']                 = filtfilt(b, a, data[:, 2])    
    # input_data['normalizedGaitLength'] = normalized_gait_length    
    
    # avg_result_right, std_result_right, result_list_right = compute_gait_average_profile(input_data)  
    # print("avg_result :", avg_result_left.shape, result_list_left.shape)  
    
    # index_list = np.linspace(0, 1, normalized_gait_length)  
    # extend_index_list = np.linspace(0, 1, 141) 
    # extend_left_result = np.interp(extend_index_list, index_list, avg_result_left)    
    # extend_right_result = np.interp(extend_index_list, index_list, avg_result_right)      
    
    # left_gait_data  = []  
    # right_gait_data = []  
    # for i in range(10):   
    #     left_gait_data.append(cp.deepcopy(extend_left_result))   
    #     right_gait_data.append(cp.deepcopy(extend_right_result))    
        
    # reference_data       = np.zeros((np.array(left_gait_data).reshape(-1, 1).shape[0], 2))
    # reference_data[:, 0] = np.array(left_gait_data).reshape(-1, 1)[:, 0]   
    # reference_data[:, 1] = np.array(right_gait_data).reshape(-1, 1)[:, 0] 
    # print("left gait data shape :", reference_data.shape)    

    # np.savetxt('reference_gait_speed_075_Ivan.csv', reference_data, delimiter=',')   
