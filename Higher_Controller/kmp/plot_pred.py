import numpy as np 
import matplotlib.pyplot as plt  
from scipy.io import loadmat   
from pathlib import Path

try:
    from .GMRbasedGP.utils.gmr import plot_gmm, Gmr
except ImportError:  # Allow running this file directly from the kmp folder.
    from GMRbasedGP.utils.gmr import plot_gmm, Gmr
import seaborn as sns   

import numpy as np   
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 

# sns.set(palette="muted", color_codes=True, font_scale=1.5)   
custom_params = {"axes.spines.right": False, "axes.spines.top": False}   
sns.set_theme(style="ticks", font_scale=2.0, rc=custom_params)    
colors = np.array(["#0072B2", "#F0E442", "#D55E00"])      
random_state, n_components, n_features = 2, 3, 2    
font_size = 20

mycolors = {
    'nr': [213/255,15/255,37/255],  # new red
    'ng': [0/255,153/255,37/255],  # new green
    'nb': [51/255,105/255,232/255],  # new blue  
    'ny': [238/255,178/255,17/255],  # new yellow   
    'r': [180/255,20/255,47/255],  # red  
    'b': [0,114/255,189/255],  # blue  
    'db': [0,100/255,200/255],  # blue  
    'g': [119/255,172/255,48/255],  # green  
	'o': [217/255,83/255,25/255],  # orange  
	'y': [237/255,177/255,32/255], 
	'p': [126/255,47/255,142/255], 
	'pi': [204/255,102/255,102/255], 
	'lb': [77/255,190/255,238/255], 
	'li': [164/255,196/255,0], 
	'lr': [229/255,20/255,0], 
	'lg': [220/255,220/255,220/255], 
	'dr': [102/255,0,0],  
	'em': [0,138/255,0],  
    'br': [0.6510, 0.5725, 0.3412],  
    'gy': [0.6, 0.6, 0.6],  
    'vgy': [160/255, 160/255, 160/255],  
    'm': [1, 0, 1],  
    'c':  [0, 1, 1], 
    'rr': [1.0, 0.4, 0.4],  
    'gl': [0.8314, 0.7020, 0.7843]   
}   

color_name = ['nr', 'ng', 'nb', 'ny', 'r', 'b', 'db', 'g', 'o', 'y', 
	'p', 'pi', 'lb', 'li', 'lr', 'lg', 'dr', 'em', 'br', 'gy', 'vgy', 'm', 'c', 'rr', 'gl']   

X_LIM = [-30.0, 30.0]        
Y_LIM = [-30.0, 30.0]        


def plot_gait_distribution(
    result, signal_names=None, gait_data=None, font_size=14
):
    """Plot GMR and KMP means with one-standard-deviation bands."""
    if font_size <= 0:
        raise ValueError("font_size must be positive")
    gmr_mean = np.asarray(result["gmr_mean"])
    n_signals = gmr_mean.shape[1]
    names = signal_names or [f"Signal {index + 1}" for index in range(n_signals)]
    if len(names) != n_signals:
        raise ValueError("signal_names length must match the signal dimension")

    figure, axes = plt.subplots(
        n_signals, 1, figsize=(9, 3.3 * n_signals), sharex=True, squeeze=False
    )
    gait_percent = np.asarray(result["gait_percent"])
    raw_gaits = None if gait_data is None else np.asarray(gait_data, dtype=float)
    if raw_gaits is not None and raw_gaits.ndim == 2:
        raw_gaits = raw_gaits[:, :, None]

    for signal_index, axis in enumerate(axes[:, 0]):
        if raw_gaits is not None:
            for gait in raw_gaits:
                axis.plot(
                    gait_percent, gait[:, signal_index], color="0.75",
                    linewidth=0.75, alpha=0.45,
                )

        for prefix, color, label in (
            ("gmr", "#0072B2", "GMR"),
            ("kmp", "#D55E00", "KMP"),
        ):
            mean = np.asarray(result[f"{prefix}_mean"])[:, signal_index]
            variance = np.maximum(
                np.asarray(result[f"{prefix}_variance"])[:, signal_index], 0.0
            )
            standard_deviation = np.sqrt(variance)
            axis.plot(gait_percent, mean, color=color, linewidth=2.4, label=label)
            axis.fill_between(
                gait_percent,
                mean - standard_deviation,
                mean + standard_deviation,
                color=color,
                alpha=0.16,
            )

        via_indices = np.asarray(result.get("via_indices", []), dtype=int)
        via_points = np.asarray(result.get("via_points", []), dtype=float)
        if len(via_indices) > 0:
            axis.scatter(
                gait_percent[via_indices],
                via_points[:, signal_index],
                color="#7A1FA2",
                marker="X",
                s=70,
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
                label="Stretched GMR via point",
            )

        axis.set_ylabel(
            names[signal_index], fontsize=font_size, fontweight="semibold"
        )
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.tick_params(labelsize=font_size - 1)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[-1, 0].set_xlabel(
        "Normalized gait cycle (%)", fontsize=font_size, fontweight="semibold"
    )
    axes[-1, 0].set_xlim(0.0, 100.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Remove duplicate labels while preserving their visual order.
    unique = dict(zip(labels, handles))
    figure.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=max(len(unique), 1),
        frameon=False,
        columnspacing=1.8,
        handlelength=2.4,
        fontsize=font_size - 1,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), h_pad=1.0)
    return figure


def plot_bilateral_hip_mean_variance(
    result, gait_data=None, save_path=None, show=False, font_size=14
):
    """Plot GMR/KMP mean and variance for left and right hip angles."""
    gmr_mean = np.asarray(result["gmr_mean"])
    if gmr_mean.ndim != 2 or gmr_mean.shape[1] != 2:
        raise ValueError(
            "Bilateral hip result must contain exactly two signals: left, right"
        )

    figure = plot_gait_distribution(
        result=result,
        signal_names=("Left hip angle (deg)", "Right hip angle (deg)"),
        gait_data=gait_data,
        font_size=font_size,
    )
    figure.suptitle(
        "Bilateral Hip-Angle Mean and Variance",
        y=0.995,
        fontsize=font_size + 3,
        fontweight="bold",
    )
    if figure.legends:
        figure.legends[0].set_bbox_to_anchor((0.5, 0.955))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.87), h_pad=1.0)

    if save_path is not None:
        output = Path(save_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved bilateral hip distribution: {output}")
    if show:
        plt.show()
    return figure


def plot_energy_deformation(
        nb_samples=5, nb_data=200,  
        real_data=None, ref_data=None, label_list=None,  
        start=None, end=None, 
        start_index=None, end_index=None,  
        obs_center=None,  
        font_size=18,  
        font_name='G', 
        save_fig=None, 
        fig_path=None   
    ):      	   
    plt.figure(figsize=(5, 5))        
    
    for p in range(nb_samples):       
        X = real_data[p][:, 0]      
        Y = real_data[p][:, 1]      
        plt.plot(X, Y, label=label_list[p], linewidth=2.5, color=mycolors[color_name[p]])    

    plt.plot(ref_data[:, 1], ref_data[:, 0], color="black", linewidth=3, label=r'$\chi_d$')       
    # plt.plot(ref_data[:, 1], ref_data[:, 0], color="black", linewidth=3, label='Reference')    
    
    plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], fc='k', ec='k', lw=3, length_includes_head=True, head_width=1.0, head_length=1.0, color='k') # fc='k', ec='k', lw=2, , length_includes_head=True, head_width=0.2, head_length=0.3, color='k'
    
    plt.scatter(ref_data[0, 0], ref_data[0, 1], color='black', marker='X', s=80)   
    plt.scatter(ref_data[start_index, 0], ref_data[start_index, 1], color=[0.20, 0.93, 0.54], marker='X', s=80)    
    plt.scatter(ref_data[end_index, 0], ref_data[end_index, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)    

    draw_circle_obs = plt.Circle(obs_center, 5, fill=True, color=mycolors['lb'], alpha=0.6)     
    # plt.text(center_point_left[0]-1, center_point_left[1]-4, 'Impaired')    mycolors[color_name 
    plt.gcf().gca().add_artist(draw_circle_obs)      

    plt.text(end[0] - 6.0, end[1] + 0.2, r'$F_h$', size=font_size)    
    
    plt.text(ref_data[start_index, 0] - 5.0, ref_data[start_index, 1] + 0.5, r'$t_s$', size=font_size)   
    plt.text(ref_data[end_index, 0] + 2.0, ref_data[end_index, 1] + 0.5, r'$t_e$', size=font_size)   

    ax = plt.gca()  
    # plt.ylabel("Y", fontsize=font_size)          
    # plt.xlabel("X", fontsize=font_size)       
    
    plt.ylabel(r"$x_2$[mm]", fontsize=font_size)          
    plt.xlabel(r"$x_1$[mm]", fontsize=font_size)         
    # plt.ylabel("Y[mm]", fontsize=font_size)          
    # plt.xlabel("X[mm]", fontsize=font_size)          
    
    ax.set_xlim(X_LIM)    
    ax.set_ylim(Y_LIM)    
    ax.set_aspect(1)    
    plt.tight_layout()        
    plt.locator_params(nbins=3)        
    plt.tick_params(labelsize=font_size)         

    plt.legend(handlelength=1.0, fontsize=font_size-2.5, bbox_to_anchor=(0.70, 0.60), columnspacing=0.40, handletextpad=0.05) 

    if save_fig:  
        print("save figure success !!!", fig_path + font_name + '.pdf')    
        plt.savefig(fig_path + font_name + '.pdf', bbox_inches='tight', pad_inches=0.0)     
    plt.show()     


def plot_raw_data(font_name='G', nb_samples=5, nb_dim=5, nb_data=200, X=None, Y=None, mu_gmr=None, sigma_gmr=None, ref_data=None):  	 
    plt.figure(figsize=(10, 5))   
    X_t = np.array(X[0:nb_data])   
    
    for p in range(nb_samples):   
        for j in range(nb_dim):    
            plt.plot(X_t, Y[p * nb_data:(p + 1) * nb_data, j], color=mycolors[color_name[j+7]])    
            # plt.plot(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='o')    
    # print("mu :", gmr_model.mu)    
    
    for i in range(2):  
        # plt.plot(X_t, ref_data[:, i], color='black', linewidth=3)  
        plt.plot(X_t, mu_gmr[:, i], color=[0.20, 0.54, 0.93], linewidth=3)   
        # plt.plot(X_t, mu_gmr[:, 0], color=[0.93, 0.54, 0.20], linewidth=3)    
        miny = mu_gmr[:, i] - np.sqrt(sigma_gmr[:, i, i])    
        maxy = mu_gmr[:, i] + np.sqrt(sigma_gmr[:, i, i])     
        # print(X_t.shape, np.array(miny[:, None]).shape, maxy.shape)   
        plt.fill_between(np.squeeze(X_t), np.array(miny), np.array(maxy), color=[0.20, 0.54, 0.93], alpha=0.3)     
    
    # miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])    
    # maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])     
    # # print(X_t.shape, np.array(miny[:, None]).shape, maxy.shape)   
    # plt.fill_between(np.squeeze(X_t), np.array(miny), np.array(maxy), color=[0.93, 0.54, 0.20], alpha=0.3)     
    
    axes = plt.gca()   
    # axes.set_xlim([-14., 14.])     
    # axes.set_ylim([0., 1.])    
    plt.xlabel('Time[s]', fontsize=font_size)      
    plt.ylabel('Output', fontsize=font_size)     
    plt.locator_params(nbins=3)   
    plt.tick_params(labelsize=font_size)     
    plt.tight_layout()   
    plt.legend()   
    # plt.savefig('figures/GMM_' + font_name +'.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()  


def plot_GMM_raw_data(
        nb_samples=5, nb_data=200, Y=None, ref_data=None, 
        gmr_model=None, gmr_model_second=None,  
        font_name='G', fig_path=None, save_fig=False
    ):	
    plt.figure(figsize=(5, 5))   

    plt.plot(ref_data[1, :], ref_data[0, :], color="black", linewidth=3)   

    for p in range(nb_samples):   
        plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
        plt.plot(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='o')  

    if gmr_model is not None: 
        plot_gmm(np.array(gmr_model.mu)[:, 1:3], np.array(gmr_model.sigma)[:, 1:3, 1:3], alpha=0.45, color=[0.1, 0.34, 0.73])
    
    if gmr_model_second is not None:  
        plot_gmm(np.array(gmr_model_second.mu)[:, 1:3], np.array(gmr_model_second.sigma)[:, 1:3, 1:3], alpha=0.25, color=[0.1, 0.73, 0.34])

    ax = plt.gca()   
    # axes.set_xlim([-8., 8.])     
    # axes.set_ylim([-4., 14.])     
    # plt.xlabel('$y_1$', fontsize=30)      
    # plt.ylabel('$y_2$', fontsize=30)      

    # plt.ylabel("X[mm]", fontsize=font_size)  # fontsize=25  
    # plt.xlabel("Y[mm]", fontsize=font_size)  # fontsize=25  fontsize=font_size

    plt.ylabel("Y[mm]", fontsize=font_size)  # fontsize=25  
    plt.xlabel("X[mm]", fontsize=font_size)  # fontsize=25  fontsize=font_size
    ax.set_xlim(X_LIM)        
    ax.set_ylim(Y_LIM)          
    plt.locator_params(nbins=3)     
    plt.tick_params(labelsize=20)    
    plt.tight_layout()  
    
    if save_fig:  
        plt.savefig(fig_path + font_name + '_gmm.pdf', bbox_inches='tight', pad_inches=0.0)   
    plt.show()  
    

def plot_mean_var(font_name='G', nb_samples=5, nb_data=200, Xt=None, Y=None, mu_gmr=None, sigma_gmr=None, ref_data=None, fig_path=None, save_fig=False): 
    plt.figure(figsize=(5, 5))    
    for p in range(nb_samples):    
        plt.plot(Y[p*nb_data:(p+1)*nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.55, .55, .55])      
        plt.scatter(Y[p*nb_data, 0], Y[p*nb_data, 1], color=[.55, .55, .55], marker='X', s=80)     
    
    print("mu_gmr :", mu_gmr[:, :2].shape, "sigma_gmr :", sigma_gmr[:, :2, :2].shape)   
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=2)   
    plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)    

    plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2] * 1.5, alpha=0.20, color=[0.20, 0.54, 0.93])    

    # ref data 
    plot_gmm(ref_data.T[:, :2], sigma_gmr[:, :2, :2] * 1.5, alpha=0.20, color=[0.20, 0.93, 0.54])   
    
    plt.scatter(ref_data[0, 0], ref_data[1, 0], color=[0.20, 0.54, 0.93], marker='X', s=80)    

    plt.plot(ref_data[1, :], ref_data[0, :], color="black", linewidth=3)   
    
    ax = plt.gca()    
    # axes.set_xlim([-8., 8.])       
    # axes.set_ylim([-4., 14.])     
    # axes.set_xlim([-14, 14.])  
    # axes.set_ylim([-14., 14.])     
    # plt.xlabel('$y_1$', fontsize=30)     
    # plt.ylabel('$y_2$', fontsize=30)     
    # plt.ylabel("X[mm]", fontsize=font_size)   #, fontsize=15
    # plt.xlabel("Y[mm]", fontsize=font_size)   #, fontsize=15

    plt.ylabel("Y[mm]", fontsize=font_size)   #, fontsize=15
    plt.xlabel("X[mm]", fontsize=font_size)   #, fontsize=15
    ax.set_xlim(X_LIM)         
    ax.set_ylim(Y_LIM)        
    plt.locator_params(nbins=3)       
    plt.tick_params(labelsize=font_size)         
    plt.tight_layout()      
    
    if save_fig:  
        print("save gmr !!!", fig_path + font_name + '_gmr.pdf')    
        plt.savefig(fig_path + font_name + '_gmr.pdf', bbox_inches='tight', pad_inches=0.0)     
    plt.show()   


def plot_via_points(
    font_name='G',  
    nb_posterior_samples=None,  
    via_points=None,  
    mu_gmr=None,  
    pred_gmr=None,  
    sigma_gmr=None,   
    sigma_kmp=None,   
    ref_data=None,   
    obs_center=None,    
    real_data=None,   
    start=None,  
    end=None,   
    save_fig=None,   
    fig_path=None    
):  
    plt.figure(figsize=(5, 5))   
    
    draw_circle_obs = plt.Circle(obs_center, 5, fill=True, color=mycolors['lb'], alpha=0.6)      
    # plt.text(center_point_left[0]-1, center_point_left[1]-4, 'Impaired')   
    plt.gcf().gca().add_artist(draw_circle_obs)     
    # # plt.text(center_point_left[0]-1, center_point_left[1]-4, 'Impaired')     
    # plt.gcf().gca().add_artist(draw_circle_obs)     

    plt.plot(ref_data[0, :], ref_data[1, :], color="black", linewidth=3)    

    for p in range(len(real_data)-1, len(real_data)):     
        X = real_data[p][:, 0]      
        Y = real_data[p][:, 1]      
        plt.plot(X, Y, linewidth=2.5, color=mycolors[color_name[0]])      

    #################  via points  ######################3
    plt.scatter(via_points[0, 0], via_points[0, 1], color=[0.64, 0., 0.65], marker='X', s=80, label='via-points')   
    for i in range(1, nb_posterior_samples):  
        # plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)  
        plt.scatter(via_points[i, 0], via_points[i, 1], color=[0.64, 0., 0.65], marker='X', s=80)      
    
    plt.text(via_points[1, 0] - 5.0, via_points[1, 1] + 0.5, r'$t_s$', size=font_size)   
    plt.text(via_points[-2, 0] + 2.0, via_points[-2, 1] + 0.5, r'$t_e$', size=font_size)   

    # # ori value    
    # plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color='black', marker='X', s=80, label='start-points')   
    # plt.scatter(mu_gmr[198, 0], mu_gmr[198, 1], color='green', marker='X', s=80, label='end-points')  

    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.94, 0.54], linewidth=3)  
    plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2], alpha=0.1, color=[0.20, 0.93, 0.54])  

    # # pred value  
    # plt.scatter(pred_gmr[0, 0], pred_gmr[0, 1], color='black', marker='X', s=80, label='start-points')   
    # plt.scatter(pred_gmr[196, 0], pred_gmr[196, 1], color='green', marker='X', s=80, label='end-points')   

    plt.plot(pred_gmr[:, 0], pred_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3, linestyle='--')  
    plot_gmm(pred_gmr[:, :2], sigma_kmp[:, :2, :2] * 0.5, alpha=0.1, color=[0.20, 0.54, 0.93])  

    # plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color='black', marker='X', s=80)   
    # plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2], alpha=0.05, color=[0.20, 0.54, 0.93])    
    
    # obs_center = np.array([0, 18])  

    # draw_circle_obs = plt.Circle(obs_center, 5, fill=True, color='black')     
    # # plt.text(center_point_left[0]-1, center_point_left[1]-4, 'Impaired')     
    # plt.gcf().gca().add_artist(draw_circle_obs)   

    plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], fc='k', ec='k', 
              lw=3, length_includes_head=True, head_width=1.0, head_length=1.0, color='k') # fc='k', ec='k', lw=2, , length_includes_head=True, head_width=0.2, head_length=0.3, color='k'
    # plt.text(end[0] + 1.5, end[1] + 1.2, r'$F_h$', size=font_size)    
    plt.text(end[0] - 5.0, end[1] + 0.2, r'$F_h$', size=font_size)     
    # plt.annotate('annotate', xy=(3, 2))   

    plt.ylabel("Y[mm]", fontsize=font_size)         
    plt.xlabel("X[mm]", fontsize=font_size)      
    plt.legend()       
    # ax.set_xlim(-20.0, 20.0)       
    # ax.set_ylim(-20.0, 20.0)  
    # ax.set_xlim(-30.0, 30.0)       
    # ax.set_ylim(-30.0, 30.0)  
    # axes.set_xlim([-14, 14.])   
    # axes.set_ylim([-14., 14.])   
    # plt.xlabel('$y_1$', fontsize=30)   
    # plt.ylabel('$y_2$', fontsize=30)   
    plt.locator_params(nbins=3)   
    plt.tick_params(labelsize=font_size)   
    plt.tight_layout()   

    if save_fig:   
        print(fig_path + font_name + '_kmp.pdf')    
        plt.savefig(fig_path + font_name + '_kmp.pdf', bbox_inches='tight', pad_inches=0.0)    
    plt.show()   
    

def plot_mean_var_fig(font_name='G', nb_samples=5, nb_data=200, Xt=None, Y=None, mu_gmr=None, sigma_gmr=None, pred_gmr=None, pred_sigma=None, ref_data=None, via_points=None, via_time=None):
    plt.figure(figsize=(14, 5))  
    font_size = 15 
    plt.subplot(1,3,1) 
    for p in range(nb_samples):   
        plt.plot(Y[p*nb_data:(p+1)*nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.7, .7, .7])   
        plt.scatter(Y[p*nb_data, 0], Y[p*nb_data, 1], color=[.7, .7, .7], marker='X', s=80)    
    
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3)   
    plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)    
    plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2], alpha=0.05, color=[0.20, 0.54, 0.93])    
    
    # plt.plot(ref_data[1, :], ref_data[0, :], color="black", linewidth=3)   
    for i in range(via_points.shape[0]):  
        # plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)  
        plt.scatter(via_points[i, 0], via_points[i, 1], color=[0.64, 0., 0.65], marker='X', s=80)  

        # pred value  
    plt.plot(pred_gmr[:, 0], pred_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3, linestyle='--')  
    
    ax = plt.gca()    
    
    plt.ylabel("X[mm]", fontsize=15)         
    plt.xlabel("Y[mm]", fontsize=15)          
    # ax.set_xlim(-30.0, 30.0)         
    # ax.set_ylim(-30.0, 30.0)      
    plt.locator_params(nbins=3)        
    plt.tick_params(labelsize=15)        
    plt.tight_layout()       

    ax_2 = plt.subplot(1,3,2) 
    # plt.figure(figsize=(5, 4))  
    for p in range(nb_samples):   
        plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 0], color=[.7, .7, .7]) 
        
    plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3) 
    plt.plot(Xt[:, 0], pred_gmr[:, 0], color=[0.93, 0.54, 0.20], linewidth=3, linestyle='--') 
    miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])  
    maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])  
    plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)  
    for i in range(via_points.shape[0]):  
        # plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)  
        plt.scatter(via_time[i], via_points[i, 0], color=[0.64, 0., 0.65], marker='X', s=80)  

    
    # plt.plot(Xt[:, 0], mu_gmr[:, 0] + ref_data[1, :], color="green", linewidth=3)   
    # plt.plot(Xt[:, 0], -1 * ref_data[1, :], color="black", linewidth=3)   
    
    # ax_2 = plt.gca()
    # ax_2.set_ylim([-30., 30.]) 
    plt.xlabel('$t$', fontsize=font_size) 
    plt.ylabel('$y_1$', fontsize=font_size) 
    plt.tick_params(labelsize=font_size)   
    plt.tight_layout()
    # plt.savefig('figures/GMR_' + font_name + '_1.png', bbox_inches='tight', pad_inches=0.0)  

    ax_3 = plt.subplot(1,3,3)  
    # plt.figure(figsize=(5, 4))   
    for p in range(nb_samples):   
        plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])  
    
    plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3) 
    plt.plot(Xt[:, 0], pred_gmr[:, 1], color=[0.93, 0.54, 0.20], linewidth=3, linestyle='--') 
    miny = mu_gmr[:, 1] - np.sqrt(sigma_gmr[:, 1, 1])  
    maxy = mu_gmr[:, 1] + np.sqrt(sigma_gmr[:, 1, 1])   
    plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)  
    for i in range(via_points.shape[0]):  
        # plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)  
        plt.scatter(via_time[i], via_points[i, 1], color=[0.64, 0., 0.65], marker='X', s=80)  

    # plt.plot(Xt[:, 0], mu_gmr[:, 1] + ref_data[0, :], color="green", linewidth=3)   
    # plt.plot(Xt[:, 0], -1 * ref_data[0, :], color="black", linewidth=3) 
       
    # ax_3 = plt.gca()
    # ax_3.set_ylim([-30., 30.])
    plt.xlabel('$t$', fontsize=font_size)
    plt.ylabel('$y_2$', fontsize=font_size)
    plt.tick_params(labelsize=font_size)    
    plt.tight_layout()  
    # plt.savefig('figures/GMR_' + font_name + '_2.png', bbox_inches='tight', pad_inches=0.0)  
    
    plt.savefig('figures/GMR_' + font_name + '.png', bbox_inches='tight', pad_inches=0.0)  
    plt.show()   


def plot_poster_samples(
    mu_gmr=None,  
    mu_gp_rshp=None,   
    sigma_gp_rshp=None,     
    Y_obs=None,  
    nb_posterior_samples=None,   
    mu_posterior=None   
):  
    # Posterior
    plt.figure(figsize=(5, 5))  
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
    plot_gmm(mu_gp_rshp, sigma_gp_rshp, alpha=0.05, color=[0.83, 0.06, 0.06])

    for i in range(nb_posterior_samples):
        plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)
        plt.scatter(mu_posterior[i][0, 0], mu_posterior[i][1, 0], color=[0.64, 0., 0.65], marker='X', s=80)

    plt.plot(mu_gp_rshp[:, 0], mu_gp_rshp[:, 1], color=[0.83, 0.06, 0.06], linewidth=3.)
    plt.scatter(mu_gp_rshp[0, 0], mu_gp_rshp[0, 1], color=[0.83, 0.06, 0.06], marker='X', s=80)
    plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)

    ax = plt.gca()  
    plt.ylabel("X[mm]")         
    plt.xlabel("Y[mm]")         
    ax.set_xlim(-30.0, 30.0)       
    ax.set_ylim(-30.0, 30.0) 
    ax.set_aspect(1)   
    plt.tight_layout()        
    plt.locator_params(nbins=3)     
    plt.tick_params(labelsize=20)    
    plt.tight_layout()
    # plt.savefig('figures/GMRbGP_B_posterior_datasup.png')  
    plt.show()  


def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(
            means[n], eig_vals[0], eig_vals[1], 180 + angle, edgecolor="black"
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor("#56B4E9")
        ax.add_artist(ell)


def plot_results(ax1, ax2, estimator, X, y, title, plot_title=False):
    ax1.set_title(title)
    ax1.scatter(X[:, 0], X[:, 1], s=5, marker="o", color=colors[y], alpha=0.8)
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-3.0, 3.0)
    ax1.set_xticks(())
    ax1.set_yticks(())
    plot_ellipses(ax1, estimator.weights_, estimator.means_, estimator.covariances_)

    ax2.get_xaxis().set_tick_params(direction="out")
    ax2.yaxis.grid(True, alpha=0.7)
    for k, w in enumerate(estimator.weights_):
        ax2.bar(
            k,
            w,
            width=0.9,
            color="#56B4E9",
            zorder=3,
            align="center",
            edgecolor="black",
        )
        ax2.text(k, w + 0.007, "%.1f%%" % (w * 100.0), horizontalalignment="center")
    ax2.set_xlim(-0.6, 2 * n_components - 0.4)
    ax2.set_ylim(0.0, 1.1)
    ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax2.tick_params(axis="x", which="both", top=False)

    if plot_title:
        ax1.set_ylabel("Estimated Mixtures")
        ax2.set_ylabel("Weight of each component") 
        
        
def plot_error_data(font_name=None, X=None, nb_data=200, error_data=None): 
    plt.figure(figsize=(10, 5))   
    X_t = np.array(X[0:nb_data])   
    
    for i in range(3):  
        plt.plot(X_t, error_data[:, i], linewidth=3, label='state_'+str(i))    
        # plt.plot(X_t, mu_gmr[:, i], color=[0.20, 0.54, 0.93], linewidth=3)   
        # # plt.plot(X_t, mu_gmr[:, 0], color=[0.93, 0.54, 0.20], linewidth=3)    
        # miny = mu_gmr[:, i] - np.sqrt(sigma_gmr[:, i, i])    
        # maxy = mu_gmr[:, i] + np.sqrt(sigma_gmr[:, i, i])     
        # # print(X_t.shape, np.array(miny[:, None]).shape, maxy.shape)   
        # plt.fill_between(np.squeeze(X_t), np.array(miny), np.array(maxy), color=[0.20, 0.54, 0.93], alpha=0.3)     
    
    # miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])    
    # maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])     
    # # print(X_t.shape, np.array(miny[:, None]).shape, maxy.shape)   
    # plt.fill_between(np.squeeze(X_t), np.array(miny), np.array(maxy), color=[0.93, 0.54, 0.20], alpha=0.3)     
    
    axes = plt.gca()   
    # axes.set_xlim([-14., 14.])     
    # axes.set_ylim([0., 1.])    
    plt.xlabel('Time[s]', fontsize=font_size)      
    plt.ylabel('Output', fontsize=font_size)     
    plt.locator_params(nbins=3)   
    plt.tick_params(labelsize=font_size)     
    plt.tight_layout()   
    plt.legend() 
    plt.savefig('figures/GMR_error_' + font_name +'.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()   
    
    
def plot_mean_var_error(font_name=None, real_data=None, ref_data=None):   
    plt.figure(figsize=(10, 5))   
    # axes = plt.gca()   
    # axes.set_xlim([-14., 14.])     
    # axes.set_ylim([0., 1.])    
    label_xyz_list = ['X', 'Y', 'Z']  
    label_xyz_t_list = ['X_t', 'Y_t', 'Z_t']    
    for i in range(3): 
        plt.plot(ref_data[:, i], label=label_xyz_list[i])   
        
    for j in range(1):  
        for i in range(3): 
            plt.plot(real_data[j][:, i], label=label_xyz_t_list[i])    
            
    plt.xlabel('Time[s]', fontsize=font_size)        
    plt.ylabel('Output', fontsize=font_size)      
    plt.locator_params(nbins=3)    
    plt.tick_params(labelsize=font_size)     
    plt.tight_layout()   
    plt.legend() 
    plt.savefig('figures/Mean_std_' + font_name +'.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()  
