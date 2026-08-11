# loading data from matlab 
from scipy.io import loadmat  
from utils.gmr import Gmr, plot_gmm   
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

X_LIM = [-15, 15]     
Y_LIM = [-15, 15]     


def plot_word_gmm(args=None, gmr_model=None, X=None, Y=None):  
    # Plots
	plt.figure(figsize=(5, 5))
	for p in range(args.nb_samples):
		plt.plot(Y[p * args.nb_data:(p + 1) * args.nb_data, 0], Y[p * args.nb_data:(p + 1) * args.nb_data, 1], color=[.7, .7, .7])
		plt.plot(Y[p * args.nb_data, 0], Y[p * args.nb_data, 1], color=[.7, .7, .7], marker='o')    
		
	plot_gmm(np.array(gmr_model.mu)[:, 1:], np.array(gmr_model.sigma)[:, 1:, 1:], alpha=0.6, color=[0.1, 0.34, 0.73])
	axes = plt.gca()  
	axes.set_xlim(X_LIM)   
	axes.set_ylim(Y_LIM)   
	plt.xlabel(r'$X$', fontsize=font_size)   
	plt.ylabel(r'$Y$', fontsize=font_size)   
	plt.locator_params(nbins=3)  
	plt.tick_params(labelsize=font_size)   
	plt.tight_layout()   
	
	if args.save_fig:     
		plt.savefig(args.fig_path + args.letter + '_gmm.pdf', bbox_inches='tight', pad_inches=0.0)  
		plt.show()    


def plot_word_gmr(args=None, gmr_model=None, 
		  X=None, Y=None,   
		  mu_gmr=None, sigma_gmr=None,    
		  X_obs=None, Y_obs=None   
	):    
	plt.figure(figsize=(5, 5))    
	for p in range(args.nb_samples):  
		plt.plot(Y[p* args.nb_data: (p+1) * args.nb_data, 0], Y[p * args.nb_data:(p+1) * args.nb_data, 1], color=[.7, .7, .7])
		plt.scatter(Y[p * args.nb_data, 0], Y[p * args.nb_data, 1], color=[.7, .7, .7], marker='X', s=80)
	plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3)   
	plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)  
	plot_gmm(mu_gmr, sigma_gmr, alpha=0.05, color=[0.20, 0.54, 0.93])  

	if Y_obs is not None:    
		plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=50, s=100)    

		for obs_index in X_obs:   
			obs_index = int(obs_index/args.T * args.nb_data)  
			print("obs_index :", obs_index)   
			# observation points    
			plt.scatter(mu_gmr[obs_index, 0], mu_gmr[obs_index, 1], color=[0.23, 0.23, 0.23], marker='X', s=80)    

	axes = plt.gca()    
	axes.set_xlim(X_LIM)      
	axes.set_ylim(Y_LIM)       
	plt.xlabel(r'$X$', fontsize=font_size)     
	plt.ylabel(r'$Y$', fontsize=font_size)    
	plt.locator_params(nbins=3)   
	plt.tick_params(labelsize=font_size)   
	plt.tight_layout()   

	if args.save_fig:   
		plt.savefig(args.fig_path + args.letter + '_gmr' + args.flag_letter + '.pdf', bbox_inches='tight', pad_inches=0.0)  
		plt.show()  
	

def plot_word_gmr_gp(  
	args=None, gmr_model=None,  
	X=None, Y=None,   
	Y_obs=None, mu_posterior=None,   
	mu_gmr=None, sigma_gmr=None,   
	mu_gp_rshp=None, sigma_gp_rshp=None   
):  
	plt.figure(figsize=(5, 5))    
	plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)   
	
	nb_posterior_samples = len(mu_posterior)       
	for i in range(nb_posterior_samples):     
		plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)   
		plt.scatter(mu_posterior[i][0, 0], mu_posterior[i][1, 0], color=[0.64, 0., 0.65], marker='X', s=80)     
	
	plt.plot(mu_gp_rshp[:, 0], mu_gp_rshp[:, 1], color=[0.83, 0.06, 0.06], linewidth=3.)   
	plt.scatter(mu_gp_rshp[0, 0], mu_gp_rshp[0, 1], color=[0.83, 0.06, 0.06], marker='X', s=80)   
	plot_gmm(mu_gp_rshp, sigma_gp_rshp, alpha=0.05, color=[0.83, 0.06, 0.06])   

	# observation points  
	plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)   
	
	axes = plt.gca()    
	axes.set_xlim(X_LIM)     
	axes.set_ylim(Y_LIM)     
	plt.xlabel(r'$X$', fontsize=font_size)      
	plt.ylabel(r'$Y$', fontsize=font_size)      
	plt.locator_params(nbins=3)     
	plt.tick_params(labelsize=font_size)      
	plt.tight_layout()     

	if args.save_fig:    
		plt.savefig(args.fig_path + args.letter + '_gmr_gp' + args.flag_letter + '.pdf', bbox_inches='tight', pad_inches=0.0)    
		plt.show()