from numpy.lib import NumpyVersion  
import numpy as np  
import math  
import os  
import scipy.io as scio  
import ctypes  
import time   
import glob   
import scipy  
import argparse    
from sklearn.metrics import mean_squared_error     

# from path_planning.robInfLibpy.demo_GMR import *    
# from plot_main import *    
# from path_planning.utils_functions import *     
# # from path_planning.spm_path.spm_kinematics import *    

import GPy    
from scipy.io import loadmat # loading data from matlab
from utils.gmr import Gmr
from utils.gmr import plot_gmm
from utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
from utils.gmr_mean_mapping import GmrMeanMapping
from utils.gmr_kernels import Gmr_based_kernel 

from scipy.io import loadmat  
from utils.gmr import Gmr, plot_gmm   

from matplotlib.pyplot import title      
import matplotlib.pyplot as plt     
import seaborn as sns         
# sns.set(font_scale=1.5)        
#  
np.set_printoptions(precision=4)       
from plot_functions import plot_word_gmm, plot_word_gmr, plot_word_gmr_gp          


if __name__ == "__main__":     
    parser = argparse.ArgumentParser()      

    parser.add_argument('--mode', type=int, default=0, help='choose mode first !!!!')       
    parser.add_argument('--ctl_mode', type=str, default="zero_force", help='choose mode first !!!!')       
    # //// basics ///////////  
    parser.add_argument('--num', type=int, default=3000, help='choose index first !!!!')       
    parser.add_argument('--iter', type=int, default=1, help='choose index first !!!!')       
    parser.add_argument('--ee_force', type=int, default=0, help='select from {1, 2, 3}')       
    parser.add_argument('--speed', type=int, default=500, help='select from {1, 2, 3}')      
    parser.add_argument('--delay_time', type=float, default=5.0, help='select from {1, 2, 3}')      
    # //// game /////////////
    parser.add_argument('--bilateral', type=int, default=0, help='choose mode first !!!!')       
    parser.add_argument('--use_vr', type=int, default=0, help='choose mode first !!!!')       
    parser.add_argument('--game_index', type=int, default=1, help='select from {1, 2, 3}')        
    parser.add_argument('--motion_mode', type=int, default=0, help='select from {1, 2, 3}')        
    parser.add_argument('--target_index', type=int, default=1, help='select from {0, ..., 7}')        
    parser.add_argument('--target_random', type=int, default=0, help='select from {0, 1}')        
    # //// path /////////////
    parser.add_argument('--flag', type=str, default='fifth', help='choose index first !!!!')      
    parser.add_argument('--file_name', type=str, default='x_p', help='load reference trajectory !!!')       
    parser.add_argument('--root_path', type=str, default='../data/2Dletters/', help='choose mode first !!!!')   
    # //// learning /////////    
    parser.add_argument('--nb_data', type=int, default=200, help='choose index first !!!!')      
    parser.add_argument('--nb_samples', type=int, default=5, help='load reference trajectory !!!')       
    parser.add_argument('--nb_states', type=int, default=10, help='choose index first !!!!')      
    parser.add_argument('--data_name', type=str, default="iteration_learning", help='choose index first !!!!')      
    parser.add_argument('--sample_num', type=int, default=5000, help='choose index first !!!!')      
    parser.add_argument('--input_dim', type=int, default=1, help='choose index first !!!!')      
    parser.add_argument('--output_dim', type=int, default=2, help='load reference trajectory !!!')     
    parser.add_argument('--nb_prior_samples', type=int, default=10, help='choose index first !!!!')      
    parser.add_argument('--nb_posterior_samples', type=int, default=3, help='choose index first !!!!')      
    
    parser.add_argument('--save_fig', type=int, default=1, help='choose index first !!!!')     
    parser.add_argument('--fig_path', type=str, default='./figures/', help='choose index first !!!!')    
    parser.add_argument('--folder', type=str, default='./data/wrist_demo', help='choose index first !!!!')       
    parser.add_argument('--letter', type=str, default='A', help='choose mode first !!!!')    
    parser.add_argument('--flag_letter', type=str, default='', help='choose index first !!!!')   

    parser.add_argument('--dt', type=float, default=0.01, help='choose mode first !!!!')     

    args = parser.parse_args()    

    # Load data 
    # letter = 'C'   
    # datapath = '../data/2Dletters/'
    data = loadmat(args.root_path + '%s.mat' % args.letter)    
    demos = [d['pos'][0][0].T for d in data['demos'][0]]   
    # print("demos shape :", demos) 

    # font_name = 'D'  
    # import scipy.io as scio      
    # data = scio.loadmat('./path_planning/data/2Dletters/' + font_name + '.mat')
    # print("real_data :", data.keys(), data['demos'])     
    # demos = [d['pos'][0][0].T for d in data['demos'][0]]   
    # print(len(demos))    

    # Parameters
    nb_data = demos[0].shape[0]    
    print("nb_data :", nb_data)    
    nb_data_sup = 50    
    nb_samples = 5    
    dt = 0.01   
    input_dim = 1    
    output_dim = 2    
    in_idx = [0]   
    out_idx = [1, 2]    
    nb_states = args.nb_states   
    args.T = dt * args.nb_data   

    # Create time data
    demos_t = [np.arange(demos[i].shape[0])[:, None] for i in range(nb_samples)]
    # Stack time and position data
    demos_tx = [np.hstack([demos_t[i]*dt, demos[i]]) for i in range(nb_samples)]

    # Stack demos
    demos_np = demos_tx[0]
    for i in range(1, nb_samples):
        demos_np = np.vstack([demos_np, demos_tx[i]])

    X = demos_np[:, 0][:, None]    
    Y = demos_np[:, 1:]    

    # # Test data
    # Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]   

    # Test data   
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]
    nb_data_test = Xt.shape[0]
    Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])   

    # Define via-points (new set of observations)  
    if args.letter == 'B':   
        X_obs = np.array([0.0, 1., 1.9])[:, None]
        Y_obs = np.array([[-12.5, -11.5], [-0.5, -1.5], [-14.0, -7.5]])   
    elif args.letter == 'D':   
        # X_obs = np.array([1.1])[:, None]   
        # Y_obs = np.array([[8.0, 5.0]])    
        X_obs = np.array([1.99])[:, None]
        Y_obs = np.array([[-9.0, -8.0]])     
    else:  
        print("please give new observation points")   
        exit()   

    X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]   
    Y_obs_list = [Y_obs[:, i][:, None] for i in range(output_dim)]    
    print("X_obs :", X_obs.shape, X_obs_list, "Y_obs :", Y_obs.shape, Y_obs_list)    

    # GMM
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim+output_dim, in_idx=in_idx, out_idx=out_idx)
    gmr_model.init_params_kbins(demos_np.T, nb_samples=nb_samples)
    gmr_model.gmm_em(demos_np.T)

    # GMR
    mu_gmr = []   
    sigma_gmr = []   
    for i in range(Xt.shape[0]):
        mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])
        mu_gmr.append(mu_gmr_tmp)
        sigma_gmr.append(sigma_gmr_tmp)

    mu_gmr = np.array(mu_gmr)   
    sigma_gmr = np.array(sigma_gmr)   

    # #####################################################  
    # plot_word_gmm(args=args, gmr_model=gmr_model, X=X, Y=Y)   

    # plot_word_gmr(args=args, gmr_model=gmr_model, X=X, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, X_obs=X_obs, Y_obs=Y_obs)    
    # #####################################################

    if args.flag == 'generate':   
        # Train data for GPR
        X_list = [np.hstack((X, X)) for i in range(output_dim)]    
        Y_list = [Y[:, i][:, None] for i in range(output_dim)]    
        print("X :", np.hstack((X, X)).shape, "Y :", (Y[:, 0][:, None]).shape)  

        # Define GPR likelihood and kernels     
        likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" %j, variance=0.01) for j in range(output_dim)]    
        # kernel_list = [GPy.kern.RBF(1, variance=1., lengthscale=0.1) for i in range(gmr_model.nb_states)]   
        kernel_list = [GPy.kern.Matern52(1, variance=1., lengthscale=5.) for i in range(gmr_model.nb_states)]     

        # Fix variance of kernels
        for kernel in kernel_list:   
            kernel.variance.fix(1.)   
            kernel.lengthscale.constrain_bounded(0.01, 10.)   

        # Bound noise parameters   
        for likelihood in likelihoods_list:   
            likelihood.variance.constrain_bounded(0.001, 0.05)   

        # GPR model   
        K = Gmr_based_kernel(gmr_model=gmr_model, kernel_list=kernel_list)
        mf = GmrMeanMapping(2*input_dim+1, 1, gmr_model)   
        
        m = GPCoregionalizedWithMeanRegression(
                X_list, Y_list, kernel=K, 
                likelihoods_list=likelihoods_list, 
                mean_function=mf
            )   

        # Parameters optimization   
        m.optimize('bfgs', max_iters=100, messages=True)   

        # Print model parameters   
        print(m)   

        # GPR prior (no observations)
        prior_traj = []   
        prior_mean = mf.f(Xtest)[:, 0]    
        prior_kernel = m.kern.K(Xtest)    
        for i in range(args.nb_prior_samples):   
            prior_traj_tmp = np.random.multivariate_normal(prior_mean, prior_kernel)   
            prior_traj.append(np.reshape(prior_traj_tmp, (output_dim, -1)))   

        prior_kernel_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))   
        for i in range(output_dim):   
            for j in range(output_dim):    
                prior_kernel_tmp[:, :, i * output_dim + j] = prior_kernel[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
        prior_kernel_rshp = np.zeros((nb_data_test, output_dim, output_dim))   
        for i in range(nb_data_test):   
            prior_kernel_rshp[i] = np.reshape(prior_kernel_tmp[i, i, :], (output_dim, output_dim))   

        # GPR posterior -> new points observed (the training points are discarded as they are "included" in the GMM)
        m_obs = GPCoregionalizedWithMeanRegression(
                X_obs_list, Y_obs_list, kernel=K, 
                likelihoods_list=likelihoods_list, 
                mean_function=mf
            )   
        mu_posterior_tmp = m_obs.posterior_samples_f(Xtest, full_cov=True, size=args.nb_posterior_samples)   
        mu_posterior = []   
        for i in range(args.nb_posterior_samples):    
            mu_posterior.append(np.reshape(mu_posterior_tmp[:, 0, i], (output_dim, -1)))    

        # GPR prediction   
        mu_gp, sigma_gp = m_obs.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})  
        mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1)).T    
        print("mu_gp_rshp :", mu_gp_rshp.shape)   
        sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))    
        for i in range(output_dim):    
            for j in range(output_dim):     
                sigma_gp_tmp[:, :, i * output_dim + j] = sigma_gp[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
        
        sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))    
        for i in range(nb_data_test):    
            sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (output_dim, output_dim))  

        plot_word_gmr_gp(  
            args=args,    
            gmr_model=gmr_model, X=X, Y=Y,    
            Y_obs=Y_obs, mu_posterior=mu_posterior,      
            mu_gmr=mu_gmr, sigma_gmr=sigma_gmr,     
            mu_gp_rshp=mu_gp_rshp, sigma_gp_rshp=sigma_gp_rshp    
        )   