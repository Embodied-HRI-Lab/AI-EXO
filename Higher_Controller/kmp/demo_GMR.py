import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)   
import numpy as np   
import matplotlib.pyplot as plt    
from scipy.io import loadmat   
from scipy import interpolate   
  
from .GMRbasedGP.utils.gmr import Gmr, plot_gmm   
from .KMP_functions import *      
# from .plot_pred import *    

import seaborn as sns    
import copy as cp   
import argparse    
import math    

# import GPy   
# from GMRbasedGP.utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
# from GMRbasedGP.utils.gmr_mean_mapping import GmrMeanMapping 
# from GMRbasedGP.utils.gmr_kernels import Gmr_based_kernel
from sklearn.mixture import BayesianGaussianMixture



def GMR_pred(  
    demos_np=None,     
    X=None,   
    Xt=None,        
    Y=None,    
    nb_data=None,       
    nb_samples=5,   
    nb_states=5,   
    input_dim=1,   
    output_dim=2    
):  
    in_idx = list(range(input_dim)) 
    out_idx = list(range(1, output_dim+1))   

    # GMM + GMR 
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim+output_dim, in_idx=in_idx, out_idx=out_idx)  
    gmr_model.init_params_kbins(demos_np.T, nb_samples=nb_samples)  
    gmr_model.gmm_em(demos_np.T)    

    # GMR  
    mu_gmr = [] 
    sigma_gmr = []  
    sigma_gmr_1 = []   
    # sigma_gmr_2 = []   
    for i in range(Xt.shape[0]):  
        mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])  
        mu_gmr.append(mu_gmr_tmp)  
        sigma_gmr.append(sigma_gmr_tmp)  
        sigma_gmr_1.append(sigma_gmr_tmp[0, 0])  
        # sigma_gmr_2.append(sigma_gmr_tmp[1, 1])  
    
    mu_gmr = np.array(mu_gmr)  
    sigma_gmr = np.array(sigma_gmr)    
    return mu_gmr, sigma_gmr, gmr_model  


def KMP_pred(
    Xt=None,   
    mu_gmr=None,   
    sigma_gmr=None,    
    viaNum=2,    
    viaFlag=np.ones(2),      
    via_time=None,     
    via_points=None,   
    via_var_list=None,   
    dt=0.01,    
    len=None,     
    lamda_1=0.01,      
    lamda_2=0.6,       
    kh=6, 
    output_dim=2,    
    dim=2      
): 
    # ////////////////////////////////////////////////    
    refTraj = {}    
    refTraj['t'] = Xt    
    refTraj['mu'] = mu_gmr     
    refTraj['sigma'] = sigma_gmr      

    ori_refTraj = {}   
    ori_refTraj['t'] = cp.deepcopy(Xt)    
    ori_refTraj['mu'] = cp.deepcopy(mu_gmr)     
    ori_refTraj['sigma'] = cp.deepcopy(sigma_gmr)     

    #  KMP parameters
    len = Xt.shape[0]    
    
    # update reference trajectory using desired points
    newRef = refTraj      
    newLen = len      

    # # insert points     
    # # print("newLen :", newLen)     
    # # print("via_point_before:", newRef['mu'].shape)     
    for viaIndex in range(viaNum):     
        # print("viaIndex :", viaIndex)     
        via_var = via_var_list[viaIndex]  
        newRef, newLen = kmp_insertPoint(newRef, newLen, via_time[viaIndex], via_points[viaIndex, :], via_var)   
    print("newNum, newLen", viaNum, newLen)  

    # print("via_point_after:", newRef['mu'].shape)   

    # Prediction using kmp   
    # Kinv = kmp_estimateMatrix_mean(newRef, newLen, kh, lamda, dim) 
    Kinv_1, Kinv_2 = kmp_estimateMatrix_mean_var(newRef, newLen, kh, lamda_1, lamda_2, dim, output_dim)  
    # print("Kinv_1 :", Kinv_1.shape)   

    uncertainLen = 0.0 * len    
    totalLen     = int(len + uncertainLen)      
    
    # kmpPredTraj  
    kmpPredTraj = {}     
    new_time_t  = np.zeros((totalLen, 1))   
    new_mu_t    = np.zeros((totalLen, output_dim))    
    new_sigma_t = np.zeros((totalLen, output_dim, output_dim))       

    # t, mu, sigma    
    for index in range(totalLen):     
        t = index * dt   
        # mu = kmp_pred_mean(t, newRef, newLen, kh, Kinv, dim)  

        pred_mu, pred_sigma = kmp_pred_mean_var(t, newRef, newLen, kh, Kinv_1, Kinv_2, lamda_2, dim, output_dim)    
        new_time_t[index, 0] = t   
        new_mu_t[index, :] = pred_mu.T    
        # print("pred_sigma :", pred_sigma)      
        new_sigma_t[index, :, :] = pred_sigma          

    kmpPredTraj['t']     = new_time_t       
    kmpPredTraj['mu']    = new_mu_t          
    kmpPredTraj['sigma'] = new_sigma_t         

    return ori_refTraj, refTraj, kmpPredTraj        

  
def GMR_GP(
    X=None,  
    Y=None,  
    X_t=None,    
    X_obs=None,   
    Y_obs=None,  
    gmr_model=None,    
    mu_gmr=None,    
    sigma_gmr=None,    
    nb_prior_samples=None, 
    nb_posterior_samples=None,  
    input_dim=1, 
    output_dim=2    
): 
    # Train data for GPR
    X_list = [np.hstack((X, X)) for i in range(output_dim)]
    Y_list = [Y[:, i][:, None] for i in range(output_dim)]  

    # Test data
    # Xt = dt * np.arange(demos[0].shape[0] + nb_data_sup)[:, None]
    nb_data_test = Xt.shape[0]  
    Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])

    # obs list 
    X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]    
    Y_obs_list = [Y_obs[:, i][:, None] for i in range(output_dim)]     

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

    m = GPCoregionalizedWithMeanRegression(X_list, Y_list, kernel=K, likelihoods_list=likelihoods_list, mean_function=mf)

    # Parameters optimization  
    m.optimize('bfgs', max_iters=100, messages=True)  

    # Print model parameters  
    print(m)

    # GPR prior (no observations)
    prior_traj = []
    prior_mean = mf.f(Xtest)[:, 0]  
    prior_kernel = m.kern.K(Xtest)   
    for i in range(nb_prior_samples):
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
    m_obs = GPCoregionalizedWithMeanRegression(X_obs_list, Y_obs_list, kernel=K, likelihoods_list=likelihoods_list, mean_function=mf)
    mu_posterior_tmp = m_obs.posterior_samples_f(Xtest, full_cov=True, size=nb_posterior_samples)

    mu_posterior = []
    for i in range(nb_posterior_samples):
        mu_posterior.append(np.reshape(mu_posterior_tmp[:, 0, i], (output_dim, -1)))

    # GPR prediction
    mu_gp, sigma_gp = m_obs.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})

    mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1)).T  

    sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
    for i in range(output_dim):  
        for j in range(output_dim):  
            sigma_gp_tmp[:, :, i * output_dim + j] = sigma_gp[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
    sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))
    for i in range(nb_data_test):  
        sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (output_dim, output_dim))

    return mu_gp_rshp, sigma_gp_rshp, Y_obs, mu_posterior   


def Avoid_obs():  
    # Create a 3-dimensional NURBS Curve
    curve = NURBS.Curve()

    # Set degree
    curve.degree = 2

    # Set control points
    curve.ctrlpts = [[0, 0], [-10, 0], [-10, 10], [-10,20], [0,20], [10,20],[10, 10], [10,0],[0,0]]

    # circle weight
    w = math.sqrt(2)/2  

    # Set Weigths vector
    curve.weights = [1, w, 1, w, 1, w, 1, w, 1]  

    # Set knot vector
    curve.knotvector = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]

    # Set evaluation delta (controls the number of curve points)
    curve.delta = 0.01  

    # Plot the control point polygon and the evaluated curve
    curve.vis = VisMPL.VisCurve2D()

    # Don't pop up the plot window, instead save it as a PDF file
    curve.render(filename="circle-curve2d.pdf", plot=True)  
    

def Bayesian_GMM(
    ori_path=None, 
):  
    # Parameters of the dataset
    random_state, n_components, n_features = 2, 3, 2   

    covars = np.array(  
        [[[0.7, 0.0], [0.0, 0.1]], [[0.5, 0.0], [0.0, 0.1]], [[0.5, 0.0], [0.0, 0.1]]]
    )
    
    samples = np.array([200, 500, 200])
    means = np.array([[0.0, -0.70], [0.0, 0.0], [0.0, 0.70]])

    # mean_precision_prior= 0.8 to minimize the influence of the prior
    estimators = [
        (
            "Finite mixture with a Dirichlet distribution\nprior and " r"$\gamma_0=$",
            BayesianGaussianMixture(
                weight_concentration_prior_type="dirichlet_distribution",
                n_components=2 * n_components,
                reg_covar=0,
                init_params="random",
                max_iter=1500,
                mean_precision_prior=0.8,
                random_state=random_state,
            ),
            [0.001, 1, 1000],
        ),
        (
            "Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
            BayesianGaussianMixture(
                weight_concentration_prior_type="dirichlet_process",
                n_components=2 * n_components,
                reg_covar=0,
                init_params="random",
                max_iter=1500,
                mean_precision_prior=0.8,
                random_state=random_state,
            ),
            [1, 1000, 100000],
        ),
    ]

    # Generate data
    rng = np.random.RandomState(random_state)
    X = np.vstack(  
        [
            rng.multivariate_normal(means[j], covars[j], samples[j])
            for j in range(n_components)
        ]   
    )   
    y = np.concatenate([np.full(samples[j], j, dtype=int) for j in range(n_components)])  

    # Plot results in two different figures
    for title, estimator, concentrations_prior in estimators:  
        plt.figure(figsize=(4.7 * 3, 8))  
        plt.subplots_adjust(
            bottom=0.04, top=0.90, hspace=0.05, wspace=0.05, left=0.03, right=0.99
        )

        gs = gridspec.GridSpec(3, len(concentrations_prior))  
        for k, concentration in enumerate(concentrations_prior):  
            estimator.weight_concentration_prior = concentration
            estimator.fit(X)  
            plot_results(
                plt.subplot(gs[0:2, k]),
                plt.subplot(gs[2, k]),
                estimator,  
                X,  
                y,  
                r"%s$%.1e$" % (title, concentration),
                plot_title=k == 0,
            )

    plt.show()


def cal_error(error_data, sigma_data, alpha_t=0.1, beta_t=0.1):   
    stiff_data = []  
    print("shape :", sigma_data[0, ::].shape)
    for i in range(error_data.shape[0]):   
        stiff = (1 - math.exp(-alpha_t * np.linalg.norm(sigma_data[i, ::], ord=2))) * beta_t * error_data[i, :]  
        stiff_data.append(stiff)    
        # path_name = path_name_list[i]    
        # path_angle = np.loadtxt(path_name, delimiter=',', skiprows=1)    
        # mse = mean_squared_error(path_angle[:, 8], path_angle[:, 10])    
        # print("mse :", math.sqrt(mse))    

    scale = 1/50.0   
    stiff_data = np.array(stiff_data)    
    damping_data = scale * stiff_data    
    
    return stiff_data, damping_data  


def cal_stiff_scale(sigma_demo_data, alpha_t=0.1, beta_t=0.1):     
    stiff_scale = []    
    for i in range(sigma_demo_data.shape[0]):     
        stiff = (1 - beta_t) * math.exp(-alpha_t * np.linalg.norm(sigma_demo_data[i, ::], ord=2)) + beta_t    
        stiff_scale.append(stiff)    
    
    stiff_scale = np.array(stiff_scale)     
    
    return stiff_scale     


def cal_iterp_data(stiff_data, N=None):   
    t = np.linspace(0, 2.0, N)   
    time_t = np.linspace(0, 2.0, 200)    
    
    print("time_t :", stiff_data.shape)       
    # print("t :", t)   
    stiff_interp = []   
    for i in range(stiff_data.shape[1]):     
        int_stiff = interpolate.interp1d(time_t, stiff_data[:, i], kind='linear')  
        # int_damping = interpolate.interp1d(time_t, damping_data, kind='linear') 
        
        stiff_interp_data = int_stiff(t)   
        # damping_interp_data = int_damping(t)    
        stiff_interp.append(stiff_interp_data)    
     
    return np.squeeze(np.array(stiff_interp))     


def cal_iterp_scale(stiff_scale, N=None):   
    t = np.linspace(0, 2.0, N)   
    time_t = np.linspace(0, 2.0, 200)    
    
    # print("time_t :", time_t, stiff_data[:, 0].shape)    
   
    int_stiff = interpolate.interp1d(time_t, stiff_scale, kind='linear')  
    # int_damping = interpolate.interp1d(time_t, damping_data, kind='linear') 
    
    scale_interp_data = int_stiff(t)   
     
    return np.array(scale_interp_data)      


if __name__ == "__main__":   
    parser = argparse.ArgumentParser()   

    parser.add_argument('--save_fig', type=bool, default=True, help='choose index first !!!!')  
    parser.add_argument('--data_name', type=str, default="tracking_epi_circ", help='data name!!!!')  
    parser.add_argument('--file_path', type=str, default='wrist', help='choose index first !!!!')  
    parser.add_argument('--root_path', type=str, default='data/', help='choose index first !!!!')  

    parser.add_argument('--nb_samples', type=int, default=5, help='choose mode first !!!!')  
    parser.add_argument('--nb_states', type=int, default=10, help='choose mode first !!!!')   
    parser.add_argument('--nb_data', type=int, default=200, help='choose mode first !!!!')  

    args = parser.parse_args()   
    
    # Avoid_obs()    

    # Load data    
    datapath = args.root_path + args.file_path + '/'  
    data = np.load(datapath + '%s.npy' % args.data_name, allow_pickle=True)       
    print("data :", data.shape)   
    
    ref_data = np.load(args.root_path + args.file_path + '/' + '%s_des.npy' % args.data_name, allow_pickle=True)

    # Parameters
    # nb_data = demos.shape[2]   
    
    nb_data = 200     
    nb_samples = args.nb_samples      
    
    # demos = data.reshape((nb_data, nb_samples, 3))  
    demos = data    
    print("data shape :", demos.shape)     

    nb_data_sup = 0      
    dt = 0.01     
    demodura = dt * nb_data      
    print("demodura :", demodura)     
    
    # model parameter 
    input_dim = 1   
    # output_dim = 2     
    output_dim = 3     
    # in_idx = [0]       
    # out_idx = [1, 2, 3, 4]       
    # nb_states = 6       
    # dim = 2       
    
    # Create time data     
    demos_t = [np.arange(nb_data)[:, None] for i in range(nb_samples)]  
    print("demos_t :", np.array(demos_t[0]).shape)     

    # # Stack time and position data
    # demos_tx = [np.hstack([demos_t[i] * dt, demos[i, 1, :][:, None], demos[i, 0, :][:, None]]) for i in range(nb_samples)]
    # print("demos_tx :", np.array(demos_tx).shape) 
    
    # Stack time and position data  
    demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 0][:, None], demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
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

    # # Define via-points (new set of observations)    
    # X_obs = np.array([0.0, 1.4])[:, None]   
    # Y_obs = np.array([[0.0, -20.0], [-10.0, 12.0]])   
    
    # Bayesian_GMM(
    #     ori_path=None 
    # )

    # plot_epi_data(
    #     font_name=args.data_name,   
    #     nb_samples=5,   
    #     nb_dim=5,   
    #     nb_data=200,   
    #     X=X,   
    #     Y=Y    
    # )   

    mu_gmr, sigma_gmr, gmr_model = GMR_pred(
        demos_np=demos_np,    
        X=X,   
        Xt=Xt,  
        Y=Y,    
        nb_data=args.nb_data,    
        nb_samples=args.nb_samples,     
        nb_states=args.nb_states,     
        input_dim=input_dim,    
        output_dim=output_dim,    
        data_name=args.data_name     
    )   
    
    phi_range = [0, 2*np.pi]   
    theta_range = [7/8*np.pi, 7/8*np.pi]   
    sample_num_1 = [1, 200]  
    theta = np.linspace(theta_range[0], theta_range[1], sample_num_1[0])       
    phi = np.linspace(phi_range[0], phi_range[1], sample_num_1[1])      
    t, p = np.meshgrid(theta, phi)    
    
    r = 70     
    x = r*np.sin(t)*np.cos(p)      
    y = r*np.sin(t)*np.sin(p)     
    z = r*np.cos(t)     
    
    x = x.flatten()     
    y = y.flatten()     
    z = z.flatten()     

    ref_x = np.vstack((x, y, z))    
    print("ref_x shape :", ref_x.shape)  
    
    ref_data = ref_data[:nb_data, :]  
    plot_raw_data(font_name=args.data_name, nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)    
    # plot_GMM_raw_data(font_name=data_name, nb_samples=nb_samples, nb_data=nb_data, Y=Y, gmr_model=gmr_model)   
    # plot_mean_var(font_name=data_name, nb_samples=nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_x=ref_x)   
    # plot_mean_var_fig(font_name=data_name, nb_samples=nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_x=ref_x)   
    
    error_data = np.array(mu_gmr) - ref_data      
    # print("error_data :", error_data)     
    
    plot_error_data(font_name=args.data_name, X=X, nb_data=nb_data, error_data=error_data)    
    
    stiff_data, damping_data = cal_error(error_data, sigma_gmr, alpha_t=0.1, beta_t=0.1)    
    
    stiff_scale = cal_stiff_scale(sigma_gmr, alpha_t=0.2, beta_t=0.75)    
    
    # stiff_interp_data, damping_interp_data = cal_iterp_data(Xt, stiff_data, damping_data, N=5000)  
    
    stiff_interp = cal_iterp_data(stiff_data, N=5000)    
    print("shape :", stiff_interp.shape)  
    
    damping_interp = cal_iterp_data(damping_data, N=5000)    
    print("shape :", damping_interp.shape)    
    
    scale_interp = cal_iterp_scale(stiff_scale, N=5000)    
    print("shape :", damping_interp.shape)    
    
    plot_stiff_damping(font_name=args.data_name, stiff_data=stiff_interp, damping_data=damping_interp, stiff_scale=scale_interp)   
    
    # for i in range(200):     
    #     ref_x[i, 0] = 70      
    
    # # ////////////// via points //////////////
    # # viaFlag = np.array([1, 1, 1, 1])  
    # # viaNum = 4  
    
    # # // wrist robot   
    # viaNum = 2   
    # viaFlag = np.ones(viaNum)    
    # via_time = np.zeros(viaNum)       
    # via_points = np.zeros((viaNum, output_dim))   
    
    # via_time[0] = dt    
    # via_points[0, :] = np.array([0.0, -20.0])     
    # via_time[1] = 1.  
    # via_points[1, :] = np.array([0.0, 20.0])          

    # via_var = 1E-6 * np.eye(output_dim)    
    # # via_var = 1E-6 * np.eye(4)  
    # # via_var[2, 2] = 1000   
    # # via_var[3, 3] = 1000   
    
    # ori_refTraj, refTraj, kmpPredTraj = KMP_pred(
    #     X_t=Xt,  
    #     mu_gmr=mu_gmr,   
    #     sigma_gmr=sigma_gmr,   
    #     viaNum=2,    
    #     viaFlag=viaFlag,      
    #     via_time=via_time,     
    #     via_points=via_points,     
    #     dt=0.01, 
    #     len=200,     
    #     lamda_1=1,      
    #     lamda_2=6,          
    #     kh=6,   
    #     output_dim=output_dim,    
    #     dim=1,      
    #     data_name=args.data_name       
    # )  
    
    # # gmr = np.zeros((4, newLen))   
    # # kmp = np.zeros((4, newLen))   
    # # sigma_kmp = np.zeros((4, 4, newLen)) 
    # # for i in range(newLen):    
    # #     gmr[:, i] = refTraj['mu'][i]    
        
    # #     kmp[:, i] = kmpPredTraj['mu'][i]       
    # #     sigma_kmp[:, :, i] = kmpPredTraj['sigma'][i]   

    # # print("sigma_kmp :", sigma_kmp)    
    # plot_via_points(  
    #     font_name=args.data_name,   
    #     nb_posterior_samples=viaNum, 
    #     via_points=via_points,   
    #     mu_gmr=ori_refTraj['mu'],   
    #     pred_gmr=kmpPredTraj['mu'], 
    #     sigma_gmr=ori_refTraj['sigma'], 
    #     sigma_kmp=kmpPredTraj['sigma']  
    # )    

    # plot_mean_var(
    #     font_name=letter, 
    #     nb_samples=nb_samples, 
    #     nb_data=nb_data, 
    #     Xt=kmpPredTraj['t'], 
    #     Y=Y, 
    #     mu_gmr=kmpPredTraj['mu'], 
    #     sigma_gmr=kmpPredTraj['sigma']
    # )   
    
    # mu_gp_rshp, sigma_gp_rshp, Y_obs, mu_posterior = GMR_GP(
    #     X=X,     
    #     Y=Y,    
    #     X_t=Xt,     
    #     X_obs=X_obs,    
    #     Y_obs=Y_obs,    
    #     gmr_model=gmr_model,      
    #     mu_gmr=mu_gmr,    
    #     sigma_gmr=sigma_gmr,    
    #     nb_prior_samples=5, 
    #     nb_posterior_samples=3,  
    #     input_dim=1, 
    #     output_dim=2    
    # )  
    
    
    # plot_poster_samples(
    #     mu_gmr=mu_gmr,  
    #     mu_gp_rshp=mu_gp_rshp,   
    #     sigma_gp_rshp=sigma_gp_rshp,     
    #     Y_obs=Y_obs,  
    #     nb_posterior_samples=3,    
    #     mu_posterior=mu_posterior    
    # )   

    # ////////////// via points //////////////
    # viaFlag = np.array([1, 1, 1, 1])  
    # viaNum = 4  

    # via_time = np.zeros(viaNum)   
    # via_point = np.zeros((viaNum, 4))   
    # via_time[0] = dt   
    # via_point[0,:] = np.array([8, 10, -50, 0])   
    # via_time[1] = 0.25    
    # via_point[1,:] = np.array([-1, 6, -25, -40])   
    # via_time[2] = 1.2   
    # via_point[2,:] = np.array([8, -4, 30, 10])     
    # via_time[3] = 2   
    # via_point[3, :] = np.array([-3, 1, -10, 3])  

    # via_var = 1E-6 * np.eye(viaNum, dtype='float')   
    # via_var(3,3) = 1000   
    # via_var(4,4) = 1000   

    # //// B 
    # viaFlag = np.array([1, 1, 1])  # determine which via-points are used
    # viaNum = 3  
    # via_time = np.zeros(viaNum)   
    # via_point = np.zeros((viaNum, 4))   

    # via_time[0] = dt
    # via_point[0, :] = np.array([-12, -12, 0, 0])  # format:[2D-pos 2D-vel]
    # via_time[1] = 1
    # via_point[1, :] = np.array([0, -1, 0, 0])     
    # via_time[2] = 1.99
    # via_point[2, :] = np.array([-14, -8, 0, 0])  

    # via_var = 1E-6 * np.eye(4)  
    # via_var[2, 2] = 1000  
    # via_var[3, 3] = 1000   

    # //// F
    # viaFlag = np.array([1, 1])  
    # viaNum = 2 
    # via_time = np.zeros(viaNum)   
    # via_point = np.zeros((viaNum, 4))   
    
    # via_time[0] = dt    
    # via_point[0, :] = np.array([0.0, 0.0, 0, 0])    
    # via_time[1] = 1.  
    # via_point[1, :] = np.array([-8, -10, 0, 0])    

    # via_var=1E-6 * np.eye(4)
    # via_var[2, 2] = 1000 
    # via_var[3, 3] = 1000    
    
    # // wrist robot 
    # viaFlag = np.array([1, 1])  
    # viaNum = 2   
    # via_time = np.zeros(viaNum)   
    # via_point = np.zeros((viaNum, output_dim))   
    
    # via_time[0] = dt    
    # via_point[0, :] = np.array([0.0, -20.0])     
    # via_time[1] = 1.  
    # via_point[1, :] = np.array([0.0, 20.0])      

    # via_var=1E-6 * np.eye(2)  
    # # via_var[2, 2] = 1000 
    # # via_var[3, 3] = 1000    
    
# # Load data
# file_name = 'data/' 

# letter = 'B' 
# datapath = file_name + '2Dletters/'   
# data = loadmat(datapath + '%s.mat' % letter)   

# demos_pos = [d['pos'][0][0].T for d in data['demos'][0]]   
# demos_vel = [d['vel'][0][0].T for d in data['demos'][0]]    
# # print("demos_pos :", demos_pos.shape)     
# # print("demos_vel :", demos_vel.shape)     
# # demos = np.vstack((demos_pos, demos_vel))     
# demos = demos_pos  

# # Parameters
# nb_data = demos[0].shape[0]   
# print("nb_data :", demos[0].shape)     
# nb_data_sup = 0    
# nb_samples = 5   
# dt = 0.01   
# demodura = dt * nb_data    

# demos_pos = [d['pos'][0][0].T for d in data['demos'][0]]   
# demos_vel = [d['vel'][0][0].T for d in data['demos'][0]]    
# # print("demos_pos :", demos_pos.shape)     
# # print("demos_vel :", demos_vel.shape)     
# # demos = np.vstack((demos_pos, demos_vel))     
# demos = demos_pos  

# # /////////////////////////////////////////////////////////////////////////
# refTraj = {}   
# refTraj['t'] = Xt   
# refTraj['mu'] = mu_gmr   
# refTraj['sigma'] = sigma_gmr   

# ori_refTraj = {}
# ori_refTraj['t'] = cp.deepcopy(Xt)   
# ori_refTraj['mu'] = cp.deepcopy(mu_gmr)    
# ori_refTraj['sigma'] = cp.deepcopy(sigma_gmr)    

# #  KMP parameters
# dt = 0.01   
# len = int(demodura/dt)    
# lamda_1 = 1    
# lamda_2 = 60   
# kh = 60  

# # newRef=refTraj    
# # newLen=len    

# # ////////////// via points //////////////
# # viaFlag = np.array([1, 1, 1, 1])  
# # viaNum = 4  

# # via_time = np.zeros(viaNum)   
# # via_point = np.zeros((viaNum, 4))   
# # via_time[0] = dt   
# # via_point[0,:] = np.array([8, 10, -50, 0])   
# # via_time[1] = 0.25    
# # via_point[1,:] = np.array([-1, 6, -25, -40])   
# # via_time[2] = 1.2   
# # via_point[2,:] = np.array([8, -4, 30, 10])     
# # via_time[3] = 2   
# # via_point[3, :] = np.array([-3, 1, -10, 3])  

# # via_var = 1E-6 * np.eye(viaNum, dtype='float')   
# # # via_var(3,3) = 1000   
# # # via_var(4,4) = 1000   

# # # # //// B 
# # viaFlag = np.array([1, 1, 1])  # determine which via-points are used
# # viaNum = 3  
# # via_time = np.zeros(viaNum)   
# # via_point = np.zeros((viaNum, 4))   

# # via_time[0] = dt
# # via_point[0, :] = np.array([-12, -12, 0, 0])  # format:[2D-pos 2D-vel]
# # via_time[1] = 1
# # via_point[1, :] = np.array([0, -1, 0, 0])     
# # via_time[2] = 1.99
# # via_point[2, :] = np.array([-14, -8, 0, 0])  

# # via_var = 1E-6 * np.eye(4)  
# # via_var[2, 2] = 1000  
# # via_var[3, 3] = 1000   

# viaFlag = np.array([1, 1])  
# viaNum = 2 
# via_time = np.zeros(viaNum)   
# via_point = np.zeros((viaNum, 4))   
   
# via_time[0] = dt    
# via_point[0, :] = np.array([0, 0, 0, 0])    
# via_time[1] = 1   
# via_point[1, :] = np.array([-4, -8, 0, 0])    

# via_var=1E-6 * np.eye(4) 
# via_var[0, 0] = 10 
# via_var[1, 1] = 10  
# via_var[2, 2] = 1000 
# via_var[3, 3] = 1000    

# # update reference trajectory using desired points
# newRef = refTraj     
# newLen = int(len)      

# # insert points   
# # print("newLen :", newLen)    
# # print("via_point_before:", newRef['mu'].shape)     
# for viaIndex in range(viaNum):     
#     print("viaIndex :", viaIndex)     
#     if (viaFlag[viaIndex]==1):        
#         newRef, newLen = kmp_insertPoint(newRef, newLen, via_time[viaIndex], via_point[viaIndex, :], via_var)   

# print("newRef", newRef)  
# print("newNum", newLen)  
# # print("via_point_after:", newRef['mu'].shape)   

# # Prediction using kmp   
# Kinv = kmp_estimateMatrix_mean(newRef, newLen, kh, lamda_1, dim) 
# # Kinv_1, Kinv_2 = kmp_estimateMatrix_mean_var(newRef, newLen, kh, lamda_1, lamda_2, dim)  
# # print("Kinv_1 :", Kinv_1.shape)   

# uncertainLen = 0.0 * len   
# totalLen = int(len + uncertainLen)     
  
# # kmpPredTraj
# kmpPredTraj = {}
# new_time_t = np.zeros((totalLen, 1))   
# new_mu_t = np.zeros((totalLen, 4))   
# new_sigma_t = np.zeros((totalLen, 4, 4))    

# # t, mu, sigma 
# for index in range(totalLen):   
#     t = index * dt   
#     mu = kmp_pred_mean(t, newRef, newLen, kh, Kinv, dim)  
#     # mu, sigma = kmp_pred_mean_var(t, newRef, newLen, kh, Kinv_1, Kinv_2, lamda_2, dim)  
#     new_time_t[index, 0] = t 
#     new_mu_t[index, :] = mu.T 
#     print("mu :", mu)
#     # new_sigma_t[index, :, :] = sigma  

# kmpPredTraj['t'] = new_time_t      
# kmpPredTraj['mu'] = new_mu_t          
# kmpPredTraj['sigma'] = new_sigma_t      

# # gmr = np.zeros((4, newLen))   
# # kmp = np.zeros((4, newLen))   
# # sigma_kmp = np.zeros((4, 4, newLen)) 
# # for i in range(newLen):    
# #     gmr[:, i] = refTraj['mu'][i]    
       
# #     kmp[:, i] = kmpPredTraj['mu'][i]       
# #     sigma_kmp[:, :, i] = kmpPredTraj['sigma'][i]   

# # print("sigma_kmp :", sigma_kmp)    
# plot_via_points(
#     font_name=data_name, 
#     nb_posterior_samples=viaNum, 
#     via_points=via_point, 
#     mu_gmr=ori_refTraj['mu'], 
#     pred_gmr=kmpPredTraj['mu'], 
#     sigma_gmr=ori_refTraj['sigma'], 
#     sigma_kmp=kmpPredTraj['sigma']
# )  