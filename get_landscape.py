import copy
import yaml
import math
import os
import re
import numpy as np
import pandas as pd

# import matplotlib.pylab as plt # for get_landscape
import matplotlib.pyplot as plt # for HDVW_csv2fig
from matplotlib import cm

import torch
import loss_landscapes

import ops.loss_landscapes as lls
import ops.tests as tests

from pyhessian import hessian
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D

def get_landscape(model_initial, model_final, criterion,
                  train_loader, save_path, DEVICE,
                  STEPS = 40):
    
    x, y = iter(train_loader).__next__()
    x, y = x.to(DEVICE), y.to(DEVICE)
    metric = loss_landscapes.metrics.Loss(criterion, x, y)
    
    ## Linear Interpolations of Loss between Two Points ##
    loss_data = loss_landscapes.linear_interpolation(model_initial, model_final,
                                                     metric, STEPS, deepcopy_model=True)
    
    plt.plot([1/STEPS * i for i in range(STEPS)], loss_data)
    plt.title('Linear Interpolation of Loss')
    plt.xlabel('Interpolation Coefficient')
    plt.ylabel('Loss')
    axes = plt.gca()
    # axes.set_ylim([2.300,2.325])
    plt.savefig(os.path.join(save_path,'loss_lin_interpolation.png'))
    plt.show()
    
    ## Planar Approximations of Loss Around a Point ##
    loss_data_fin = loss_landscapes.random_plane(model_final, metric, 10,
                                                 STEPS, normalization='filter',
                                                 deepcopy_model=True)
    
    # Loss Contours around Trained Model #
    plt.contour(loss_data_fin, levels=50)
    plt.title('Loss Contours around Trained Model')
    plt.savefig(os.path.join(save_path,'loss_contour.png'))
    plt.show()
    
    # Surface Plot of Loss Landscape #
    fig = plt.figure()
    ax = Axes3D(fig)
#     ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"].update({"linewidth":0})
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_linewidth(0.1)
        
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none', antialiased=False)
    ax.set_title('Surface Plot of Loss Landscape')
    fig.show()
    plt.savefig(os.path.join(save_path,'loss_landscape.png'))
    
def get_HDVW_landscape(model, train_loader, run_path, data_ratio, z_lim, scale = 1e-0, n = 21):
    gpu = torch.cuda.is_available()
    
    print(type(scale))
    metrics_grid = lls.get_loss_landscape( model, 1, train_loader, transform=None,
                                          kws=["pos_embed", "relative_position"],
                                          x_min=-1.0 * scale, x_max=1.0 * scale, n_x=n,
                                          y_min=-1.0 * scale, y_max=1.0 * scale, n_y=n,
                                          gpu=gpu,)
    
    metrics_dir = os.path.join(run_path,'%s_scale-%s_n-%s_loss_landscape_HDVW.csv' % (data_ratio, int(1 / scale),n))
    metrics_list = [[*grid, *metrics] for grid, metrics in metrics_grid.items()]
    tests.save_metrics(metrics_dir, metrics_list)
    
    HDVW_csv2fig(metrics_dir, z_lim, scale)
    

    
def HDVW_csv2fig(csv_path,  z_lim, scale, run_path = '', weight_decay = 0):
    if run_path == '':
        fin_idx = [m.start() for m in re.finditer('/',csv_path)][-1]
        run_path = csv_path[:fin_idx+1]
        
    # load losslandscape raw data of ResNet-50 or ViT-Ti
    names = ["x", "y", "l1", "l2", "NLL", "Cutoff1", "Cutoff2", "Acc", "Acc-90", "Unc", "Unc-90", "IoU", "IoU-90", "Freq", "Freq-90", "Top-5", "Brier", "ECE", "ECSE"]
    path =  csv_path
    data = pd.read_csv(path, names=names)
    data["loss"] = data["NLL"] + weight_decay * data["l2"]  # NLL + l2

    # prepare data
    p = int(math.sqrt(len(data)))
    shape = [p, p]
    xs = data["x"].to_numpy().reshape(shape) 
    ys = data["y"].to_numpy().reshape(shape)
    zs = data["loss"].to_numpy().reshape(shape)

    zs = zs - zs[np.isfinite(zs)].min()
    # zs[zs > 42] = np.nan

    norm = plt.Normalize(zs[np.isfinite(zs)].min(), zs[np.isfinite(zs)].max())  # normalize to [0,1]
    colors = cm.plasma(norm(zs))
    rcount, ccount, _ = colors.shape

    fig = plt.figure(figsize=(4, 4), dpi=120)
    ax = fig.gca(projection="3d")
    ax.view_init(elev=15, azim=15)  # angle

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    surf = ax.plot_surface(
        xs, ys, zs, 
        rcount=rcount, ccount=ccount,
        facecolors=colors, shade=False,
    )
    surf.set_facecolor((0,0,0,0))

    # remove white spaces
    adjust_lim = 0.8 * scale
    ax.set_xlim(-1 * adjust_lim, 1 * adjust_lim)
    ax.set_ylim(-1 * adjust_lim, 1 * adjust_lim)
    ax.set_zlim(zs.min(), z_lim ) #,0.5) #(10, 32)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    plt.savefig(csv_path[:-4] + '.png')

    plt.show()
    
    
    
def get_Hessian_eig(model, train_loader, criterion, run_path, DEVICE):
    max_eigens = []  # a list of batch-wise top-k hessian max eigenvalues
    model = model.to(DEVICE)
    
    for xs, ys in tqdm(train_loader):
        hessian_comp = hessian(model, data=(xs, ys), criterion=criterion, cuda=True)  # measure hessian max eigenvalues with NLL + L2 on data augmented (`transform`) datasets
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=5)  # collect top-5 hessian eigenvaues by using power-iteration (https://en.wikipedia.org/wiki/Power_iteration)
        max_eigens = max_eigens + top_eigenvalues  # aggregate top-5 max eigenvalues
    
    return max_eigens