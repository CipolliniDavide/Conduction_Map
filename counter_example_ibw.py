import networkx as nx
from scipy.optimize import curve_fit
from networkx.algorithms.distance_measures import resistance_distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from plot_help_functions import plot_network, plot_imshow, histogram
import random
from networkx import grid_graph
from neo import io as nio
import argparse
import os
import copy


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    # print(domain)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def define_grid_graph_2(rows, cols, seed):
    '''
    Reference : Kevin Montano et al 2022 Neuromorph. Comput. Eng. 2 014007
    Code based on https://github.com/ MilanoGianluca/Grid-Graph_Modeling_Memristive_Nanonetworks
    :param rows:
    :param cols:
    :param seed:
    :return:
    '''
    ##define a grid graph

    Ggrid = grid_graph(dim=[rows, cols])
    random.seed(seed)
    ##define random diagonals
    for r in range(rows - 1):
        for c in range(cols - 1):
            k = random.randint(0, 1)
            if k == 0:
                Ggrid.add_edge((c, r), (c + 1, r + 1))
            else:
                Ggrid.add_edge((c + 1, r), (c, r + 1))

    ##define a graph with integer nodes and positions of a grid graph
    G = nx.convert_node_labels_to_integers(Ggrid, first_label=0, ordering='default', label_attribute='coord')
    #import itertools
    #grid = list(itertools.product(range(nrows), range(ncols)))
    #coordinates = [(n, {'coord': grid[n]}) for n in G.nodes]
    #nx.set_node_attributes(G, values=dict(coordinates), name='coord')
    return G


def add_ground_node(G, GROUND_NODE_X = 480, GROUND_NODE_Y = 250, add_x=100, add_y=0):
    ground_node = (G.number_of_nodes(), {'coord': (GROUND_NODE_X + add_x, GROUND_NODE_Y + add_y)})
    connected_to_electrode = [(node, ground_node[0]) for node, feat in G.nodes(data=True) if
                              feat['coord'][0] > GROUND_NODE_X]
    G.add_nodes_from([ground_node])
    try:
        weights = [w['weight'] for u, v, w in G.edges(data=True)]
        w_gnd = np.max(weights)
    except:
        w_gnd = 1
    G.add_edges_from(connected_to_electrode, weight=w_gnd)


def effective_resistence(G, gnd):
    # Invert_weight=True if 'weight' is a resistance; else set False if it is a conductance
    eff_res = [resistance_distance(G=G, nodeA=node_name, nodeB=gnd, weight='weight', invert_weight=True)
               for node_name in G.nodes() if node_name != gnd]
               #for node_name in range(G.number_of_nodes()-1)]
    return np.asarray(eff_res)

def set_new_weights(G, new_weights, key='weight'):
    edge_list = [(n1, n2) for n1, n2, weight in list(G.edges(data=True))]
    #edge_weight_list = [weight['weight'] for n1, n2, weight in list(G.edges(data=True))]
    edge_list_with_attr = [(edge[0], edge[1], {key: w}) for (edge, w) in zip(edge_list, new_weights)]
    G.add_edges_from(edge_list_with_attr)


def init_G(i, *par):

    weights = par
    set_new_weights(G, weights)
    R = effective_resistence(G, gnd=gnd)
    # Effective Conductance
    Y = 1/R

    return Y

def apply_contrast(image_gray, perc_min=2, perc_max=98):
    '''
    Apply contrast enhancement and scale values in range [0, 1]
    :param image_gray: original array
    :param perc_min:
    :param perc_max:
    :return: array after contrast and normalize between 0 and 1 is applied to it
    '''
    pixvals = image_gray
    minval = np.percentile(pixvals, perc_min)
    maxval = np.percentile(pixvals, perc_max)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 1
    #plt.imshow(pixvals)
    return pixvals

def reduce_grid(image_gray, nrows, ncols, scale_range=None):
    l = np.array_split(image_gray, nrows, axis=0)
    new_l = []
    for a in l:
        l = np.array_split(a, ncols, axis=1)
        new_l += l
    #for l in new_l:
    #    print(l.shape)
    int_split = [l.mean() for l in new_l]
    int_split = np.reshape(int_split, newshape=(nrows, ncols), order='F')
    int_split = np.flip(int_split, axis=1)
    if scale_range:
        int_split = scale(int_split, scale_range)
    return int_split


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fit', '--do_fit', type=int, default=0)
    parser.add_argument('-nr', '--nrows', type=int, default=15)
    parser.add_argument('-nc', '--ncols', type=int, default=15)
    parser.add_argument('-sp', '--save_path', type=str, default='./Output/')
    parser.add_argument('-lp', '--load_path', type=str, default='./JR38_CoCr_6_70000.ibw')
    args = parser.parse_args()

    # Create './Output' folder
    ensure_dir(args.save_path)
    save_name = '{:s}counterEx_ibw_'.format(args.save_path)

    # Load InPlane sample
    r = nio.IgorIO(filename=args.load_path)
    current_map = np.array(r.read_analogsignal())[..., 3]
    current_map = np.rot90(current_map, 1)

    # Pre-processing to produce an irregular distribution of effective conductance
    # Contrast enhancement and scaling in range [0,1]
    cmap_contrast = apply_contrast(image_gray=current_map, perc_min=2, perc_max=99)
    # We parcel the original sample and scale the values between [1e-3, 1] [A]
    cmap_grid_reduced = reduce_grid(image_gray=cmap_contrast, nrows=args.nrows, ncols=args.ncols,
                                    scale_range=(1e-3, 1))

    # Save and Plot the preprocessing steps in the './Output' folder
    np.save(arr=cmap_grid_reduced, file='{:s}sample_reduced.npy'.format(args.save_path))

    plot_imshow(array=current_map, title='IP sample', cb_label='Current [A]', last_ticks_num=1,
                save_fold='{:s}IP_sample.png'.format(args.save_path))
    plot_imshow(array=cmap_contrast, title='IP sample after contrast enhancement', cb_label='', last_ticks_num=None,
                save_fold='{:s}IP_sample_afterContrast.png'.format(args.save_path))
    plot_imshow(array=np.rot90(cmap_grid_reduced, 1), title='IP sample parceled in {:d}x{:d} patches'.format(args.nrows, args.ncols),
                cb_label='',
                last_ticks_num=None,
                save_fold='{:s}IP_sample_reduced.png'.format(args.save_path))

    # We define an arbitrary distribution of currents over the nodes.
    node_current_distribution = cmap_grid_reduced.reshape(-1)

    # Create Grid graph
    G = define_grid_graph_2(args.nrows, args.ncols, seed=1)
    gnd = G.number_of_nodes()
    add_ground_node(G, GROUND_NODE_X=args.nrows-2, GROUND_NODE_Y=args.ncols//2, add_x=4)

    # Plot desired distribution of current over the nodes
    labels = [(gnd, 'Pt\nElectrode')]
    plot_network(G=G, figsize=(8, 6), labels=labels, node_color=list(node_current_distribution) + [0], show=False,
                 save_fold='./{:s}_desired_distribution'.format(save_name), vmin=node_current_distribution.min(),
                 cb_nodes_lab='Effective Conductance [a.u.]',
                 #cb_edge_lab='Edge Resistance [$\\mathregular{\Omega}$]',
                 # title='PearsonCorr = {:.2f}, pval={:.5}'.format(corr, p)
                 )

    # Fit Weights
    if args.do_fit == 1:
        # p0: initial distribution of resistances
        p0 = np.random.uniform(low=1e-16, high=1, size=G.number_of_edges())
        set_new_weights(G, p0)
        bounds = (1e-16, 50)
        best_val, cov = curve_fit(f=init_G, xdata=np.arange(G.number_of_nodes()-1),
                                  ydata=node_current_distribution,
                                  p0=p0, method='trf', bounds=bounds, maxfev=400)
        np.save(file='./{:s}.npy'.format(save_name), arr=best_val)
        np.save(file='./{:s}.npy'.format(save_name), arr=best_val)

    ####################################################################################################################
    ####################################################################################################################

    # Load optimized resistances of the edges
    best_resistance = np.load('{:s}.npy'.format(save_name))
    histogram(values=best_resistance, n_bins=20, xlabel='Edge resistance [a.u.]', ylabel='#',
              save_name='resistances_hist', save_fold=args.save_path)

    # Set optimized resistances to a new graph
    G_optim = copy.deepcopy(G)
    set_new_weights(G_optim, best_resistance)
    # Measure the effective resistance between each node in the graph and the gnd
    R = effective_resistence(G_optim, gnd=gnd)
    node_eff_conductance = 1 / R

    corr, p = pearsonr(node_eff_conductance, node_current_distribution)
    print('Pearson correlation between desired distribution of effective conductances and distribution obtained \n \
          after optimization of the resistances in the network:\t{:.2f}'.format(corr))

    #L1 = np.sum(np.abs(node_current_distribution-node_eff_conductance))/len(R)
    labels = [(gnd, 'Pt\nElectrode')]
    plot_network(G=G_optim, figsize=(8, 6), labels=labels, node_color=list(node_eff_conductance) + [0], show=False,
                 save_fold='./{:s}'.format(save_name), vmin=node_eff_conductance.min(),
                 cb_nodes_lab='Effective Conductance [a.u.]', cb_edge_lab='Edge Resistance [a.u.]',
                 #title='PearsonCorr = {:.2f}, pval={:.5}'.format(corr, p)
                 )

    a = 0

