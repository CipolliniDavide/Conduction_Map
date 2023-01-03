import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_network(G, node_color=list(), figsize=(10, 14), nodes_cmap=plt.cm.viridis, edge_cmap=plt.cm.Reds,
                 cb_nodes_lab='', cb_edge_lab='', edge_vmin=None, edge_vmax=None, vmin=None, vmax=None, up_text=1,
                 node_size=100, numeric_label=False,
                 labels=None, save_fold='./', title='', show=False):
    if labels is None:
        labels = [(node, '') for i, node in enumerate(G.nodes())]

    # colors = [G[u][v]['weight'] for u, v in G.edges]
    # edge_weight = [weight['weight'] for n1, n2, weight in list(G.edges(data=True))]
    try:
        coordinates = [(node, (feat['coord'])) for node, feat in G.nodes(data=True)]
    except:
        coordinates = nx.spring_layout(G, seed=63)

    fontdict_cb = {'fontsize': 'x-large', 'fontweight': 'bold'}
    fontdict_cb_ticks_label = {'fontsize': 'large', 'fontweight': 'bold'}

    fig, ax = plt.subplots(figsize=figsize)
    try:
        weights = np.array([G[u][v]['weight'] for u, v in G.edges])
    except:
        weights = np.ones(G.number_of_edges())

    if len(node_color) == 0:
        nx.draw_networkx(G, pos=dict(coordinates), with_labels=True, node_size=150,
                         labels=dict(labels), width=weights,
                         edge_color=weights, edge_vmin=np.min(weights), edge_vmax=np.max(weights))
    else:
        if not edge_vmin:
            edge_vmin = np.min(weights)
        if not edge_vmax:
            edge_vmax = np.max(weights)
        if not vmin:
            vmin = np.min(node_color)

        if not vmax:
            vmax = np.max(node_color)
        nx.draw_networkx(G, pos=dict(coordinates),
                         with_labels=numeric_label,
                         node_size=node_size,
                         cmap=nodes_cmap,
                         node_color=node_color,
                         vmin=vmin,
                         vmax=vmax,
                         # labels=dict(labels),
                         edge_cmap=edge_cmap,
                         width=4,
                         edge_color=weights,
                         edge_vmin=edge_vmin,
                         edge_vmax=edge_vmax,
                         font_size='x-large',
                         font_weight='bold', ax=ax)

        shrink_bar = 0.65
        pad = .001
        ticks_num = 2
        if cb_nodes_lab != '':
            sm_nodes = plt.cm.ScalarMappable(cmap=nodes_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            # cb_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=shrink_bar, pad=pad)
            # divider = make_axes_locatable(ax)
            # cax = divider.new_horizontal(size="3%", pad=pad, pack_start=False)
            # fig.add_axes(cax)
            cb_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=shrink_bar, pad=pad, orientation="vertical")

            cb_nodes.set_label(cb_nodes_lab, fontdict=fontdict_cb)
            cb_nodes_ticks = np.linspace(start=vmin, stop=vmax, num=ticks_num, endpoint=True)
            cb_tick_lab = ['{:.0e}'.format(i)  for i in cb_nodes_ticks]
            # cb_tick_lab = ['$\mathregular{1 \\times 10^{-16}}$', '$\mathregular{1}$']
            cb_nodes.ax.set_yticks(cb_nodes_ticks)
            cb_nodes.ax.set_yticklabels(cb_tick_lab, fontdict=fontdict_cb_ticks_label)

        #############3
        if cb_edge_lab != '':

            sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax))
            # cb_edges = plt.colorbar(sm_edges, shrink=shrink_bar, orientation="horizontal", pad=pad)
            # divider = make_axes_locatable(ax)
            # cax_edge = divider.new_vertical(size="3%", pad=pad, pack_start=True)
            # fig.add_axes(cax_edge)
            cb_edges = plt.colorbar(sm_edges, ax=ax, shrink=shrink_bar, orientation="horizontal", pad=pad)

            cb_edges.set_label(cb_edge_lab, fontdict=fontdict_cb)
            cb_edge_ticks = np.linspace(start=edge_vmin, stop=edge_vmax, num=ticks_num, endpoint=True)
            cb_edge_tick_lab = ['{:.0e}'.format(i) for i in cb_edge_ticks]
            # cb_edge_tick_lab = ['$\mathregular{4 \\times 10^{-6}}$', '$\mathregular{3 \\times 10^{2}}$' ]
            cb_edges.ax.set_xticks(cb_edge_ticks)
            cb_edges.ax.set_xticklabels(cb_edge_tick_lab, fontdict=fontdict_cb_ticks_label)

    if labels:
        for (n, lab) in labels:
            x, y = coordinates[n][1]
            plt.text(x, y + up_text, s=lab, fontdict=fontdict_cb, horizontalalignment='center')  # bbox=dict(facecolor='red', alpha=0.5),


    plt.box(False)
    fontdict = {'fontsize': 'xx-large', 'fontweight': 'bold'}
    ax.set_title(title, fontdict=fontdict)
    plt.tight_layout()
    if save_fold:
        plt.savefig(save_fold +'.png')
    if show:
        plt.show()
    else:
        plt.close()


def plot_imshow(array, figsize=(8, 8), cb_label='', ticks_num=3, vmin=None, vmax=None,
                last_ticks_num=None,
                save_fold='./', title='', show=False):
    fontdict_cb = {'fontsize': 'x-large', 'fontweight': 'bold'}
    fontdict_cb_ticks_label = {'fontsize': 'large', 'fontweight': 'bold'}

    if not vmin:
        vmin = np.min(array)
    if not vmax:
        vmax = np.max(array)

    fig, ax = plt.subplots(figsize=figsize)

    fontdict = {'fontsize': 'xx-large', 'fontweight': 'bold'}
    ax.set_title(title, fontdict=fontdict)

    img = ax.imshow(array)
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cb = plt.colorbar(img, cax=cax)
    cb.set_label(cb_label, fontdict=fontdict_cb)
    cb_ticks = np.linspace(start=vmin, stop=vmax, num=ticks_num, endpoint=True)
    if last_ticks_num:
        cb_ticks = cb_ticks[-last_ticks_num:]
    cb_tick_lab = ['{:.0e}'.format(i) for i in cb_ticks]
    # cb_tick_lab = ['$\mathregular{1 \\times 10^{-16}}$', '$\mathregular{1}$']
    cb.ax.set_yticks(cb_ticks)
    cb.ax.set_yticklabels(cb_tick_lab, fontdict=fontdict_cb_ticks_label)
    ax.autoscale_view()
    plt.tight_layout()
    if save_fold:
        plt.savefig(save_fold)
    if show:
        plt.show()
    else:
        plt.close()

def histogram(n_bins, values, save_name=None, save_fold='./', title='', ylabel='', xlabel='', ticks_num=3,
              figsize=(8,8)):
    fontdict = {'fontsize': 'x-large', 'fontweight': 'bold'}
    fontdict_ticks_label = {'fontsize': 'large', 'fontweight': 'bold'}

    fig = plt.figure(save_name, figsize=figsize)

    ax1 = fig.add_subplot(111)

    # ax1.plot(bins[1:], values, "b-", marker="o")
    ax1.hist(bins=n_bins, x=values)  # , "b-", marker="o")
    ax1.set_title(title, fontdict=fontdict)
    ax1.set_ylabel(ylabel, fontdict=fontdict)
    ax1.set_xlabel(xlabel, fontdict=fontdict)
    ax1.xaxis.set_tick_params(labelsize=18)
    ax1.yaxis.set_tick_params(labelsize=18)

    cb_edge_ticks = np.linspace(start=values.min(), stop=values.max(), num=ticks_num, endpoint=True)
    cb_edge_tick_lab = ['{:.0e}'.format(i) for i in cb_edge_ticks]
    ax1.set_xticks(cb_edge_ticks)
    ax1.set_xticklabels(cb_edge_tick_lab, fontdict=fontdict_ticks_label)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_fold + save_name + '.png')
        plt.close()
    else:
        plt.show()

