from math import acos, sin, pi, cos, radians, ceil
from functools import lru_cache
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns

from mpl_toolkits.basemap import Basemap

clr = sns.color_palette("deep").as_hex()


@lru_cache
def l2(v):  # L2 vector norm
    return (v[0] ** 2 + v[1] ** 2) ** 0.5


@lru_cache
def calcPenalty(x0: float, y0: float, x1: float, y1: float) -> float:
    theta = acos(round((x1 * x0 + y1 * y0) / ((x0**2 + y0**2) ** 0.5 * (x1**2 + y1**2) ** 0.5), 6))
    # return (1 + sin(theta - pi / 2)) / 2
    return 5 * (1 + sin(theta - pi / 2)) / 2  # Might speed up calculations


def calcPathTime(path: list, speed: float, orient_init: tuple) -> float:
    x0, y0 = orient_init
    t = 0
    for v_c, v_n in zip(path[:-1], path[1:]):
        x1, y1 = v_n[0] - v_c[0], v_n[1] - v_c[1]
        dx = l2((x1, y1)) / speed
        dt = dx + dx * calcPenalty(x0, y0, x1, y1)
        t += dt
        x0, y0 = x1, y1
    return t * 60


def calcPathLength(path):
    cum_length = 0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        cum_length += ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

    return cum_length


def calcPathProb(path, POC_init, pod):
    POC = POC_init.copy()
    pof = 1
    for v in path:
        poc = POC[v[0], v[1]]
        pos = pod * poc
        pof *= 1 - pos
        POC *= 1 / (1 - pos)
        POC[v[0], v[1]] = poc * (1 - pod) / (1 - pos)
    return 1 - pof


def animate(
    path,
    v_init,
    orient_init,
    sensor_perf,
    speed,
    mat_lat,
    mat_lon,
    mat_weights,
    delta_lat,
    delta_t,
    resolution,
    G,
    bm,
    node_pos,
    **kwargs,
):
    n_particles = mat_weights.shape[0]
    delta_lat = resolution / 111000.0  # m to degrees
    delta_lon = delta_lat / cos(radians((np.nanmin(mat_lat) + np.nanmax(mat_lat)) / 2))
    lats = np.arange(np.nanmin(mat_lat) - delta_lat, np.nanmax(mat_lat) + delta_lat, delta_lat) - delta_lat / 2
    lons = np.arange(np.nanmin(mat_lon) - delta_lon, np.nanmax(mat_lon) + delta_lon, delta_lon)
    POC = np.histogram2d(mat_lon.T[0, :], mat_lat.T[0, :], weights=mat_weights[:, 0], bins=(lons, lats), density=False)[0] / n_particles

    # Figure setup
    cols = 3
    rows = ceil(len(path) / cols)
    fig = plt.figure(figsize=(15, 5 * rows))

    # node_pos = {(x, y): (x, y) for (x, y) in G.nodes}

    x0, y0 = orient_init
    pof = 1
    dt = 0
    t = 0

    edge_list = []
    # clr_cont = sns.color_palette("Spectral", n_colors=len(path)).as_hex()
    clr_cont = sns.light_palette(clr[4], n_colors=len(path)).as_hex()

    t = -1
    idx = 1
    for j in range(len(path) - 1):
        v_c, v_n = path[j], path[j + 1]  # Current vertex, next vertex
        x1, y1 = v_n[0] - v_c[0], v_n[1] - v_c[1]  # Next orientation

        # Calculate the new Probability of Failure
        pod = sensor_perf
        poc = POC[v_c[0], v_c[1]]
        pos = pod * poc
        pof *= 1 - pos

        x0, y0 = x1, y1

        step = int(dt // delta_t)  # Convert real time to drift index)
        # print(f"Step: {step}, Time: {t}, dt: {dt}, dt/delta_t: {dt/delta_t}, dt//delta_t: {dt//delta_t}")
        if t < step:  # Convert real time to drift index
            t = step
            # Draw
            ax = fig.add_subplot(rows, cols, idx)
            idx += 1

            # ax.set_title(f"POS: {1-pof:.2%}")
            # ax.set_title(f"Time {dt:.2f}")
            # ax.set_xlabel(f"Movement Time: {dt:.2f} | Simulation Time: {t:.2f}")

            bm.drawcoastlines(ax=ax, color=clr[2], zorder=0)
            bm.fillcontinents(ax=ax, color=clr[2], zorder=1)

            bm.imshow(POC.T, cmap=sns.color_palette("Blues", as_cmap=True))  # , origin="upper")
            nx.draw_networkx_nodes(G, pos=node_pos, nodelist=[v_c], node_size=100, node_color=clr[4], ax=ax)
            nx.draw_networkx_edges(G, pos=node_pos, width=3, edgelist=edge_list, edge_color=clr_cont[-j:], ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        edge_list.append((v_c, v_n))

        # Update the drift particles according to Bayers Rule
        dx = l2((x1, y1)) / speed  # Calculate the direct distance between current and next vertex
        dt += dx + dx * calcPenalty(x0, y0, x1, y1)  # Calculate the time based on turning

        # Update weights
        POC = np.histogram2d(mat_lon.T[t, :], mat_lat.T[t, :], weights=mat_weights[:, t], bins=(lons, lats), density=False)[0] / n_particles
        mask_lon = (lons[v_c[0]] <= mat_lon[:, t]) * (mat_lon[:, t] <= lons[v_c[0] + 1])
        mask_lat = (lats[v_c[1]] <= mat_lat[:, t]) * (mat_lat[:, t] <= lats[v_c[1] + 1])
        mask = mask_lat * mask_lon  # Selects all the particles which have been searched
        mat_weights[mask, t:] *= (1 - pod) / (1 - pos)
        mat_weights[~mask, t:] *= 1 / (1 - pos)

    v_c = path[-1]
    # edgelist.append(v_c)
    pod = sensor_perf
    poc = POC[v_c[0], v_c[1]]
    pos = pod * poc
    pof *= 1 - pos

    # Draw
    ax = fig.add_subplot(rows, cols, idx)
    # ax.set_title(f"POS: {1-pof:.2%}")
    # ax.set_title(f"Time {dt:.2f}")
    # ax.set_xlabel(f"Movement Time: {dt:.2f} | Simulation Time: {t:.2f}")

    bm.drawcoastlines(ax=ax, color=clr[2], zorder=0)
    bm.fillcontinents(ax=ax, color=clr[2], zorder=1)

    bm.imshow(POC.T, cmap=sns.color_palette("Blues", as_cmap=True))  # , origin="upper")
    nx.draw_networkx_nodes(G, pos=node_pos, nodelist=[v_c], node_size=100, node_color=clr[4], ax=ax)
    nx.draw_networkx_edges(G, pos=node_pos, width=3, edgelist=edge_list, edge_color=clr_cont[-j - 1 :], ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    fig.tight_layout()
    plt.savefig("opendrift.pdf", bbox_inches="tight")
    # plt.show()
