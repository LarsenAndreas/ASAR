from functools import cache
import heapq

import numpy as np
import psutil
from tqdm import tqdm
from math import cos, radians

try:
    from .utils import calcPenalty, l2, animate
except ImportError:
    from utils import calcPenalty, l2, animate


def updateWeights(v: tuple, t: int, pod: float, pos: float, mat_lat: np.ndarray, mat_lon: np.ndarray, mat_weights: np.ndarray, lons: np.ndarray, lats: np.ndarray):
    """Updates weights inplace

    Args:
        v (tuple): Vertex just visited
        t (int): Time vertex was visited
        pod (float): Probability of Detection
        pos (float): Probability of Success
        mat_lat (np.ndarray): Particle lattitudes
        mat_lon (np.ndarray): Particle longitudes
        mat_weights (np.ndarray): Particle weights
        lons (np.ndarray): Cartesian to Polar Longitude lookup table
        lats (np.ndarray): Cartesian to Polar Lattitude lookup table
    """

    mask_lon = (lons[v[0]] <= mat_lon[:, t]) * (mat_lon[:, t] <= lons[v[0] + 1])
    mask_lat = (lats[v[1]] <= mat_lat[:, t]) * (mat_lat[:, t] <= lats[v[1] + 1])
    mask = mask_lat * mask_lon  # Selects all the particles which have been searched
    mat_weights[mask, t:] *= (1 - pod) / (1 - pos)
    mat_weights[~mask, t:] *= 1 / (1 - pos)


def h(POC, pof, pof_target, sensor_perf, speed, mat_lat, mat_lon, mat_weights, lons, lats, n_particles):
    if pof <= pof_target:
        return 0
    ttm = 1 / speed
    t = 0
    weights = np.copy(mat_weights)
    while True:
        v = np.unravel_index(np.argmax(POC, axis=None), POC.shape)
        poc = POC[v[0], v[1]]
        pod = sensor_perf
        pos = poc * pod
        pof *= 1 - pos
        t += ttm
        if round(pof, 10) <= round(pof_target, 10):
            return t
        else:
            updateWeights(v=v, t=0, pod=pod, pos=pos, mat_lat=mat_lat, mat_lon=mat_lon, mat_weights=weights, lons=lons, lats=lats)
            POC = np.histogram2d(mat_lon.T[0, :], mat_lat.T[0, :], weights=mat_weights[:, 0], bins=(lons, lats), density=False)[0] / n_particles


def findPath(v_init, orient_init, pof_target, i_max, sensor_perf, time_max, speed, mat_lat, mat_lon, mat_weights, delta_lat, delta_t, resolution, G):
    # Initial Conversion
    n_particles = mat_weights.shape[0]
    delta_lat = resolution / 111000.0  # m to degrees
    delta_lon = delta_lat / cos(radians((np.nanmin(mat_lat) + np.nanmax(mat_lat)) / 2))
    lats = np.arange(np.nanmin(mat_lat) - delta_lat, np.nanmax(mat_lat) + delta_lat, delta_lat) - delta_lat / 2
    lons = np.arange(np.nanmin(mat_lon) - delta_lon, np.nanmax(mat_lon) + delta_lon, delta_lon)
    POC_init = np.histogram2d(mat_lon.T[0, :], mat_lat.T[0, :], weights=mat_weights[:, 0], bins=(lons, lats), density=False)[0] / n_particles

    POC = np.copy(POC_init)
    weights = np.copy(mat_weights)
    weights_neighbor = np.copy(mat_weights)

    i, pbar = 0, tqdm(total=i_max, mininterval=0.5)

    pof_min = 1.0
    g = {(v_init,): 0}
    f = [
        (
            h(
                POC=POC,
                pof=1,
                pof_target=pof_target,
                sensor_perf=sensor_perf,
                speed=speed,
                mat_lat=mat_lat,
                mat_lon=mat_lon,
                mat_weights=weights,
                lons=lons,
                lats=lats,
                n_particles=n_particles,
            ),
            1,
            (v_init,),
        )
    ]
    heapq.heapify(f)

    while len(f) != 0 and i < i_max:
        # Faster to set than copy
        POC[:, :] = POC_init[:, :]
        weights[:, :] = mat_weights[:, :]

        f_score, length, path = heapq.heappop(f)
        x0, y0 = orient_init
        pof = 1
        dt = 0
        # print(f"\n\n{path}")
        for j in range(len(path) - 1):
            v_c, v_n = path[j], path[j + 1]  # Current vertex, next vertex
            x1, y1 = v_n[0] - v_c[0], v_n[1] - v_c[1]  # Next orientation

            # Calculate the new Probability of Failure
            pod = sensor_perf
            poc = POC[v_c[0], v_c[1]]
            pos = pod * poc
            pof *= 1 - pos

            # Update the drift particles according to Bayers Rule
            dx = l2((x1, y1)) / speed  # Calculate the direct distance between current and next vertex
            dt += dx + dx * calcPenalty(x0, y0, x1, y1)  # Calculate the time based on turning
            t = int(dt // delta_t)  # Convert real time to drift index
            # print(f"{v_c=} | {dt=} | {t=}")
            updateWeights(v=v_c, t=t, pod=pod, pos=pos, mat_lat=mat_lat, mat_lon=mat_lon, mat_weights=weights, lons=lons, lats=lats)
            POC = np.histogram2d(mat_lon.T[t, :], mat_lat.T[t, :], weights=weights[:, t], bins=(lons, lats), density=False)[0] / n_particles
            x0, y0 = x1, y1

        v_c = path[-1]
        pod = sensor_perf
        poc = POC[v_c[0], v_c[1]]
        pos = pod * poc
        pof_current = pof * (1 - pos)
        # No need to update POC just yet

        pof_min = min(pof_current, pof_min)
        pbar.set_description_str(f"POF: {pof_current:.4%} | {pof_min:.2%}", refresh=False)
        pbar.set_postfix_str(
            f"Time: {g[path]:.2f}+{f_score-g[path]:.2f}={f_score:.2f} | Length: {length} | Frontier: {len(f) + 1} | Mem: {psutil.virtual_memory().percent}%", refresh=False
        )
        if round(pof_current, 10) <= round(pof_target, 10):
            pbar.set_description_str(f"POF: {pof_current:.4%} | {pof_min:.2%}", refresh=True)
            pbar.set_postfix_str(
                f"Time: {g[path]:.2f}+{f_score-g[path]:.2f}={f_score:.2f} | Length: {length} | Frontier: {len(f) + 1} | Mem: {psutil.virtual_memory().percent}%", refresh=True
            )
            return {"path": path, "args": dict(v=v_init, direction=orient_init)}

        for v_n in G.neighbors(v_c):
            weights_neighbor[:, :] = weights[:, :]
            dt_neighbor = dt

            # We haven't updated the weights for visiting v_c yet. Thus, assuming we are moving to v_n,
            # update the drift particles according to Bayers Rule
            x1, y1 = v_n[0] - v_c[0], v_n[1] - v_c[1]
            dx = l2((x1, y1)) / speed  # Calculate the direct distance between current and next vertex
            dt_neighbor += dx + dx * calcPenalty(x0, y0, x1, y1)  # Calculate the time based on turning
            t = int(dt_neighbor // delta_t)  # Convert real time to drift index
            # print(f"{v_c=} | {dt_neighbor=} | {t=}")
            updateWeights(v=v_c, t=t, pod=pod, pos=pos, mat_lat=mat_lat, mat_lon=mat_lon, mat_weights=weights_neighbor, lons=lons, lats=lats)
            POC_neighbor = np.histogram2d(mat_lon.T[t, :], mat_lat.T[t, :], weights=weights_neighbor[:, t], bins=(lons, lats), density=False)[0] / n_particles

            # Calculate pof at neighbor
            pod = sensor_perf
            poc = POC_neighbor[v_n[0], v_n[1]]
            pos = poc * pod
            pof_neighbor = pof_current * (1 - pos)

            # We assume no kinematic penalty when calculating h, thus, we update the weight without regards for the next move, i.e. t is unchanged
            updateWeights(v=v_n, t=t, pod=pod, pos=pos, mat_lat=mat_lat, mat_lon=mat_lon, mat_weights=weights_neighbor, lons=lons, lats=lats)
            POC_neighbor = np.histogram2d(mat_lon.T[t, :], mat_lat.T[t, :], weights=weights_neighbor[:, t], bins=(lons, lats), density=False)[0] / n_particles

            h_score = h(
                POC=POC_neighbor,
                pof=pof_neighbor,
                pof_target=pof_target,
                sensor_perf=sensor_perf,
                speed=speed,
                mat_lat=mat_lat,
                mat_lon=mat_lon,
                mat_weights=weights_neighbor,
                lons=lons,
                lats=lats,
                n_particles=n_particles,
            )
            dx = l2((x1, y1)) / speed
            dt_neighbor = dx + dx * calcPenalty(x0, y0, x1, y1)
            g_score = g[path] + dt_neighbor  # Time To Move
            f_score = g_score + h_score
            if time_max is None or f_score < time_max:
                g[path + (v_n,)] = g_score
                heapq.heappush(f, (f_score, len(path) + 1, path + (v_n,)))

        i += 1
        pbar.update()

    pbar.set_postfix_str(f"DID NOT COMPLETE!", refresh=True)
    *_, path = heapq.heappop(f)
    return {"path": path, "args": dict(v=v_init, direction=orient_init)}


if __name__ == "__main__":
    import networkx as nx
    from matplotlib import pyplot as plt
    from itertools import combinations
    from mpl_toolkits.basemap import Basemap

    # LOAD PREDEFINED EXAMPLE
    folder = "./data"
    with np.load(f"{folder}/np_arr_lat_2.npz") as npz:
        od_lat_time = np.ma.MaskedArray(**npz)
    with np.load(f"{folder}/np_arr_lon_2.npz") as npz:
        od_lon_time = np.ma.MaskedArray(**npz)
    with np.load(f"{folder}/np_arr_z_2.npz") as npz:
        od_z_time = np.ma.MaskedArray(**npz)
        od_z_time[:, :] = 1

    od_lat_time = np.ma.getdata(od_lat_time)
    od_lon_time = np.ma.getdata(od_lon_time)
    od_z_time = np.ma.getdata(od_z_time)

    for p in range(od_z_time.shape[0]):
        idx = np.flatnonzero(od_lat_time[p])[-1]
        od_lat_time[p, idx:] = od_lat_time[p][idx]
        od_lon_time[p, idx:] = od_lon_time[p][idx]

    resolution = 2000
    n_particles = od_z_time.shape[0]
    delta_lat = resolution / 111000.0  # m to degrees
    delta_lon = delta_lat / cos(radians((np.nanmin(od_lat_time) + np.nanmax(od_lat_time)) / 2))
    lats = np.arange(np.nanmin(od_lat_time) - delta_lat, np.nanmax(od_lat_time) + delta_lat, delta_lat) - delta_lat / 2
    lons = np.arange(np.nanmin(od_lon_time) - delta_lon, np.nanmax(od_lon_time) + delta_lon, delta_lon)

    POC = np.zeros(shape=(len(lons) - 1, len(lats) - 1))
    for t in tqdm(range(od_lon_time.shape[1])):
        POC += np.histogram2d(od_lon_time[:, t], od_lat_time[:, t], weights=od_z_time[:, t], bins=(lons, lats), density=False)[0]

    nodes = list(map(tuple, np.argwhere(POC != 0)))  # Only create nodes where there is poc
    max_dist = 1
    G = nx.Graph()
    G.add_nodes_from(nodes)

    edges = []
    for v, u in combinations(nodes, 2):
        dist = l2((v[0] - u[0], v[1] - u[1]))
        if dist <= max_dist:
            edges.append((v, u))
    G.add_edges_from(edges)

    extend = (
        od_lat_time[od_lat_time.nonzero()].min(),
        od_lat_time[od_lat_time.nonzero()].max(),
        od_lon_time[od_lon_time.nonzero()].min(),
        od_lon_time[od_lon_time.nonzero()].max(),
    )
    m = Basemap(projection="merc", llcrnrlat=extend[0], urcrnrlat=extend[1], llcrnrlon=extend[2], urcrnrlon=extend[3], lat_ts=20, resolution="h", suppress_ticks=True)
    left, bottom = m(extend[2], extend[0])
    right, top = m(extend[3], extend[1])
    extent = (left, right, bottom, top)
    # hist, *_ = np.histogram2d(data[:, 0], data[:, 1], bins=25)
    tx, ty = POC.shape
    dx = (extent[1] - extent[0]) / tx
    dy = (extent[3] - extent[2]) / ty
    node_pos = {
        (x, y): (
            extent[0] + dx / 2 + x * dx,
            extent[2] + dy / 2 + y * dy,
        )
        for x, y in G.nodes()
    }
    G.remove_nodes_from([(x, y) for (x, y) in G.nodes if m.is_land(*node_pos[(x, y)])])

    print("Finshed pruning!")
    idx = {node: i for i, node in enumerate(G.nodes)}
    nx.set_node_attributes(G, idx, "index")

    print(f"Graph Nodes: {len(G.nodes)}")
    print(f"Graph Edges: {len(G.edges)}")

    kwargs = dict(
        v_init=(2, 9),
        orient_init=(0, 1),
        pof_target=0.4,
        i_max=10000,
        sensor_perf=1,
        time_max=1e16,
        speed=1,
        mat_lat=od_lat_time,
        mat_lon=od_lon_time,
        mat_weights=od_z_time,
        delta_lat=delta_lat,
        delta_t=1,  # Relation between time and drift index | delta_t=1 means 1 drift index is 1 second
        resolution=resolution,
        G=G,
    )

    path = findPath(**kwargs)

    print(path)
    animate(path=path["path"], bm=m, node_pos=node_pos, **kwargs)
