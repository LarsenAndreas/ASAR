import heapq

import numpy as np
import psutil
from tqdm import tqdm

from .utils import calcPenalty, l2
from networkx import Graph


def h(POC, pof, pof_target, sensor_perf, speed):
    """
    Calculate the time it takes for the probability of failure (pof) to reach a target value.

    Parameters:
    POC (numpy.ndarray): 2D array representing the Probability of Correct detection (POC) values.
    pof (float): Initial probability of failure.
    pof_target (float): Target probability of failure.
    sensor_perf (float): Sensor performance factor.
    speed (float): Speed factor.

    Returns:
    float: The time it takes for the pof to reach the target value.
    """
    ttm = 1 / speed
    t = 0
    while round(pof, 10) > round(pof_target, 10):
        v = np.unravel_index(np.argmax(POC, axis=None), POC.shape)
        pod = sensor_perf
        poc = POC[v[0], v[1]]
        pos = poc * pod
        updatePOC(POC, v, pod, poc)

        pof *= 1 - pos
        t += ttm
    return t


def updatePOC(POC: np.ndarray, v: tuple, pod: float, poc: float):
    """Update the Probability of Containment (inplace) using Bayers Rule

    Args:
        POC (np.ndarray): Probability map
        v (tuple): Searched vertex
        pod (float): Probability of Detection
        poc (float): Probability of Containment
    """

    pos = pod * poc
    POC *= 1 / (1 - pos)
    POC[v[0], v[1]] = poc * (1 - pod) / (1 - pos)


def findPath(v_init: tuple, orient_init: tuple, pof_target: float, i_max: int, sensor_perf: float, time_max: float, speed: float, POC_init: np.ndarray, G: Graph) -> dict:
    """
    Find a path in a graph using the A* algorithm.

    Args:
        v_init (tuple): Initial vertex coordinates.
        orient_init (tuple): Initial orientation coordinates.
        pof_target (float): Target probability of failure.
        i_max (int): Maximum number of iterations.
        sensor_perf (float): Sensor performance.
        time_max (float): Maximum time allowed for the path.
        speed (float): Speed of movement.
        POC_init (numpy.ndarray): Initial Probability of Coverage (POC) matrix.
        G (networkx.Graph): Graph representing the environment.

    Returns:
        dict: A dictionary containing the path, initial arguments, POC matrix, heap, and g values.

    Note:
        - The POC matrix is updated during the path finding process.
        - The A* algorithm is used to find the path.
        - The function returns the path and other relevant information if the target probability of failure is reached.
        - If the maximum number of iterations is reached or the time limit is exceeded, the function returns the best path found so far.
    """
    POC = np.copy(POC_init)
    POC_neighbor = np.copy(POC_init)
    i, pbar = 0, tqdm(total=i_max, mininterval=0.5)
    pof_min = 1.0
    g = {(v_init,): 0}
    f = [(h(POC, pof=1, pof_target=pof_target, sensor_perf=sensor_perf, speed=speed), 1, (v_init,))]
    heapq.heapify(f)
    while len(f) != 0 and i < i_max:
        f_score, length, path = heapq.heappop(f)
        pof = 1
        POC[:, :] = POC_init[:, :]  # Faster to set than copy
        x0, y0 = orient_init
        for j in range(len(path) - 1):
            v_c, v_n = path[j], path[j + 1]  # Current vertex, next vertex
            x0, y0 = v_n[0] - v_c[0], v_n[1] - v_c[1]  # Next orientation

            # Calculate the new Probability of Failure
            pod = sensor_perf
            poc = POC[v_c[0], v_c[1]]
            pos = pod * poc
            pof *= 1 - pos

            updatePOC(POC, v_c, pod, poc)

        v_c = path[-1]
        pod = sensor_perf
        poc = POC[v_c[0], v_c[1]]
        pos = pod * poc
        pof *= 1 - pos

        updatePOC(POC, v_c, pod, poc)

        pof_min = min(pof, pof_min)
        pbar.set_description_str(f"POF: {pof:.4%} | {pof_min:.2%}", refresh=False)
        pbar.set_postfix_str(
            f"Time: {g[path]:.2f}+{f_score-g[path]:.2f}={f_score:.2f} | Length: {length} | Frontier: {len(f) + 1} | Mem: {psutil.virtual_memory().percent}%", refresh=False
        )

        if round(pof, 10) <= round(pof_target, 10):
            return {"path": path, "args": dict(v=v_init, direction=orient_init), "POC": POC, "heap": f, "g": g}

        for v_n in G.neighbors(v_c):
            # Calculate pof at neighbor
            POC_neighbor[:, :] = POC[:, :]  # Faster to set than copy
            pod = sensor_perf
            poc = POC_neighbor[v_n[0], v_n[1]]
            pos = poc * pod
            pof_neighbor = pof * (1 - pos)
            updatePOC(POC_neighbor, v_n, pod, poc)

            h_score = h(POC_neighbor, pof_neighbor, pof_target, sensor_perf, speed=speed)
            x1, y1 = v_n[0] - v_c[0], v_n[1] - v_c[1]
            dx = l2((x1, y1)) / speed
            dt = dx + dx * calcPenalty(x0, y0, x1, y1)
            g_score = g[path] + dt  # Time To Move
            f_score = g_score + h_score
            if time_max is None or f_score < time_max:
                g[path + (v_n,)] = g_score
                heapq.heappush(f, (f_score, len(path) + 1, path + (v_n,)))

        i += 1
        pbar.update()

    pbar.set_postfix_str(f"DID NOT COMPLETE!", refresh=True)
    *_, path = heapq.heappop(f)
    return {"path": path, "args": dict(v=v_init, direction=orient_init)}
