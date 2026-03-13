import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensures it works even without a GUI (optional but nice)
import matplotlib.pyplot as plt

SEED = 7
URL = "https://snap.stanford.edu/data/email-Enron.txt.gz"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figs"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "email-Enron.txt.gz"


def download_if_needed():
    if not DATA_PATH.exists():
        print(f"Downloading dataset to {DATA_PATH} ...")
        urllib.request.urlretrieve(URL, DATA_PATH)
        print("Download complete.")


def load_graph():
    """
    Returns:
      G: full undirected simple graph
      H: largest connected component subgraph (copy)
    """
    download_if_needed()

    # read_edgelist can read gz directly
    G = nx.read_edgelist(DATA_PATH)
    G = nx.Graph(G)  # ensure simple undirected

    largest_cc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc_nodes).copy()
    return G, H


def print_basic_metrics(G, name="G"):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    avg_degree = 2 * E / N if N else 0.0

    print(f"\n--- Basic metrics for {name} ---")
    print("N =", N)
    print("E =", E)
    print("Average degree =", avg_degree)
    print("Density =", nx.density(G))
    print("Self-loops =", nx.number_of_selfloops(G))
    print("Number of connected components =", nx.number_connected_components(G))
    print("Is connected?", nx.is_connected(G) if N > 0 else False)
    print("Average clustering coefficient =", nx.average_clustering(G))
    print("Transitivity =", nx.transitivity(G))


def topk_degree_centrality(G, k=10):
    dc = nx.degree_centrality(G)
    return sorted(dc.items(), key=lambda x: x[1], reverse=True)[:k]


def topk_betweenness(H, k=10, approx_k=200, seed=SEED):
    # Approximate betweenness by node sampling (faster)
    bc = nx.betweenness_centrality(H, k=approx_k, seed=seed)
    return sorted(bc.items(), key=lambda x: x[1], reverse=True)[:k]


def save_current_fig(filename: str):
    out = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved figure:", out)


def init_states_sis(H, infected_frac, rng):
    G = H.copy()
    nx.set_node_attributes(G, "S", "state")
    nodes = list(G.nodes())
    k = max(1, int(infected_frac * len(nodes)))
    infected = rng.choice(nodes, size=k, replace=False)
    for u in infected:
        G.nodes[u]["state"] = "I"
    return G


def init_states_sir(H, infected_frac, rng):
    G = H.copy()
    nx.set_node_attributes(G, "S", "state")
    nodes = list(G.nodes())
    k = max(1, int(infected_frac * len(nodes)))
    infected = rng.choice(nodes, size=k, replace=False)
    for u in infected:
        G.nodes[u]["state"] = "I"
    return G


def sis_step(G, beta, mu, rng):
    new_state = {}
    for u in G.nodes():
        s = G.nodes[u]["state"]
        if s == "S":
            m = sum(1 for v in G.neighbors(u) if G.nodes[v]["state"] == "I")
            p_infect = 1 - (1 - beta) ** m
            new_state[u] = "I" if rng.random() < p_infect else "S"
        else:
            new_state[u] = "S" if rng.random() < mu else "I"
    nx.set_node_attributes(G, new_state, "state")
    I = sum(1 for u in G.nodes() if G.nodes[u]["state"] == "I")
    return I / G.number_of_nodes()


def run_sis(G, beta, mu, T, seed):
    rng = np.random.default_rng(seed)
    traj = []
    I0 = sum(1 for u in G.nodes() if G.nodes[u]["state"] == "I") / G.number_of_nodes()
    traj.append(I0)
    for _ in range(T):
        traj.append(sis_step(G, beta, mu, rng))
    return np.array(traj)


def sir_step(G, beta, mu, rng):
    new_state = {}
    for u in G.nodes():
        s = G.nodes[u]["state"]
        if s == "S":
            m = sum(1 for v in G.neighbors(u) if G.nodes[v]["state"] == "I")
            p_infect = 1 - (1 - beta) ** m
            new_state[u] = "I" if rng.random() < p_infect else "S"
        elif s == "I":
            new_state[u] = "R" if rng.random() < mu else "I"
        else:
            new_state[u] = "R"

    nx.set_node_attributes(G, new_state, "state")

    N = G.number_of_nodes()
    S = sum(1 for u in G.nodes() if G.nodes[u]["state"] == "S") / N
    I = sum(1 for u in G.nodes() if G.nodes[u]["state"] == "I") / N
    R = sum(1 for u in G.nodes() if G.nodes[u]["state"] == "R") / N
    return S, I, R


def run_sir(G, beta, mu, T, seed):
    rng = np.random.default_rng(seed)
    S_list, I_list, R_list = [], [], []

    N = G.number_of_nodes()
    S0 = sum(1 for u in G.nodes() if G.nodes[u]["state"] == "S") / N
    I0 = sum(1 for u in G.nodes() if G.nodes[u]["state"] == "I") / N
    R0 = sum(1 for u in G.nodes() if G.nodes[u]["state"] == "R") / N

    S_list.append(S0)
    I_list.append(I0)
    R_list.append(R0)

    for _ in range(T):
        S, I, R = sir_step(G, beta, mu, rng)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)

    return np.array(S_list), np.array(I_list), np.array(R_list)


def mean_std_sis(H, beta, mu, T, trials, seed, infected_frac=0.01):
    all_traj = []
    for r in range(trials):
        rng_init = np.random.default_rng(seed + r)
        G0 = init_states_sis(H, infected_frac, rng_init)
        traj = run_sis(G0, beta, mu, T, seed + r)
        all_traj.append(traj)
    A = np.vstack(all_traj)
    return A.mean(axis=0), A.std(axis=0)


def mean_std_sir_final_size(H, beta, mu, T, trials, seed, infected_frac=0.01):
    final_R = []
    for r in range(trials):
        rng_init = np.random.default_rng(seed + r)
        G0 = init_states_sir(H, infected_frac, rng_init)
        _, _, R = run_sir(G0, beta, mu, T, seed + r)
        final_R.append(R[-1])
    final_R = np.array(final_R)
    return final_R.mean(), final_R.std()


def fig_degree_distribution(G):
    deg = [d for _, d in G.degree()]
    plt.figure()
    plt.hist(deg, bins=50)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Enron degree distribution (full graph)")
    save_current_fig("01_degree_distribution.png")


def fig_sis_mean_band(H):
    beta, mu = 0.03, 0.10
    T = 50
    trials = 20  # assignment-style averaging

    mean, std = mean_std_sis(H, beta, mu, T, trials, SEED, infected_frac=0.01)
    t = np.arange(len(mean))

    plt.figure()
    plt.plot(t, mean, label="Mean infected fraction")
    plt.fill_between(t, mean - std, mean + std, alpha=0.2, label="±1 std")
    plt.xlabel("Time")
    plt.ylabel("Infected fraction")
    plt.title(f"SIS mean ± std over {trials} runs (beta={beta}, mu={mu})")
    plt.legend()
    save_current_fig("02_sis_mean_band.png")


def fig_sir_single_run(H):
    beta, mu = 0.05, 0.10
    T = 80

    rng_init = np.random.default_rng(SEED)
    G0 = init_states_sir(H, infected_frac=0.01, rng=rng_init)
    S, I, R = run_sir(G0, beta, mu, T, SEED)

    plt.figure()
    plt.plot(S, label="S")
    plt.plot(I, label="I")
    plt.plot(R, label="R")
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.title(f"SIR single run (beta={beta}, mu={mu})")
    plt.legend()
    save_current_fig("03_sir_single_run.png")

    print("\nSIR single run summary:")
    print("Final epidemic size R(T) =", R[-1])
    print("Final I(T) =", I[-1])



def fig_sir_final_size_vs_beta(H):
    mu = 0.10
    T = 60
    trials = 20
    betas = np.linspace(0.005, 0.08, 8)

    means = []
    stds = []
    for beta in betas:
        m, s = mean_std_sir_final_size(H, beta, mu, T, trials, SEED, infected_frac=0.01)
        means.append(m)
        stds.append(s)

    plt.figure()
    plt.plot(betas, means, marker="o")
    plt.xlabel("Beta")
    plt.ylabel("Final epidemic size (mean)")
    plt.title(f"SIR final size vs beta (trials={trials}, mu={mu})")
    save_current_fig("04_sir_final_size_vs_beta.png")

def main():
    G, H = load_graph()

    # Console outputs for report (metrics + centrality)
    print_basic_metrics(G, "G (full graph)")
    print_basic_metrics(H, "H (largest connected component)")

    print("\nTop 10 nodes by degree centrality (on full graph G):")
    for node, val in topk_degree_centrality(G, k=10):
        print(node, val)

    print("\nTop 10 nodes by betweenness centrality approx (on largest CC H):")
    for node, val in topk_betweenness(H, k=10, approx_k=200, seed=SEED):
        print(node, val)

    # Figures (saved to ./figs)
    fig_degree_distribution(G)
    fig_sis_mean_band(H)
    fig_sir_single_run(H)
    fig_sir_final_size_vs_beta(H)

    print("\nDone. All figures saved in:", FIG_DIR)


if __name__ == "__main__":
    main
