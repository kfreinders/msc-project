import numpy as np
import pandas as pd
import networkx as nx


def safe_stat(func, x, default=0):
    try:
        val = func(x.dropna())
        return default if pd.isna(val) else val
    except Exception:
        return default


def compute_summary_statistics(
    hosts_table: pd.DataFrame, nosoi_settings: dict, metadata: dict
) -> pd.DataFrame:
    # Basic statistics
    ss_noninf = (~hosts_table["hosts.ID"].isin(hosts_table["inf.by"])).sum()
    freq_table = hosts_table["inf.by"].value_counts()
    result_table = freq_table.reset_index()
    result_table.columns = ["hosts.ID", "Frequency"]

    ss_mean_secinf = safe_stat(np.mean, result_table["Frequency"])
    ss_med_secinf = safe_stat(np.median, result_table["Frequency"])
    ss_var_secinf = safe_stat(np.var, result_table["Frequency"])

    # Fraction of infectors responsible for 50%
    result_table = result_table.sort_values("Frequency", ascending=False)
    result_table["Cumulative"] = result_table["Frequency"].cumsum()
    total_infections = result_table["Frequency"].sum()
    half = total_infections * 0.5
    ss_fractop50 = ((result_table["Cumulative"] >= half).idxmax() + 1) / len(
        result_table
    )

    ss_hostspertime = len(hosts_table) / metadata["simtime"]

    # Infection time stats
    inf_time = hosts_table["out.time"] - hosts_table["inf.time"]
    ss_mean_inftime = safe_stat(np.mean, inf_time)
    ss_med_inftime = safe_stat(np.median, inf_time)
    ss_var_inftime = safe_stat(np.var, inf_time)

    ss_prop_infectors = len(freq_table) / len(hosts_table)

    ss_active_final = hosts_table["active"].sum()
    ss_hosts_total = len(hosts_table)
    ss_frac_active_final = ss_active_final / ss_hosts_total if ss_hosts_total > 0 else 0

    # Infection timing difference
    merged = hosts_table.merge(
        hosts_table,
        left_on="inf.by",
        right_on="hosts.ID",
        suffixes=("", "_infector"),
    )
    merged["inf_time_diff"] = merged["inf.time"] - merged["inf.time_infector"]

    ss_mean_inflag = safe_stat(
        np.mean, merged["inf_time_diff"], metadata["simtime"] + 1
    )
    ss_min_inflag = (
        safe_stat(lambda x: x.min(), merged.groupby("inf.by")["inf_time_diff"].min())
        if len(merged) > 1
        else metadata["simtime"] + 1
    )
    ss_med_inflag = safe_stat(np.median, merged["inf_time_diff"])
    ss_var_inflag = safe_stat(np.var, merged["inf_time_diff"])

    ss_frac_runtime = metadata["simtime"] / nosoi_settings["length"]

    # Network statistics
    edges = hosts_table[["inf.by", "hosts.ID"]].dropna()
    G = nx.from_pandas_edgelist(
        edges, source="inf.by", target="hosts.ID", create_using=nx.DiGraph
    )

    ss_g_degree = safe_stat(np.mean, pd.Series(dict(G.degree())).values())
    ss_g_clustcoef = nx.transitivity(G)
    ss_g_density = nx.density(G)
    ss_g_diam = nx.diameter(G) if nx.is_connected(G.to_undirected()) else np.nan
    ss_g_meanego = safe_stat(
        np.mean, pd.Series([len(nx.ego_graph(G, n)) for n in G.nodes])
    )
    ss_g_radius = nx.radius(G) if nx.is_connected(G.to_undirected()) else np.nan
    ss_g_meanalpha = safe_stat(np.mean, pd.Series(nx.alpha_centrality(G)))
    ss_g_effglob = nx.global_efficiency(G)

    # Deaths
    if "fate" in hosts_table.columns:
        deaths = hosts_table["fate"] == 1
        recovered = hosts_table["fate"] == 2
        time_to_death = (hosts_table["out.time"] - hosts_table["inf.time"])[deaths]

        ss_deaths = deaths.sum()
        ss_mean_deaths = safe_stat(np.mean, deaths.astype(int))
        ss_mean_ttd = safe_stat(np.mean, time_to_death)
        ss_med_ttd = safe_stat(np.median, time_to_death)
        ss_var_ttd = safe_stat(np.var, time_to_death)
        ss_death_recov_ratio = (
            ss_deaths / recovered.sum() if recovered.sum() > 0 else np.nan
        )
    else:
        ss_deaths = ss_mean_deaths = ss_mean_ttd = ss_med_ttd = ss_var_ttd = (
            ss_death_recov_ratio
        ) = np.nan

    return pd.DataFrame(
        [
            {
                "SS_01": ss_noninf,
                "SS_02": ss_mean_secinf,
                "SS_03": ss_med_secinf,
                "SS_04": ss_var_secinf,
                "SS_05": ss_fractop50,
                "SS_06": ss_hostspertime,
                "SS_07": ss_mean_inftime,
                "SS_08": ss_med_inftime,
                "SS_09": ss_var_inftime,
                "SS_10": ss_prop_infectors,
                "SS_11": ss_hosts_total,
                "SS_12": ss_active_final,
                "SS_13": ss_frac_active_final,
                "SS_14": ss_mean_inflag,
                "SS_15": ss_min_inflag,
                "SS_16": ss_med_inflag,
                "SS_17": ss_var_inflag,
                "SS_18": ss_frac_runtime,
                "SS_19": ss_g_degree,
                "SS_20": ss_g_clustcoef,
                "SS_21": ss_g_density,
                "SS_22": ss_g_diam,
                "SS_23": ss_g_meanego,
                "SS_24": ss_g_radius,
                "SS_25": ss_g_meanalpha,
                "SS_26": ss_g_effglob,
                "SS_27": ss_deaths,
                "SS_28": ss_mean_deaths,
                "SS_29": ss_mean_ttd,
                "SS_30": ss_med_ttd,
                "SS_31": ss_var_ttd,
                "SS_32": ss_death_recov_ratio,
            }
        ]
    )
