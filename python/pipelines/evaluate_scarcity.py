from pathlib import Path
from dataproc.simulation_loader import reconstruct_hosts_ID
from dataproc.scarcity_strategies import RandomNodeDrop
from dataproc.summary_stats import compute_summary_statistics
import pandas as pd
import pyarrow.parquet as pq
import os
from tqdm import tqdm
from networkx import DiGraph


SCARCITY_LEVELS = [0.05, 0.10, 0.15, 0.20]
SUMMARY_CSV = "summaries/summary_stats.csv"
SUMMARY_COLUMNS = [
    "filename",
    "scarcity",
    "n_hosts",
    "n_dead",
    "mean_tRecov",  # etc.
    # Add true parameters like p_trans, p_fatal if stored in metadata
]


def process_single_simulation(file_path: Path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    df = reconstruct_hosts_ID(df)

    G = DiGraph()
    for idx, row in df.iterrows():
        G.add_node(row["hosts.ID"], **row.to_dict())
        if pd.notna(row["inf.by"]):
            G.add_edge(int(row["inf.by"]), row["hosts.ID"])

    summary_rows = []
    for level in SCARCITY_LEVELS:
        degrader = RandomNodeDrop(level)
        G_degraded = degrader.apply(G)

        # Convert degraded graph back to DataFrame
        df_degraded = pd.DataFrame.from_dict(
            dict(G_degraded.nodes(data=True)), orient="index"
        )
        df_degraded.reset_index(drop=True, inplace=True)

        stats = compute_summary_statistics(df_degraded)
        stats["filename"] = file_path.name
        stats["scarcity"] = level
        summary_rows.append(stats)

    return summary_rows


def write_summaries_to_disk(summary_batch, csv_path):
    df = pd.DataFrame(summary_batch)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)


def process_all_simulations(parquet_dir: str, summary_out="scarcity_summaries.csv"):
    files = sorted(Path(parquet_dir).glob("*.parquet"))

    for file_path in tqdm(files, desc="Processing simulations"):
        try:
            rows = process_single_simulation(file_path)
            write_summaries_to_disk(rows, summary_out)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")


def main() -> None:
    process_all_simulations("data/nosoi/")


if __name__ == "__main__":
    main()
