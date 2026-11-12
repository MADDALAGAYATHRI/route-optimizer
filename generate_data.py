"""Generate the synthetic dataset: data/shipments.json (+ flat data/routes.csv)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from routeopt.data import generate_dataset, routes_table, save_dataset  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of shipments")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "shipments.json"))
    a = ap.parse_args()
    shipments = generate_dataset(a.n, a.seed)
    save_dataset(shipments, a.out)
    df = routes_table(shipments)
    df.to_csv(Path(a.out).with_name("routes.csv"), index=False)
    print(f"Wrote {len(shipments)} shipments / {len(df)} routes "
          f"({int(df.is_decoy.sum())} planted decoys) to {a.out} and routes.csv")
