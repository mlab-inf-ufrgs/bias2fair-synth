import os
import pandas as pd
import numpy as np

DATASETS = {
    'compas':   {'n_total': 5278},
    'adult':    {'n_total': 10000},
    'diabetes': {'n_total': 10000},
}

TEST_SIZE  = 0.2
SCENARIOS  = ['S4', 'S5', 'S6']
SYNTH_ROOT = 'synthetic_data'


def n_train(n_total: int) -> int:
    return int(n_total * (1 - TEST_SIZE))


def main():
    print("=" * 70)
    print("Synthetic data added by scenario (S4 / S5 / S6)")
    print(f"Root: {SYNTH_ROOT}/")
    print("=" * 70)

    for ds_name, cfg in DATASETS.items():
        nt = n_train(cfg['n_total'])
        print(f"\n{'─' * 70}")
        print(f"Dataset : {ds_name.upper()}  |  n_total={cfg['n_total']}  |  n_train≈{nt}")

        for scenario in SCENARIOS:
            sc_dir = os.path.join(SYNTH_ROOT, ds_name, scenario)
            if not os.path.exists(sc_dir):
                print(f"\n  {scenario}: directory not found ({sc_dir})")
                continue

            records = []
            for gen in sorted(os.listdir(sc_dir)):
                gen_dir = os.path.join(sc_dir, gen)
                if not os.path.isdir(gen_dir):
                    continue
                for fname in sorted(os.listdir(gen_dir)):
                    if fname.endswith('.csv'):
                        fpath = os.path.join(gen_dir, fname)
                        n_rows = len(pd.read_csv(fpath))
                        records.append({
                            'generator': gen,
                            'seed':      fname.replace('seed_', '').replace('.csv', ''),
                            'n_added':   n_rows,
                        })

            if not records:
                print(f"\n  {scenario}: no CSV found")
                continue

            df = pd.DataFrame(records)
            overall_mean = df['n_added'].mean()
            overall_pct  = overall_mean / nt * 100

            print(f"\n  {scenario}")
            print(f"    Added average : {overall_mean:,.1f} lines  →  {overall_pct:.1f}% of train")
            print(f"    By generator:")

            by_gen = df.groupby('generator')['n_added'].agg(['mean', 'min', 'max', 'std'])
            for gen_name, row in by_gen.iterrows():
                pct = row['mean'] / nt * 100
                std_str = f"  std={row['std']:.1f}" if row['std'] > 0 else ""
                print(
                    f"      {gen_name:<20s}  Average={row['mean']:6.1f} ({pct:.1f}%)"
                    f"   [min={row['min']:.0f}, max={row['max']:.0f}]{std_str}"
                )

    print(f"\n{'=' * 70}\n")


if __name__ == '__main__':
    main()
