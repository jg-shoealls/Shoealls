"""Compare Synapse manifests with local files to check download status.
"""

import pathlib
import pandas as pd

def main():
    data_dir = pathlib.Path("data/weargait_pd")
    manifests = list(data_dir.glob("**/synapse_metadata_manifest.tsv"))
    
    print(f"Found {len(manifests)} manifest files.")
    
    all_manifested_files = set()
    for m in manifests:
        try:
            # Manifests are TSV
            df = pd.read_csv(m, sep="\t")
            if "name" in df.columns:
                for name in df["name"]:
                    all_manifested_files.add(str(name).lower())
        except Exception as e:
            print(f"Error reading {m}: {e}")
            
    print(f"Total files listed in manifests: {len(all_manifested_files)}")
    
    # Physically present files
    physical_files = set()
    for f in data_dir.glob("**/*"):
        if f.is_file() and f.suffix.lower() in ['.csv', '.mat']:
            physical_files.add(f.name.lower())
            
    print(f"Total physical files (.csv, .mat): {len(physical_files)}")
    
    missing = all_manifested_files - physical_files
    print(f"Missing files: {len(missing)}")
    
    # Category summary
    categories = {
        "SelfPace": [f for f in all_manifested_files if "selfpace" in f],
        "Balance": [f for f in all_manifested_files if "balance" in f],
        "FreeWalk": [f for f in all_manifested_files if "freewalk" in f],
        "DoorPat": [f for f in all_manifested_files if "doorpat" in f],
    }
    
    print("\nDownload Status by Category:")
    print(f"{'Category':<15} | {'Manifested':>10} | {'Present':>10} | {'%':>6}")
    print("-" * 50)
    for cat, names in categories.items():
        present_count = sum(1 for n in names if n in physical_files)
        total_count = len(names)
        pct = (present_count / total_count * 100) if total_count > 0 else 0
        print(f"{cat:<15} | {total_count:>10} | {present_count:>10} | {pct:>5.1f}%")

if __name__ == "__main__":
    main()
