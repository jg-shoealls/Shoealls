"""Download WearGait-PD dataset from Synapse (syn52540892).

Usage:
    python scripts/download_weargait.py --username <email> --password <pw>
    python scripts/download_weargait.py  # prompts for credentials
"""

import argparse
import pathlib
import synapseclient
import synapseutils

SYNAPSE_ID = "syn52540892"
DOWNLOAD_DIR = pathlib.Path("data/weargait_pd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--token", default=None, help="Synapse Personal Access Token")
    args = parser.parse_args()

    syn = synapseclient.Synapse()

    if args.token:
        syn.login(authToken=args.token)
    elif args.username and args.password:
        syn.login(args.username, args.password)
    else:
        print("Usage:")
        print("  --token <PAT>           Personal Access Token (권장)")
        print("  --username <email> --password <pw>")
        raise SystemExit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Syncing {SYNAPSE_ID} -> {DOWNLOAD_DIR}/ (재귀 다운로드)")

    synapseutils.syncFromSynapse(
        syn,
        entity=SYNAPSE_ID,
        path=str(DOWNLOAD_DIR),
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
