"""Sync WearGait-PD from Synapse to Google Drive without keeping local data.

The script downloads one Synapse file into a temporary directory, uploads it to
Google Drive, then deletes the temporary file before moving to the next item.

Examples:
  python scripts/sync_weargait_to_gdrive.py --synapse-token <PAT> --service-account-json service-account.json --drive-folder-id <folder-id>
  python scripts/sync_weargait_to_gdrive.py --synapse-token <PAT> --oauth-client-json oauth-client.json
  python scripts/sync_weargait_to_gdrive.py --synapse-token <PAT> --service-account-json service-account.json --drive-folder-id <folder-id> --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable


SYNAPSE_ID = "syn52540892"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synapse-id", default=SYNAPSE_ID)
    parser.add_argument("--synapse-token", default=os.getenv("SYNAPSE_AUTH_TOKEN"))
    parser.add_argument("--drive-folder-id", default="root", help="Destination Google Drive parent folder ID")
    parser.add_argument(
        "--drive-root-folder-name",
        default="WearGait-PD",
        help="Dataset folder to create/reuse inside --drive-folder-id. Use '' to upload directly into the parent.",
    )
    parser.add_argument("--service-account-json", default=None)
    parser.add_argument("--oauth-client-json", default=None)
    parser.add_argument("--oauth-token-json", default=".gdrive_token.json")
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--include", action="append", default=["*.csv", "*.mat", "*.tsv"])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.synapse_token:
        raise SystemExit("Set --synapse-token or SYNAPSE_AUTH_TOKEN.")
    if bool(args.service_account_json) == bool(args.oauth_client_json):
        raise SystemExit("Pass exactly one of --service-account-json or --oauth-client-json.")
    return args


def drive_service(args: argparse.Namespace):
    from google.oauth2 import service_account
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if args.service_account_json:
        creds = service_account.Credentials.from_service_account_file(
            args.service_account_json,
            scopes=DRIVE_SCOPES,
        )
    else:
        token_path = Path(args.oauth_token_json)
        creds = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), DRIVE_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(args.oauth_client_json, DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def escape_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def find_drive_child(service, parent_id: str, name: str, mime_type: str | None = None) -> dict | None:
    clauses = [
        f"name = '{escape_query_value(name)}'",
        f"'{parent_id}' in parents",
        "trashed = false",
    ]
    if mime_type:
        clauses.append(f"mimeType = '{mime_type}'")
    query = " and ".join(clauses)
    result = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, mimeType, size)",
        pageSize=10,
    ).execute()
    files = result.get("files", [])
    return files[0] if files else None


def ensure_drive_folder(service, parent_id: str, name: str, dry_run: bool) -> str:
    if dry_run and parent_id.startswith("dry-run-folder:"):
        return f"{parent_id}/{name}"
    existing = find_drive_child(service, parent_id, name, FOLDER_MIME)
    if existing:
        return existing["id"]
    if dry_run:
        return f"dry-run-folder:{parent_id}/{name}"
    body = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
    created = service.files().create(body=body, fields="id").execute()
    return created["id"]


def should_include(path: str, include: Iterable[str], exclude: Iterable[str]) -> bool:
    name = path.replace("\\", "/")
    if not any(fnmatch.fnmatch(name.lower(), pattern.lower()) for pattern in include):
        return False
    return not any(fnmatch.fnmatch(name.lower(), pattern.lower()) for pattern in exclude)


def walk_synapse(syn, entity_id: str, rel_parts: tuple[str, ...] = ()):
    for child in syn.getChildren(entity_id):
        child_id = child["id"]
        child_name = child["name"]
        child_type = child.get("type", "").lower()
        child_parts = rel_parts + (child_name,)
        if "folder" in child_type:
            yield from walk_synapse(syn, child_id, child_parts)
        elif "file" in child_type:
            yield child_id, child_parts


def download_synapse_file(syn, entity_id: str, tmp_root: Path) -> Path:
    before = {p.resolve() for p in tmp_root.rglob("*") if p.is_file()}
    entity = syn.get(entity_id, downloadLocation=str(tmp_root))
    path = Path(entity.path)
    if path.exists():
        return path

    after = {p.resolve() for p in tmp_root.rglob("*") if p.is_file()}
    created = sorted(after - before)
    if not created:
        raise FileNotFoundError(f"Synapse download produced no local file for {entity_id}")
    return created[-1]


def upload_file(service, local_path: Path, parent_id: str, drive_name: str, overwrite: bool, dry_run: bool) -> str:
    from googleapiclient.http import MediaFileUpload

    if dry_run:
        return f"dry-run:upload:{drive_name}"
    existing = find_drive_child(service, parent_id, drive_name)
    if existing and not overwrite:
        return f"skipped:{existing['id']}"

    media = MediaFileUpload(str(local_path), resumable=True)
    if existing and overwrite:
        updated = service.files().update(
            fileId=existing["id"],
            media_body=media,
            fields="id",
        ).execute()
        return f"updated:{updated['id']}"

    body = {"name": drive_name, "parents": [parent_id]}
    created = service.files().create(
        body=body,
        media_body=media,
        fields="id",
    ).execute()
    return f"uploaded:{created['id']}"


def main() -> None:
    import synapseclient

    args = parse_args()

    syn = synapseclient.Synapse()
    syn.login(authToken=args.synapse_token)
    service = drive_service(args)
    drive_parent_id = args.drive_folder_id
    if args.drive_root_folder_name:
        drive_parent_id = ensure_drive_folder(
            service,
            args.drive_folder_id,
            args.drive_root_folder_name,
            args.dry_run,
        )

    tmp_root = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="weargait-gdrive-"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    folder_cache: dict[tuple[str, tuple[str, ...]], str] = {}

    uploaded = 0
    skipped = 0
    failed = 0
    seen = 0

    try:
        for entity_id, rel_parts in walk_synapse(syn, args.synapse_id):
            rel_path = "/".join(rel_parts)
            if not should_include(rel_path, args.include, args.exclude):
                continue
            if args.max_files is not None and seen >= args.max_files:
                break
            seen += 1

            parent_id = drive_parent_id
            folder_parts = rel_parts[:-1]
            for i, folder in enumerate(folder_parts):
                key = (parent_id, folder_parts[: i + 1])
                if key not in folder_cache:
                    folder_cache[key] = ensure_drive_folder(service, parent_id, folder, args.dry_run)
                parent_id = folder_cache[key]

            drive_name = rel_parts[-1]
            try:
                existing = None if args.dry_run else find_drive_child(service, parent_id, drive_name)
                if existing and not args.overwrite:
                    skipped += 1
                    print(f"SKIP existing: {rel_path}")
                    continue

                local_path = tmp_root / drive_name if args.dry_run else download_synapse_file(syn, entity_id, tmp_root)
                result = upload_file(service, local_path, parent_id, drive_name, args.overwrite, args.dry_run)
                uploaded += int(result.startswith(("uploaded:", "updated:", "dry-run:")))
                print(f"{result} {rel_path}")
            except Exception as exc:
                failed += 1
                print(f"FAIL {rel_path}: {exc}")
            finally:
                for file_path in tmp_root.rglob("*"):
                    if file_path.is_file():
                        file_path.unlink(missing_ok=True)

    finally:
        if not args.tmp_dir:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"Done. considered={seen} uploaded_or_planned={uploaded} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
