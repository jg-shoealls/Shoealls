"""
data/raw/ 로컬 파일 → Google Drive 업로드

구조:
  Drive: WearGait-PD/
    demographics/  (2개)
    HC/            (170개)
    PD/            (199개)

사용법:
  python scripts/upload_to_drive.py
"""

import io
import pickle
from pathlib import Path

CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.pickle")
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

LOCAL_DIRS = {
    "demographics": Path("data/raw/demographics"),
    "HC":           Path("data/raw/HC"),
    "PD":           Path("data/raw/PD"),
}
DRIVE_ROOT = "WearGait-PD"


def get_drive_service():
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    creds = None
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            try:
                creds = flow.run_local_server(port=0)
            except Exception:
                creds = flow.run_console()
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)

    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, name: str, parent_id: str | None = None) -> str:
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = service.files().list(q=q, fields="files(id)").execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    return service.files().create(body=meta, fields="id").execute()["id"]


def file_exists(service, name: str, parent_id: str) -> bool:
    q = f"name='{name}' and '{parent_id}' in parents and trashed=false"
    res = service.files().list(q=q, fields="files(id)").execute()
    return bool(res.get("files"))


def upload_file(service, path: Path, parent_id: str):
    if file_exists(service, path.name, parent_id):
        return
    from googleapiclient.http import MediaIoBaseUpload
    meta = {"name": path.name, "parents": [parent_id]}
    media = MediaIoBaseUpload(io.BytesIO(path.read_bytes()), mimetype="text/csv", resumable=True)
    service.files().create(body=meta, media_body=media, fields="id").execute()


def main():
    print("Google Drive 인증 중... (브라우저 창이 열리면 허용 클릭)")
    service = get_drive_service()
    print("인증 완료\n")

    root_id = get_or_create_folder(service, DRIVE_ROOT)
    print(f"Drive 루트: {DRIVE_ROOT} (id={root_id})\n")

    total_done = 0
    for folder_name, local_dir in LOCAL_DIRS.items():
        files = sorted(local_dir.glob("*.csv"))
        if not files:
            print(f"  {folder_name}/: 파일 없음, 스킵")
            continue

        folder_id = get_or_create_folder(service, folder_name, parent_id=root_id)
        print(f"{folder_name}/ 업로드 ({len(files)}개)...")
        for i, path in enumerate(files, 1):
            print(f"  [{i:3}/{len(files)}] {path.name}", end="\r", flush=True)
            upload_file(service, path, folder_id)
        print(f"  {folder_name}/ 완료 ({len(files)}개)        ")
        total_done += len(files)

    print(f"\n총 {total_done}개 업로드 완료 → Drive: My Drive/{DRIVE_ROOT}/")


if __name__ == "__main__":
    main()
