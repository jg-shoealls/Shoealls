"""
Synapse.zip 압축 해제 → 필요한 파일만 필터 → Google Drive 업로드

사용법:
  python scripts/organize_to_drive.py --zip "C:/path/to/Synapse.zip"

필요 패키지:
  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

Google OAuth 설정:
  1. Google Cloud Console → APIs & Services → Credentials
  2. OAuth 2.0 Client ID (Desktop app) 생성
  3. credentials.json 다운로드 → 프로젝트 루트에 저장
"""

import argparse
import io
import os
import pickle
import zipfile
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────

# Drive에 생성할 최상위 폴더 이름
DRIVE_ROOT_FOLDER = "WearGait-PD"

# 다운받을 태스크 (파일명의 _<Task>.csv 부분)
TARGET_TASKS = {"SelfPace", "TUG"}

# 로컬 임시 추출 경로
LOCAL_EXTRACT_DIR = Path("data/raw")

# Google OAuth 파일 경로
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.pickle")

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# ── 파일 분류 ─────────────────────────────────────────────────────────

def classify(filename: str) -> str | None:
    """
    파일을 분류해 폴더명 반환. 해당 없으면 None.
      demographics/ : Demographic+Clinical CSV
      HC/           : HC* SelfPace / TUG
      PD/           : PD* SelfPace / TUG
    """
    name = Path(filename).name
    stem = Path(filename).stem  # e.g. HC100_SelfPace

    if "Demographic" in name or "demographic" in name:
        return "demographics"

    if "_" in stem:
        prefix, task = stem.split("_", 1)
        if task in TARGET_TASKS:
            if prefix.startswith("HC"):
                return "HC"
            if prefix.startswith("PD"):
                return "PD"

    return None


# ── Google Drive 인증 ─────────────────────────────────────────────────

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
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    "credentials.json 없음.\n"
                    "Google Cloud Console에서 OAuth 2.0 Client ID(Desktop app)를\n"
                    "생성하고 credentials.json을 프로젝트 루트에 저장하세요."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)

    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=creds)


# ── Drive 폴더 유틸 ───────────────────────────────────────────────────

def get_or_create_folder(service, name: str, parent_id: str | None = None) -> str:
    """폴더가 있으면 ID 반환, 없으면 생성 후 ID 반환"""
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"

    results = service.files().list(q=q, fields="files(id, name)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]

    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def upload_csv(service, name: str, content: bytes, parent_id: str) -> str:
    from googleapiclient.http import MediaIoBaseUpload

    meta = {"name": name, "parents": [parent_id]}
    media = MediaIoBaseUpload(
        io.BytesIO(content),
        mimetype="text/csv",
        resumable=True,
    )
    file = service.files().create(body=meta, media_body=media, fields="id").execute()
    return file["id"]


# ── 메인 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Synapse.zip 경로")
    parser.add_argument("--no-drive", action="store_true", help="Drive 업로드 스킵 (로컬만 추출)")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"zip 파일 없음: {zip_path}")

    # 1. zip 열고 파일 목록 파악
    print(f"zip 파일 열기: {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")
    with zipfile.ZipFile(zip_path) as zf:
        all_names = zf.namelist()
        print(f"전체 파일 수: {len(all_names)}")

        # 분류
        categorized: dict[str, list[str]] = {"demographics": [], "HC": [], "PD": []}
        for name in all_names:
            cat = classify(name)
            if cat:
                categorized[cat].append(name)

        total = sum(len(v) for v in categorized.values())
        print(f"\n추출 대상: {total}개")
        for cat, files in categorized.items():
            print(f"  {cat}/: {len(files)}개")

        # 2. 로컬 추출
        print("\n로컬 추출 중...")
        extracted: dict[str, list[tuple[str, bytes]]] = {k: [] for k in categorized}
        for cat, names in categorized.items():
            dest = LOCAL_EXTRACT_DIR / cat
            dest.mkdir(parents=True, exist_ok=True)
            for i, name in enumerate(names, 1):
                basename = Path(name).name
                print(f"  [{i}/{len(names)}] {cat}/{basename}", end="\r")
                content = zf.read(name)
                (dest / basename).write_bytes(content)
                extracted[cat].append((basename, content))
            print(f"\n  {cat}/ → {dest} ({len(names)}개)")

    print(f"\n로컬 추출 완료: {LOCAL_EXTRACT_DIR.resolve()}")

    if args.no_drive:
        return

    # 3. Google Drive 업로드
    print("\nGoogle Drive 인증 중...")
    service = get_drive_service()

    # 루트 폴더 생성
    root_id = get_or_create_folder(service, DRIVE_ROOT_FOLDER)
    print(f"Drive 루트 폴더: {DRIVE_ROOT_FOLDER} (id={root_id})")

    for cat, files in extracted.items():
        folder_id = get_or_create_folder(service, cat, parent_id=root_id)
        print(f"\n{cat}/ 업로드 ({len(files)}개)...")
        for i, (name, content) in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] {name}", end="\r", flush=True)
            upload_csv(service, name, content, folder_id)
        print(f"\n  완료")

    print(f"\n모든 파일 Drive 업로드 완료!")
    print(f"  Drive 경로: My Drive / {DRIVE_ROOT_FOLDER}/")
    print(f"    demographics/ : {len(extracted['demographics'])}개")
    print(f"    HC/           : {len(extracted['HC'])}개")
    print(f"    PD/           : {len(extracted['PD'])}개")


if __name__ == "__main__":
    main()
