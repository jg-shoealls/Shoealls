"""
WearGait-PD 데이터셋 다운로드 + Google Drive 업로드 스크립트

다운로드 대상:
  - CONTROLS / PD Demographic+Clinical CSV (레이블)
  - HC* / PD* SelfPace.csv  (주요 보행 태스크)
  - HC* / PD* TUG.csv       (임상 기준 테스트)

저장 구조:
  data/raw/
    demographics/
      CONTROLS - Demographic+Clinical - datasetV1.csv
      PD - Demographic+Clinical - datasetV1.csv
    HC/   (정상군 보행 CSV)
    PD/   (파킨슨군 보행 CSV)
"""

import os
import synapseclient
from synapseclient.api import get_children
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────
SYNAPSE_TOKEN = os.environ.get("SYNAPSE_TOKEN", "")  # 환경변수로 주입 권장

SYN_IDS = {
    "controls_demo": "syn55105521",   # CONTROLS Demographic+Clinical
    "pd_demo":       "syn55105530",   # PD Demographic+Clinical
    "controls_csv":  "syn61370552",   # CONTROL PARTICIPANTS / CSV files
    "pd_csv":        None,            # 아래에서 동적으로 조회
    "pd_root":       "syn61370536",   # PD PARTICIPANTS 루트
}

TARGET_TASKS = {"SelfPace", "TUG"}   # 받을 태스크 (mat/TURN/Balance 제외)

BASE_DIR = Path("data/raw")
DEMO_DIR = BASE_DIR / "demographics"
HC_DIR   = BASE_DIR / "HC"
PD_DIR   = BASE_DIR / "PD"

GOOGLE_DRIVE_FOLDER_ID = ""   # Drive 폴더 ID (선택, 비워두면 업로드 스킵)

# ── 유틸 ──────────────────────────────────────────────────────────────

def login(token: str = "") -> synapseclient.Synapse:
    syn = synapseclient.Synapse()
    if token:
        syn.login(authToken=token, silent=True)
    else:
        syn.login(silent=True)  # ~/.synapseConfig 자동 사용
    print(f"Synapse 로그인 완료: {syn.getUserProfile()['userName']}")
    return syn


def is_target(filename: str) -> bool:
    """SelfPace.csv / TUG.csv 만 True (mat, TURN, Balance 등 제외)"""
    stem = Path(filename).stem  # e.g. HC100_SelfPace
    task = stem.split("_", 1)[-1] if "_" in stem else ""
    return task in TARGET_TASKS


def download_file(syn: synapseclient.Synapse, syn_id: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    entity = syn.get(syn_id, downloadLocation=str(dest_dir), ifcollision="overwrite.local")
    return Path(entity.path)


def find_pd_csv_folder(syn: synapseclient.Synapse) -> str:
    """PD PARTICIPANTS 루트에서 'CSV files' 폴더 ID 찾기"""
    for item in syn.getChildren(SYN_IDS["pd_root"], includeTypes=["folder"]):
        if item["name"] == "CSV files":
            return item["id"]
    raise RuntimeError("PD 'CSV files' 폴더를 찾을 수 없습니다.")


def download_task_files(
    syn: synapseclient.Synapse,
    folder_id: str,
    dest_dir: Path,
    label: str,
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = [
        item for item in syn.getChildren(folder_id, includeTypes=["file"])
        if is_target(item["name"])
    ]
    print(f"\n{label}: {len(files)}개 파일 다운로드 시작")
    paths = []
    for i, item in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {item['name']}", end="\r", flush=True)
        paths.append(download_file(syn, item["id"], dest_dir))
    print(f"\n  완료 → {dest_dir}")
    return paths


# ── Google Drive 업로드 ───────────────────────────────────────────────

def upload_to_drive(local_paths: list[Path], drive_folder_id: str):
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2.credentials import Credentials
    except ImportError:
        print("\n[Drive 업로드 스킵] google-api-python-client 미설치")
        print("  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return

    # credentials.json 이 있어야 함 (Google Cloud Console에서 발급)
    creds_path = Path("credentials.json")
    if not creds_path.exists():
        print("\n[Drive 업로드 스킵] credentials.json 없음")
        return

    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle

    token_path = Path("token.pickle")
    creds = None
    if token_path.exists():
        with open(token_path, "rb") as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(creds_path), ["https://www.googleapis.com/auth/drive.file"]
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    service = build("drive", "v3", credentials=creds)
    print(f"\nGoogle Drive 업로드 → 폴더 {drive_folder_id}")
    for path in local_paths:
        meta = {"name": path.name, "parents": [drive_folder_id]}
        media = MediaFileUpload(str(path), resumable=True)
        file = service.files().create(body=meta, media_body=media, fields="id").execute()
        print(f"  업로드 완료: {path.name} (id={file['id']})")


# ── 메인 ──────────────────────────────────────────────────────────────

def main():
    token = SYNAPSE_TOKEN  # 비어있으면 ~/.synapseConfig 자동 사용
    syn = login(token)
    all_paths: list[Path] = []

    # 1. 인구통계/임상 CSV
    print("\n[1/3] Demographics 다운로드")
    for key, syn_id in [("controls_demo", SYN_IDS["controls_demo"]),
                        ("pd_demo",       SYN_IDS["pd_demo"])]:
        path = download_file(syn, syn_id, DEMO_DIR)
        print(f"  {path.name}")
        all_paths.append(path)

    # 2. 정상군 SelfPace / TUG
    print("\n[2/3] 정상군(HC) 보행 파일 다운로드")
    all_paths += download_task_files(syn, SYN_IDS["controls_csv"], HC_DIR, "HC")

    # 3. PD군 SelfPace / TUG
    print("\n[3/3] PD군 보행 파일 다운로드")
    pd_csv_folder = find_pd_csv_folder(syn)
    all_paths += download_task_files(syn, pd_csv_folder, PD_DIR, "PD")

    print(f"\n총 {len(all_paths)}개 파일 다운로드 완료")
    print(f"  demographics/ : {len(list(DEMO_DIR.glob('*.csv')))}개")
    print(f"  HC/           : {len(list(HC_DIR.glob('*.csv')))}개")
    print(f"  PD/           : {len(list(PD_DIR.glob('*.csv')))}개")

    # 4. Google Drive 업로드 (폴더 ID 설정 시)
    if GOOGLE_DRIVE_FOLDER_ID:
        upload_to_drive(all_paths, GOOGLE_DRIVE_FOLDER_ID)
    else:
        print("\n[Drive 업로드] GOOGLE_DRIVE_FOLDER_ID 를 설정하면 자동 업로드됩니다.")


if __name__ == "__main__":
    main()
