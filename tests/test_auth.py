"""Tests for authentication and sync utility functions.

Covers pure-logic functions and mocked external-service auth flows in:
  - scripts/sync_weargait_to_gdrive.py
  - scripts/download_weargait.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import os
import importlib
from fastapi.testclient import TestClient
from api.examples import generate_sample_sensor_data

# ---------------------------------------------------------------------------
# Helpers to import the scripts without triggering their __main__ blocks
# ---------------------------------------------------------------------------

def _import_sync():
    """Import sync_weargait_to_gdrive without side effects."""
    import importlib
    import types

    # Stub heavy third-party modules so we can import without them installed
    # Build googleapiclient.http stub with MediaFileUpload pre-populated so
    # the lazy import inside upload_file() succeeds even in dry-run / skip paths.
    http_stub = types.ModuleType("googleapiclient.http")
    http_stub.MediaFileUpload = MagicMock()

    stubs = {
        "synapseclient": types.ModuleType("synapseclient"),
        "googleapiclient": types.ModuleType("googleapiclient"),
        "googleapiclient.discovery": types.ModuleType("googleapiclient.discovery"),
        "googleapiclient.http": http_stub,
        "google": types.ModuleType("google"),
        "google.oauth2": types.ModuleType("google.oauth2"),
        "google.oauth2.service_account": types.ModuleType("google.oauth2.service_account"),
        "google.oauth2.credentials": types.ModuleType("google.oauth2.credentials"),
        "google.auth": types.ModuleType("google.auth"),
        "google.auth.transport": types.ModuleType("google.auth.transport"),
        "google.auth.transport.requests": types.ModuleType("google.auth.transport.requests"),
        "google_auth_oauthlib": types.ModuleType("google_auth_oauthlib"),
        "google_auth_oauthlib.flow": types.ModuleType("google_auth_oauthlib.flow"),
    }
    for name, mod in stubs.items():
        sys.modules[name] = mod  # force-set so MediaFileUpload stub is always current

    # Add the project root to sys.path for import
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sync_weargait_to_gdrive",
        Path(__file__).parent.parent / "scripts" / "sync_weargait_to_gdrive.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_download():
    """Import download_weargait without side effects."""
    import types
    import importlib.util

    stubs = {
        "synapseclient": types.ModuleType("synapseclient"),
        "synapseutils": types.ModuleType("synapseutils"),
    }
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    spec = importlib.util.spec_from_file_location(
        "download_weargait",
        Path(__file__).parent.parent / "scripts" / "download_weargait.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# escape_query_value
# ---------------------------------------------------------------------------

class TestEscapeQueryValue:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def test_plain_string_unchanged(self):
        assert self.m.escape_query_value("hello world") == "hello world"

    def test_escapes_single_quote(self):
        assert self.m.escape_query_value("it's") == "it\\'s"

    def test_escapes_backslash(self):
        assert self.m.escape_query_value("a\\b") == "a\\\\b"

    def test_escapes_backslash_before_quote(self):
        # backslash AND quote: backslash escaped first, then quote
        assert self.m.escape_query_value("a\\'b") == "a\\\\\\'b"

    def test_empty_string(self):
        assert self.m.escape_query_value("") == ""


# ---------------------------------------------------------------------------
# should_include
# ---------------------------------------------------------------------------

class TestShouldInclude:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def test_csv_included_by_default_pattern(self):
        assert self.m.should_include("data/subject1.csv", ["*.csv"], [])

    def test_mat_included(self):
        assert self.m.should_include("sub/file.mat", ["*.mat"], [])

    def test_txt_excluded_when_not_in_include(self):
        assert not self.m.should_include("file.txt", ["*.csv", "*.mat"], [])

    def test_explicit_exclude_overrides_include(self):
        assert not self.m.should_include("file.csv", ["*.csv"], ["*.csv"])

    def test_case_insensitive_include(self):
        assert self.m.should_include("file.CSV", ["*.csv"], [])

    def test_case_insensitive_exclude(self):
        assert not self.m.should_include("file.CSV", ["*.csv"], ["*.csv"])

    def test_path_separators_normalised(self):
        assert self.m.should_include("a\\b\\c.csv", ["*.csv"], [])

    def test_no_include_patterns_excludes_all(self):
        assert not self.m.should_include("file.csv", [], [])

    def test_exclude_partial_pattern(self):
        assert not self.m.should_include("bad_file.csv", ["*.csv"], ["bad_*"])


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def _run(self, argv):
        with patch.object(sys, "argv", ["prog"] + argv):
            return self.m.parse_args()

    def test_service_account_path(self):
        args = self._run(["--synapse-token", "tok", "--service-account-json", "sa.json"])
        assert args.synapse_token == "tok"
        assert args.service_account_json == "sa.json"
        assert args.oauth_client_json is None

    def test_oauth_client_path(self):
        args = self._run(["--synapse-token", "tok", "--oauth-client-json", "oauth.json"])
        assert args.oauth_client_json == "oauth.json"
        assert args.service_account_json is None

    def test_missing_synapse_token_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(sys, "argv", ["prog", "--service-account-json", "sa.json"]):
                with pytest.raises(SystemExit):
                    self.m.parse_args()

    def test_both_service_and_oauth_raises(self):
        with patch.object(sys, "argv",
                          ["prog", "--synapse-token", "tok",
                           "--service-account-json", "sa.json",
                           "--oauth-client-json", "oauth.json"]):
            with pytest.raises(SystemExit):
                self.m.parse_args()

    def test_neither_service_nor_oauth_raises(self):
        with patch.object(sys, "argv", ["prog", "--synapse-token", "tok"]):
            with pytest.raises(SystemExit):
                self.m.parse_args()

    def test_synapse_token_from_env(self):
        with patch.dict("os.environ", {"SYNAPSE_AUTH_TOKEN": "env_tok"}):
            args = self._run(["--service-account-json", "sa.json"])
        assert args.synapse_token == "env_tok"

    def test_dry_run_default_false(self):
        args = self._run(["--synapse-token", "tok", "--service-account-json", "sa.json"])
        assert args.dry_run is False

    def test_dry_run_flag(self):
        args = self._run(["--synapse-token", "tok", "--service-account-json", "sa.json", "--dry-run"])
        assert args.dry_run is True

    def test_default_include_patterns(self):
        args = self._run(["--synapse-token", "tok", "--service-account-json", "sa.json"])
        assert "*.csv" in args.include
        assert "*.mat" in args.include
        assert "*.tsv" in args.include


# ---------------------------------------------------------------------------
# drive_service — service-account branch
# ---------------------------------------------------------------------------

class TestDriveServiceServiceAccount:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def test_service_account_builds_service(self):
        mock_creds = MagicMock()
        mock_service = MagicMock()
        args = SimpleNamespace(
            service_account_json="sa.json",
            oauth_client_json=None,
            oauth_token_json=".gdrive_token.json",
        )
        sa_module = MagicMock()
        sa_module.Credentials.from_service_account_file.return_value = mock_creds

        with patch.dict(sys.modules, {
            "google.oauth2.service_account": sa_module,
            "googleapiclient.discovery": MagicMock(build=lambda *a, **k: mock_service),
        }):
            with patch("googleapiclient.discovery.build", return_value=mock_service):
                # Patch the imports inside the function
                import types
                google_oauth2_sa = types.ModuleType("google.oauth2.service_account")
                google_oauth2_sa.Credentials = MagicMock()
                google_oauth2_sa.Credentials.from_service_account_file.return_value = mock_creds

                with patch.dict(sys.modules, {"google.oauth2.service_account": google_oauth2_sa}):
                    with patch("googleapiclient.discovery.build", return_value=mock_service):
                        # Build the patched environment so drive_service can import
                        with patch.object(sys.modules.get("google.oauth2.service_account", MagicMock()),
                                          "Credentials") as _:
                            pass  # just verifying setup

        # Simpler: patch __import__ path used inside drive_service
        google_oauth2_sa_mock = MagicMock()
        google_oauth2_sa_mock.Credentials.from_service_account_file.return_value = mock_creds
        googleapiclient_discovery_mock = MagicMock()
        googleapiclient_discovery_mock.build.return_value = mock_service

        with patch.dict(sys.modules, {
            "google.oauth2.service_account": google_oauth2_sa_mock,
            "google.oauth2.credentials": MagicMock(),
            "google.auth.transport.requests": MagicMock(),
            "google_auth_oauthlib.flow": MagicMock(),
            "googleapiclient.discovery": googleapiclient_discovery_mock,
        }):
            result = self.m.drive_service(args)

        google_oauth2_sa_mock.Credentials.from_service_account_file.assert_called_once_with(
            "sa.json", scopes=self.m.DRIVE_SCOPES
        )
        googleapiclient_discovery_mock.build.assert_called_once()
        assert result is mock_service


# ---------------------------------------------------------------------------
# drive_service — OAuth branch (existing valid token)
# ---------------------------------------------------------------------------

class TestDriveServiceOAuth:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def test_uses_existing_valid_token(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text("{}", encoding="utf-8")

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.expired = False
        mock_creds.to_json.return_value = '{"token": "valid"}'
        mock_service = MagicMock()

        args = SimpleNamespace(
            service_account_json=None,
            oauth_client_json="oauth.json",
            oauth_token_json=str(token_file),
        )

        creds_module = MagicMock()
        creds_module.Credentials.from_authorized_user_file.return_value = mock_creds
        discovery_module = MagicMock()
        discovery_module.build.return_value = mock_service

        with patch.dict(sys.modules, {
            "google.oauth2.service_account": MagicMock(),
            "google.oauth2.credentials": creds_module,
            "google.auth.transport.requests": MagicMock(),
            "google_auth_oauthlib.flow": MagicMock(),
            "googleapiclient.discovery": discovery_module,
        }):
            result = self.m.drive_service(args)

        creds_module.Credentials.from_authorized_user_file.assert_called_once_with(
            str(token_file), self.m.DRIVE_SCOPES
        )
        assert result is mock_service

    def test_refreshes_expired_token(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text("{}", encoding="utf-8")

        mock_creds = MagicMock()
        mock_creds.expired = True
        mock_creds.refresh_token = "rtoken"
        mock_creds.to_json.return_value = '{"refreshed": true}'
        mock_service = MagicMock()

        # After refresh() is called the real library sets creds.valid = True;
        # simulate that so the OAuth flow branch is not taken.
        def _do_refresh(_req):
            mock_creds.valid = True

        mock_creds.valid = False
        mock_creds.refresh.side_effect = _do_refresh

        args = SimpleNamespace(
            service_account_json=None,
            oauth_client_json="oauth.json",
            oauth_token_json=str(token_file),
        )

        request_cls = MagicMock()
        creds_module = MagicMock()
        creds_module.Credentials.from_authorized_user_file.return_value = mock_creds
        discovery_module = MagicMock()
        discovery_module.build.return_value = mock_service
        transport_module = MagicMock()
        transport_module.Request = request_cls

        with patch.dict(sys.modules, {
            "google.oauth2.service_account": MagicMock(),
            "google.oauth2.credentials": creds_module,
            "google.auth.transport.requests": transport_module,
            "google_auth_oauthlib.flow": MagicMock(),
            "googleapiclient.discovery": discovery_module,
        }):
            self.m.drive_service(args)

        mock_creds.refresh.assert_called_once()

    def test_runs_oauth_flow_when_no_token_file(self, tmp_path):
        token_file = tmp_path / "nonexistent_token.json"

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.to_json.return_value = '{"token": "new"}'
        mock_service = MagicMock()

        args = SimpleNamespace(
            service_account_json=None,
            oauth_client_json="oauth.json",
            oauth_token_json=str(token_file),
        )

        flow_mock = MagicMock()
        flow_mock.run_local_server.return_value = mock_creds
        flow_module = MagicMock()
        flow_module.InstalledAppFlow.from_client_secrets_file.return_value = flow_mock
        discovery_module = MagicMock()
        discovery_module.build.return_value = mock_service

        with patch.dict(sys.modules, {
            "google.oauth2.service_account": MagicMock(),
            "google.oauth2.credentials": MagicMock(),
            "google.auth.transport.requests": MagicMock(),
            "google_auth_oauthlib.flow": flow_module,
            "googleapiclient.discovery": discovery_module,
        }):
            self.m.drive_service(args)

        flow_module.InstalledAppFlow.from_client_secrets_file.assert_called_once_with(
            "oauth.json", self.m.DRIVE_SCOPES
        )
        flow_mock.run_local_server.assert_called_once_with(port=0)
        assert token_file.exists()


# ---------------------------------------------------------------------------
# find_drive_child
# ---------------------------------------------------------------------------

class TestFindDriveChild:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def _make_service(self, files):
        service = MagicMock()
        service.files().list().execute.return_value = {"files": files}
        return service

    def test_returns_first_match(self):
        file_entry = {"id": "abc", "name": "test.csv", "mimeType": "text/csv"}
        service = self._make_service([file_entry])
        result = self.m.find_drive_child(service, "parent_id", "test.csv")
        assert result == file_entry

    def test_returns_none_when_no_match(self):
        service = self._make_service([])
        result = self.m.find_drive_child(service, "parent_id", "missing.csv")
        assert result is None

    def test_query_includes_name_and_parent(self):
        service = MagicMock()
        service.files().list().execute.return_value = {"files": []}
        self.m.find_drive_child(service, "parent123", "my file.csv")
        list_kwargs = service.files().list.call_args[1]
        assert "parent123" in list_kwargs["q"]
        assert "my file.csv" in list_kwargs["q"]

    def test_query_includes_mime_type_when_given(self):
        service = MagicMock()
        service.files().list().execute.return_value = {"files": []}
        self.m.find_drive_child(service, "pid", "folder", mime_type="application/vnd.google-apps.folder")
        list_kwargs = service.files().list.call_args[1]
        assert "application/vnd.google-apps.folder" in list_kwargs["q"]

    def test_trashed_false_in_query(self):
        service = MagicMock()
        service.files().list().execute.return_value = {"files": []}
        self.m.find_drive_child(service, "pid", "f.csv")
        list_kwargs = service.files().list.call_args[1]
        assert "trashed = false" in list_kwargs["q"]


# ---------------------------------------------------------------------------
# ensure_drive_folder
# ---------------------------------------------------------------------------

class TestEnsureDriveFolder:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def test_returns_existing_folder_id(self):
        service = MagicMock()
        existing = {"id": "existing_id", "name": "MyFolder", "mimeType": self.m.FOLDER_MIME}
        with patch.object(self.m, "find_drive_child", return_value=existing):
            result = self.m.ensure_drive_folder(service, "parent", "MyFolder", dry_run=False)
        assert result == "existing_id"

    def test_creates_folder_if_not_exists(self):
        service = MagicMock()
        service.files().create().execute.return_value = {"id": "new_id"}
        with patch.object(self.m, "find_drive_child", return_value=None):
            result = self.m.ensure_drive_folder(service, "parent", "NewFolder", dry_run=False)
        assert result == "new_id"

    def test_dry_run_returns_placeholder_for_new_folder(self):
        service = MagicMock()
        with patch.object(self.m, "find_drive_child", return_value=None):
            result = self.m.ensure_drive_folder(service, "parent", "DryFolder", dry_run=True)
        assert result.startswith("dry-run-folder:")
        assert "DryFolder" in result

    def test_dry_run_with_existing_dry_run_parent(self):
        service = MagicMock()
        result = self.m.ensure_drive_folder(
            service, "dry-run-folder:parent/Sub", "Child", dry_run=True
        )
        assert "Child" in result

    def test_dry_run_returns_existing_folder_id_if_found(self):
        service = MagicMock()
        existing = {"id": "found_id"}
        with patch.object(self.m, "find_drive_child", return_value=existing):
            result = self.m.ensure_drive_folder(service, "parent", "Found", dry_run=False)
        assert result == "found_id"


# ---------------------------------------------------------------------------
# walk_synapse
# ---------------------------------------------------------------------------

class TestWalkSynapse:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def _make_syn(self, tree: dict):
        """Build a mock syn where getChildren returns tree[id]."""
        syn = MagicMock()
        syn.getChildren.side_effect = lambda entity_id: tree.get(entity_id, [])
        return syn

    def test_yields_files(self):
        tree = {
            "root": [
                {"id": "f1", "name": "file.csv", "type": "org.sagebionetworks.repo.model.FileEntity"},
            ]
        }
        syn = self._make_syn(tree)
        results = list(self.m.walk_synapse(syn, "root"))
        assert results == [("f1", ("file.csv",))]

    def test_recurses_into_folders(self):
        tree = {
            "root": [
                {"id": "fold1", "name": "subdir", "type": "org.sagebionetworks.repo.model.Folder"},
            ],
            "fold1": [
                {"id": "f1", "name": "data.csv", "type": "org.sagebionetworks.repo.model.FileEntity"},
            ],
        }
        syn = self._make_syn(tree)
        results = list(self.m.walk_synapse(syn, "root"))
        assert results == [("f1", ("subdir", "data.csv"))]

    def test_skips_unknown_types(self):
        tree = {
            "root": [
                {"id": "x", "name": "weird", "type": "org.sagebionetworks.repo.model.Project"},
            ]
        }
        syn = self._make_syn(tree)
        results = list(self.m.walk_synapse(syn, "root"))
        assert results == []

    def test_empty_entity(self):
        syn = self._make_syn({})
        results = list(self.m.walk_synapse(syn, "empty"))
        assert results == []

    def test_preserves_relative_path_parts(self):
        tree = {
            "root": [
                {"id": "d1", "name": "level1", "type": "folder"},
            ],
            "d1": [
                {"id": "d2", "name": "level2", "type": "folder"},
            ],
            "d2": [
                {"id": "f", "name": "deep.mat", "type": "file"},
            ],
        }
        syn = self._make_syn(tree)
        results = list(self.m.walk_synapse(syn, "root"))
        assert results == [("f", ("level1", "level2", "deep.mat"))]


# ---------------------------------------------------------------------------
# download_synapse_file
# ---------------------------------------------------------------------------

class TestDownloadSynapseFile:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def test_returns_entity_path_if_exists(self, tmp_path):
        local_file = tmp_path / "data.csv"
        local_file.write_text("x")
        mock_entity = MagicMock()
        mock_entity.path = str(local_file)
        syn = MagicMock()
        syn.get.return_value = mock_entity

        result = self.m.download_synapse_file(syn, "syn123", tmp_path)
        assert result == local_file

    def test_raises_when_no_file_downloaded(self, tmp_path):
        mock_entity = MagicMock()
        mock_entity.path = str(tmp_path / "nonexistent.csv")
        syn = MagicMock()
        syn.get.return_value = mock_entity

        with pytest.raises(FileNotFoundError):
            self.m.download_synapse_file(syn, "syn123", tmp_path)

    def test_falls_back_to_new_file_in_tmp(self, tmp_path):
        new_file = tmp_path / "new_data.csv"
        mock_entity = MagicMock()
        mock_entity.path = str(tmp_path / "nonexistent.csv")
        syn = MagicMock()

        def side_effect(entity_id, downloadLocation):
            new_file.write_text("downloaded")
            return mock_entity

        syn.get.side_effect = side_effect
        result = self.m.download_synapse_file(syn, "syn123", tmp_path)
        assert result == new_file


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------

class TestUploadFile:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_sync()

    def test_dry_run_returns_placeholder(self, tmp_path):
        local = tmp_path / "f.csv"
        local.write_text("data")
        result = self.m.upload_file(MagicMock(), local, "parent", "f.csv", False, dry_run=True)
        assert result.startswith("dry-run:upload:")

    def test_skips_existing_when_no_overwrite(self, tmp_path):
        local = tmp_path / "f.csv"
        local.write_text("data")
        service = MagicMock()
        existing = {"id": "eid"}
        with patch.object(self.m, "find_drive_child", return_value=existing):
            result = self.m.upload_file(service, local, "parent", "f.csv", overwrite=False, dry_run=False)
        assert result == "skipped:eid"

    def test_uploads_new_file(self, tmp_path):
        local = tmp_path / "f.csv"
        local.write_text("data")
        service = MagicMock()
        service.files().create().execute.return_value = {"id": "new_id"}

        media_mock = MagicMock()
        with patch.object(self.m, "find_drive_child", return_value=None):
            with patch.dict(sys.modules, {
                "googleapiclient.http": MagicMock(MediaFileUpload=MagicMock(return_value=media_mock))
            }):
                result = self.m.upload_file(service, local, "parent", "f.csv", overwrite=False, dry_run=False)
        assert result.startswith("uploaded:")

    def test_overwrites_existing_file(self, tmp_path):
        local = tmp_path / "f.csv"
        local.write_text("data")
        service = MagicMock()
        service.files().update().execute.return_value = {"id": "updated_id"}
        existing = {"id": "eid"}
        media_mock = MagicMock()
        with patch.object(self.m, "find_drive_child", return_value=existing):
            with patch.dict(sys.modules, {
                "googleapiclient.http": MagicMock(MediaFileUpload=MagicMock(return_value=media_mock))
            }):
                result = self.m.upload_file(service, local, "parent", "f.csv", overwrite=True, dry_run=False)
        assert result.startswith("updated:")


# ---------------------------------------------------------------------------
# download_weargait.py — Synapse auth branching
# ---------------------------------------------------------------------------

class TestDownloadSynapseAuth:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.m = _import_download()

    def _run_main(self, argv):
        with patch.object(sys, "argv", ["prog"] + argv):
            self.m.main()

    def test_token_auth_used_when_provided(self):
        syn_cls = MagicMock()
        syn_inst = MagicMock()
        syn_cls.return_value = syn_inst

        with patch.dict(sys.modules, {
            "synapseclient": MagicMock(Synapse=syn_cls),
            "synapseutils": MagicMock(syncFromSynapse=MagicMock()),
        }):
            # Re-import to pick up patched modules
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "download_weargait2",
                Path(__file__).parent.parent / "scripts" / "download_weargait.py",
            )
            mod = importlib.util.module_from_spec(spec)
            with patch.object(sys, "argv", ["prog", "--token", "mytoken"]):
                with patch("pathlib.Path.mkdir"):
                    spec.loader.exec_module(mod)
                    mod.main()

        syn_inst.login.assert_called_with(authToken="mytoken")

    def test_username_password_auth_used(self):
        syn_cls = MagicMock()
        syn_inst = MagicMock()
        syn_cls.return_value = syn_inst

        with patch.dict(sys.modules, {
            "synapseclient": MagicMock(Synapse=syn_cls),
            "synapseutils": MagicMock(syncFromSynapse=MagicMock()),
        }):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "download_weargait3",
                Path(__file__).parent.parent / "scripts" / "download_weargait.py",
            )
            mod = importlib.util.module_from_spec(spec)
            with patch.object(sys, "argv", ["prog", "--username", "u@x.com", "--password", "pw"]):
                with patch("pathlib.Path.mkdir"):
                    spec.loader.exec_module(mod)
                    mod.main()

        syn_inst.login.assert_called_with("u@x.com", "pw")

    def test_no_credentials_exits(self):
        syn_cls = MagicMock()
        with patch.dict(sys.modules, {
            "synapseclient": MagicMock(Synapse=syn_cls),
            "synapseutils": MagicMock(),
        }):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "download_weargait4",
                Path(__file__).parent.parent / "scripts" / "download_weargait.py",
            )
            mod = importlib.util.module_from_spec(spec)
            with patch.object(sys, "argv", ["prog"]):
                spec.loader.exec_module(mod)
                with pytest.raises(SystemExit) as exc_info:
                    mod.main()
            assert exc_info.value.code == 1


# ─────────────────────────────────────────────────────────────────────────────
# API auth middleware tests (from main branch)
# ─────────────────────────────────────────────────────────────────────────────

"""인증 미들웨어 테스트.

API_KEYS 환경변수를 통한 API 키 인증 동작을 검증합니다.
"""



def _classify_body():
    """분류 엔드포인트용 요청 바디 (ClassifyRequest 형식)."""
    d = generate_sample_sensor_data(gait_class=0)
    return {
        "sensor_data": {
            "imu": d["imu"],
            "pressure": d["pressure"],
            "skeleton": d["skeleton"],
        }
    }


def _reload_app(api_keys: str | None = None):
    """환경변수를 적용하고 앱을 재로드해 TestClient를 반환."""
    if api_keys is None:
        os.environ.pop("API_KEYS", None)
    else:
        os.environ["API_KEYS"] = api_keys

    import api.auth as m_auth
    import api.main as m_main
    importlib.reload(m_auth)
    importlib.reload(m_main)
    return TestClient(m_main.app, raise_server_exceptions=False)


# ── 데모 모드(인증 비활성) ─────────────────────────────────────────────
class TestDemoMode:
    """API_KEYS 미설정 시 모든 경로 자유 접근."""

    def setup_method(self):
        self.client = _reload_app(api_keys=None)

    def test_health_accessible(self):
        assert self.client.get("/health").status_code == 200

    def test_classify_accessible_without_key(self):
        r = self.client.post("/api/v1/classify", json=_classify_body())
        assert r.status_code == 200

    def test_sample_accessible_without_key(self):
        assert self.client.get("/api/v1/sample").status_code == 200


# ── 인증 활성 모드 ─────────────────────────────────────────────────────
class TestAuthEnabled:
    """API_KEYS 설정 시 키 없는 요청은 401 반환."""

    def setup_method(self):
        self.client = _reload_app("test-secret-key")

    def teardown_method(self):
        os.environ.pop("API_KEYS", None)

    # 공개 경로는 키 없이 접근 가능
    def test_health_public_no_key(self):
        assert self.client.get("/health").status_code == 200

    def test_docs_public_no_key(self):
        assert self.client.get("/docs").status_code == 200

    def test_sample_public_no_key(self):
        assert self.client.get("/api/v1/sample").status_code == 200

    # 보호 경로: 키 없음 → 401
    def test_classify_requires_key(self):
        r = self.client.post("/api/v1/classify", json=_classify_body())
        assert r.status_code == 401

    def test_analyze_requires_key(self):
        r = self.client.post("/api/v1/analyze", json=_classify_body())
        assert r.status_code == 401

    def test_401_body_structure(self):
        r = self.client.post("/api/v1/classify", json=_classify_body())
        body = r.json()
        assert "detail" in body
        assert "error" in body["detail"]

    # 헤더 키 인증
    def test_valid_header_key(self):
        r = self.client.post(
            "/api/v1/classify",
            json=_classify_body(),
            headers={"X-API-Key": "test-secret-key"},
        )
        assert r.status_code == 200

    def test_invalid_header_key(self):
        r = self.client.post(
            "/api/v1/classify",
            json=_classify_body(),
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    # 쿼리 파라미터 키 인증
    def test_valid_query_key(self):
        r = self.client.post(
            "/api/v1/classify?api_key=test-secret-key",
            json=_classify_body(),
        )
        assert r.status_code == 200

    # 헤더가 쿼리보다 우선
    def test_header_takes_priority_over_query(self):
        r = self.client.post(
            "/api/v1/classify?api_key=wrong-key",
            json=_classify_body(),
            headers={"X-API-Key": "test-secret-key"},
        )
        assert r.status_code == 200

    # 여러 유효 키
    def test_multiple_valid_keys(self):
        client = _reload_app("key-a,key-b,key-c")
        for key in ["key-a", "key-b", "key-c"]:
            r = client.post(
                "/api/v1/classify",
                json=_classify_body(),
                headers={"X-API-Key": key},
            )
            assert r.status_code == 200, f"key={key} should be valid"


# ── 응답 헤더 ─────────────────────────────────────────────────────────
class TestResponseHeaders:
    """로깅 미들웨어가 X-Request-ID를 응답 헤더에 포함하는지 확인."""

    def setup_method(self):
        self.client = _reload_app(api_keys=None)

    def test_health_has_request_id(self):
        r = self.client.get("/health")
        assert "x-request-id" in r.headers

    def test_request_id_is_8_chars(self):
        r = self.client.get("/health")
        assert len(r.headers.get("x-request-id", "")) == 8

    def test_different_requests_have_unique_ids(self):
        ids = {self.client.get("/health").headers.get("x-request-id") for _ in range(5)}
        assert len(ids) == 5  # 모두 고유
