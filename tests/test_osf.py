from pathlib import Path

import pytest

from backend.services import osf


def test_extract_osf_guid_variants():
    assert osf.extract_osf_guid("https://osf.io/abc12/") == "abc12"
    assert osf.extract_osf_guid("http://osf.io/abc12") == "abc12"
    assert osf.extract_osf_guid("https://osf.io/abc12/download") == "abc12"
    assert osf.extract_osf_guid("https://osf.io/download/xy9zk/") == "xy9zk"
    assert osf.extract_osf_guid("W3PMJ") == "w3pmj"
    assert osf.extract_osf_guid("https://example.com/foo") is None
    assert osf.extract_osf_guid("") is None


def test_extract_osf_guid_file_browser_urls():
    # A file-browser URL must resolve to the FILE guid, not the parent project /
    # registration (regression: osf.io/jwvrk/files/t6n5s fetched jwvrk instead).
    assert osf.extract_osf_guid("https://osf.io/jwvrk/files/t6n5s") == "t6n5s"
    assert osf.extract_osf_guid("https://osf.io/abc12/files/osfstorage/xy9zk") == "xy9zk"
    assert osf.extract_osf_guid("https://osf.io/abc12/files/osfstorage/xy9zk/?foo=1") == "xy9zk"
    # No specific file in the path → fall back to the node/registration guid.
    assert osf.extract_osf_guid("https://osf.io/abc12/files/") == "abc12"


def test_fetch_registration_flattens_responses(tmp_path):
    def fake_resolver(guid):
        assert guid == "abc12"
        return {
            "type": "registrations",
            "attributes": {
                "title": "My Study",
                "registration_responses": {
                    "q1": "We hypothesize X.",
                    "q2": ["A", "B"],
                },
            },
        }

    path, ext = osf.fetch_osf_preregistration(
        "https://osf.io/abc12/", dest_dir=tmp_path, resolver=fake_resolver
    )
    assert ext == ".txt"
    text = Path(path).read_text(encoding="utf-8")
    assert "My Study" in text
    assert "We hypothesize X." in text
    assert "A" in text and "B" in text


def test_fetch_registration_falls_back_to_registered_meta(tmp_path):
    def fake_resolver(guid):
        return {
            "type": "registrations",
            "attributes": {
                "registered_meta": {"q1": {"value": "Legacy answer."}},
            },
        }

    path, ext = osf.fetch_osf_preregistration(
        "osf.io/abc12", dest_dir=tmp_path, resolver=fake_resolver
    )
    assert "Legacy answer." in Path(path).read_text(encoding="utf-8")


def test_fetch_file_downloads(tmp_path):
    def fake_resolver(guid):
        return {"type": "files", "attributes": {"name": "prereg.pdf"}, "links": {"download": "https://x/d"}}

    def fake_downloader(data, dest_dir, guid):
        target = Path(dest_dir) / f"{guid}.pdf"
        target.write_bytes(b"%PDF-1.4 fake")
        return str(target), ".pdf"

    path, ext = osf.fetch_osf_preregistration(
        "https://osf.io/abc12/",
        dest_dir=tmp_path,
        resolver=fake_resolver,
        file_downloader=fake_downloader,
    )
    assert ext == ".pdf"
    assert path.endswith(".pdf")


def test_fetch_project_raises(tmp_path):
    with pytest.raises(ValueError, match="project"):
        osf.fetch_osf_preregistration(
            "https://osf.io/abc12/", dest_dir=tmp_path, resolver=lambda g: {"type": "nodes", "attributes": {}}
        )


def test_fetch_bad_link_raises(tmp_path):
    with pytest.raises(ValueError, match="OSF"):
        osf.fetch_osf_preregistration("not-a-link", dest_dir=tmp_path, resolver=lambda g: {})
