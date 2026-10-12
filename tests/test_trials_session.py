from backend.services import trials


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _CookieGatedSession:
    """Mimics ClinicalTrials.gov: history endpoints need the cookie set by the study call."""

    instances: list["_CookieGatedSession"] = []

    def __init__(self):
        self.has_cookie = False
        self.urls: list[str] = []
        _CookieGatedSession.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, timeout=None):
        self.urls.append(url)
        if "/api/v2/studies/" in url:
            self.has_cookie = True
            return _Response(
                {"protocolSection": {"statusModule": {"startDateStruct": {"date": "2020-06"}}}}
            )
        assert self.has_cookie, f"history request without session cookie: {url}"
        if url.endswith("/history"):
            return _Response(
                {"changes": [{"version": 0, "date": "2020-01-01"}, {"version": 1, "date": "2021-01-01"}]}
            )
        return _Response({"study": {"protocolSection": {"marker": "v0"}}})


def test_history_requests_share_session_cookie(monkeypatch):
    _CookieGatedSession.instances.clear()
    monkeypatch.setattr(trials.requests, "Session", _CookieGatedSession)

    def _no_bare_get(*args, **kwargs):
        raise AssertionError("trial fetches must go through the shared session")

    monkeypatch.setattr(trials.requests, "get", _no_bare_get)

    study, metadata = trials.fetch_trial_json_with_metadata("NCT05597618")

    assert len(_CookieGatedSession.instances) == 1
    assert len(_CookieGatedSession.instances[0].urls) == 3
    assert metadata["source"] == "history"
    assert metadata["selected_version"] == 0
    assert study == {"protocolSection": {"marker": "v0"}}
