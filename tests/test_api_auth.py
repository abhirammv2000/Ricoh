"""Authentication, and the rate limit that follows the caller's key.

Every claim in api/auth.py is a test here: the service will not start open by accident, a key never appears in a
response or a log, rotation works, a proxy cannot make callers share a limit, and guessing keys is throttled
without locking out callers who have a valid one.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

import api.main as apimod
from api.auth import ANONYMOUS, Authenticator, match_key, parse_keys
from api.main import create_app
from api.readiness import Readiness
from api.settings import Settings, SettingsError
from src.ratelimit import TokenBucketLimiter

KEY_A = "alpha-secret-key-0001"
KEY_B = "bravo-secret-key-0002"
KEYS = f"partner-a:{KEY_A},partner-b:{KEY_B}"


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    apimod._limiter = TokenBucketLimiter(rate_per_sec=10_000.0, capacity=10_000)
    monkeypatch.setattr(apimod, "compute_readiness", lambda require_manifest=True: Readiness(index_version="t", chunks=1))
    calls = []

    async def fine(q):
        calls.append(q)
        return "Choose Network [network_guide.pdf, Page 3]."

    monkeypatch.setattr(apimod, "arun_agent", fine)
    return calls


def keyed_app(**kwargs):
    return create_app(Settings(api_keys=KEYS, fallback_to_retrieval=False, **kwargs))


async def _call(app, method, url, headers=None, client_ip="203.0.113.7", **kwargs):
    # one address for everyone, which is what a service behind a proxy sees
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False, client=(client_ip, 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, headers=headers, **kwargs)


def call(app, method, url, **kwargs):
    return asyncio.run(_call(app, method, url, **kwargs))


def ask(app, key=None, bearer=None, query="how do I configure the network?", **kwargs):
    headers = {}
    if key is not None:
        headers["X-API-Key"] = key
    if bearer is not None:
        headers["Authorization"] = f"Bearer {bearer}"
    return call(app, "POST", "/query", json={"query": query}, headers=headers, **kwargs)


# ---- parsing the keys ----


def test_keys_can_have_labels_or_get_one_made_from_a_hash():
    keys = parse_keys("partner-a:s3cret, plain-secret-value ,, ")
    assert [k.label for k in keys][0] == "partner-a"
    assert keys[1].label.startswith("key-") and "plain-secret-value" not in keys[1].label
    assert len(keys) == 2


def test_a_key_may_contain_colons_after_the_label():
    keys = parse_keys("svc:abc:def")
    assert keys[0].label == "svc"
    assert match_key("abc:def", keys).label == "svc"


def test_duplicate_labels_and_empty_secrets_are_rejected():
    with pytest.raises(ValueError, match="share the label"):
        parse_keys("a:one,a:two")
    with pytest.raises(ValueError, match="empty"):
        parse_keys("a:")


def test_a_key_matches_only_itself_and_any_configured_key_works():
    keys = parse_keys(KEYS)
    assert match_key(KEY_A, keys).label == "partner-a"
    assert match_key(KEY_B, keys).label == "partner-b"
    assert match_key(KEY_A + "x", keys) is None
    assert match_key("", keys) is None
    assert match_key(KEY_A[:-1], keys) is None


# ---- the service will not start open by accident ----


def test_the_service_refuses_to_start_with_no_keys_unless_auth_is_explicitly_off():
    with pytest.raises(SettingsError, match="API_AUTH_DISABLED"):
        create_app(Settings())
    create_app(Settings(auth_disabled=True))  # the laptop case, on purpose
    create_app(Settings(api_keys=KEYS))


def test_a_bad_key_list_stops_startup_with_a_clear_message():
    with pytest.raises(SettingsError, match="API_KEYS is not valid"):
        create_app(Settings(api_keys="a:one,a:two"))


def test_printing_the_settings_does_not_print_the_keys():
    assert KEY_A not in repr(Settings(api_keys=KEYS)) and KEY_A not in str(Settings(api_keys=KEYS))


def test_settings_read_the_keys_and_the_switch_from_the_environment():
    s = Settings.from_env({"API_KEYS": KEYS, "API_AUTH_DISABLED": "false", "DAILY_BUDGET_USD": "5"})
    assert s.api_keys == KEYS and s.auth_disabled is False and s.daily_budget_usd == 5.0
    assert Settings.from_env({"API_AUTH_DISABLED": "true"}).auth_disabled is True
    with pytest.raises(SettingsError, match="DAILY_BUDGET_USD"):
        Settings.from_env({"DAILY_BUDGET_USD": "-1"})


# ---- the endpoints ----


def test_a_request_with_no_key_is_refused_and_never_reaches_the_agent(setup):
    r = ask(keyed_app())
    assert r.status_code == 401 and r.json()["code"] == "unauthorized"
    assert setup == [], "the agent ran for an unauthenticated caller"


def test_a_wrong_key_is_refused_and_the_answer_does_not_say_what_was_wrong(setup):
    r = ask(keyed_app(), key=KEY_A + "x")
    assert r.status_code == 401
    assert KEY_A not in r.text
    assert r.json()["detail"] == "A valid API key is required."
    assert setup == []


def test_a_valid_key_works_in_either_header(setup):
    app = keyed_app()
    assert ask(app, key=KEY_A).status_code == 200
    assert ask(app, bearer=KEY_B).status_code == 200
    assert len(setup) == 2


def test_the_stream_and_metrics_endpoints_need_a_key_too():
    app = keyed_app()
    assert call(app, "POST", "/query/stream", json={"query": "hello there"}).status_code == 401
    assert call(app, "GET", "/metrics").status_code == 401
    assert call(app, "GET", "/metrics", headers={"X-API-Key": KEY_A}).status_code == 200


def test_health_and_ready_stay_open_so_probes_work_with_no_key():
    app = keyed_app()
    assert call(app, "GET", "/health").status_code == 200
    assert call(app, "GET", "/ready").status_code == 200


def test_a_key_is_rotated_by_adding_the_new_one_first_and_removing_the_old_one_last():
    both = create_app(Settings(api_keys=f"old:{KEY_A},new:{KEY_B}", fallback_to_retrieval=False))
    assert ask(both, key=KEY_A).status_code == 200 and ask(both, key=KEY_B).status_code == 200
    only_new = create_app(Settings(api_keys=f"new:{KEY_B}", fallback_to_retrieval=False))
    assert ask(only_new, key=KEY_A).status_code == 401 and ask(only_new, key=KEY_B).status_code == 200


def test_a_key_never_appears_in_the_logs(caplog):
    with caplog.at_level(logging.DEBUG):
        ask(keyed_app(), key="guess-" + KEY_A)
        ask(keyed_app(), key=KEY_A)
    assert KEY_A not in caplog.text and "guess-" + KEY_A not in caplog.text


# ---- guessing keys is throttled, and valid callers are not caught by it ----


def test_repeated_wrong_keys_from_one_address_get_a_429_and_a_retry_after():
    app = keyed_app()
    app.state.authenticate.failures = TokenBucketLimiter(rate_per_sec=0.001, capacity=3)
    codes = [ask(app, key="wrong-%d" % i).status_code for i in range(5)]
    assert codes == [401, 401, 401, 429, 429]
    blocked = ask(app, key="wrong-again")
    assert blocked.json()["code"] == "too_many_failed_attempts" and int(blocked.headers["retry-after"]) > 0


def test_a_valid_key_still_works_from_an_address_that_has_been_guessing():
    app = keyed_app()
    app.state.authenticate.failures = TokenBucketLimiter(rate_per_sec=0.001, capacity=2)
    for i in range(4):
        ask(app, key="wrong-%d" % i)
    assert ask(app, key=KEY_A).status_code == 200


# ---- the rate limit follows the key, not the address ----


def test_two_callers_behind_one_proxy_address_do_not_share_a_rate_limit():
    apimod._limiter = TokenBucketLimiter(rate_per_sec=0.001, capacity=1)
    app = keyed_app()
    # both calls come from the same address (the proxy), with different keys
    assert ask(app, key=KEY_A).status_code == 200
    assert ask(app, key=KEY_A).status_code == 429, "the same key must be limited"
    assert ask(app, key=KEY_B).status_code == 200, "another caller was limited by someone else's traffic"


def test_with_auth_off_the_rate_limit_falls_back_to_the_address():
    apimod._limiter = TokenBucketLimiter(rate_per_sec=0.001, capacity=1)
    app = create_app(Settings(auth_disabled=True, fallback_to_retrieval=False))
    assert ask(app, client_ip="198.51.100.1").status_code == 200
    assert ask(app, client_ip="198.51.100.1").status_code == 429
    assert ask(app, client_ip="198.51.100.2").status_code == 200


def test_the_authenticator_is_a_noop_when_disabled_and_names_the_client_when_enabled():
    class Req:
        class url:
            path = "/query"

        client = None
        headers = {}

        class state:
            pass

    disabled = Authenticator([], enabled=False)
    assert disabled(Req) is ANONYMOUS
