from shared.intercom_engine import IntercomEngine


def test_cached_self_ip_check_does_not_resolve_hostname(monkeypatch):
    engine = IntercomEngine.__new__(IntercomEngine)
    engine._local_ips = frozenset({"192.168.1.25"})

    def fail_if_resolved():
        raise AssertionError("hot-path self-IP check attempted hostname resolution")

    monkeypatch.setattr("shared.intercom_engine.get_local_ips", fail_if_resolved)

    for _ in range(100):
        assert engine._is_self_ip("192.168.1.25") is True
        assert engine._is_self_ip("192.168.1.30") is False


def test_discovery_cache_replaces_previous_addresses():
    engine = IntercomEngine.__new__(IntercomEngine)
    engine._local_ips = frozenset({"10.0.0.5"})

    assert engine._is_self_ip("10.0.0.5") is True
    engine._local_ips = frozenset({"192.168.0.8"})
    assert engine._is_self_ip("10.0.0.5") is False
    assert engine._is_self_ip("192.168.0.8") is True
