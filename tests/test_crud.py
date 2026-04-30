import pytest

def test_domain_config():
    from domain_config import get_all_domains
    domains = get_all_domains()
    assert len(domains) == 4
    assert 'banking' in domains
    assert 'restaurant' in domains

def test_thresholds():
    from domain_config import DOMAINS
    assert DOMAINS['banking']['face_threshold'] == 75.0
    assert DOMAINS['restaurant']['face_threshold'] == 60.0
