import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_health_ping(client):
    response = client.get('/api/health/ping')
    assert response.status_code == 200

def test_root(client):
    response = client.get('/')
    assert response.status_code == 200
