from fastapi.testclient import TestClient

from web.app import app


client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert res.json()["ffmpeg"] is True


def test_meta_defaults_arabic():
    res = client.get("/api/meta")
    assert res.status_code == 200
    data = res.json()
    assert data["defaults"]["lang"] == "ar"
    assert data["defaults"]["voice"] == "ar-SA-HamedNeural"
    assert any(lang["code"] == "ar" for lang in data["languages"])
    assert len(data["stages"]) == 7


def test_index_serves_studio():
    res = client.get("/")
    assert res.status_code == 200
    assert "ProStudio" in res.text


def test_job_requires_source():
    res = client.post("/api/jobs", data={"lang": "ar"})
    assert res.status_code == 400
