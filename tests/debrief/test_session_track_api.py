from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from debrief.config import Settings
from debrief.asr import FakeASRAdapter
from debrief.main import create_app


@pytest.fixture()
def client(tmp_path: Path):
    database_path = tmp_path / "test.db"
    app = create_app(
        Settings(
            database_url=f"sqlite:///{database_path.as_posix()}",
            media_root=tmp_path / "media",
            auto_create_schema=True,
        ),
        asr_adapter=FakeASRAdapter(),
    )
    with TestClient(app) as test_client:
        yield test_client
    app.state.engine.dispose()


def create_session(client: TestClient, **overrides):
    payload = {"title": "Simülasyon 1", **overrides}
    response = client.post("/api/v1/sessions", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_health(client: TestClient):
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_session_defaults_to_local_only(client: TestClient):
    session = create_session(client)

    assert session["privacy_mode"] == "yalnız_yerel"
    assert session["status"] == "taslak"
    assert session["timezone_name"] == "Europe/Istanbul"


def test_create_and_list_tracks_with_signed_offsets(client: TestClient):
    session = create_session(client)
    video = client.post(
        f"/api/v1/sessions/{session['id']}/tracks",
        json={
            "type": "video",
            "name": "Kamera A",
            "media_path": "media/session-1/camera-a.mp4",
            "duration_ms": 1_200_000,
        },
    )
    assert video.status_code == 201, video.text

    audio = client.post(
        f"/api/v1/sessions/{session['id']}/tracks",
        json={
            "type": "ses",
            "name": "Öğrenci odası",
            "media_path": "media/session-1/student.wav",
            "offset_ms": 1250,
            "duration_ms": 1_198_750,
        },
    )
    assert audio.status_code == 201, audio.text

    tracks = client.get(f"/api/v1/sessions/{session['id']}/tracks")
    assert tracks.status_code == 200
    assert [item["type"] for item in tracks.json()] == ["video", "ses"]

    update = client.patch(
        f"/api/v1/tracks/{audio.json()['id']}/offset", json={"offset_ms": -350}
    )
    assert update.status_code == 200
    assert update.json()["offset_ms"] == -350


def test_child_track_must_belong_to_same_session(client: TestClient):
    first = create_session(client, title="Birinci")
    second = create_session(client, title="İkinci")
    audio = client.post(
        f"/api/v1/sessions/{first['id']}/tracks",
        json={"type": "ses", "name": "Ses", "media_path": "one.wav"},
    ).json()

    response = client.post(
        f"/api/v1/sessions/{second['id']}/tracks",
        json={"type": "transkript", "name": "Yanlış ilişki", "parent_track_id": audio["id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Üst kanal aynı oturuma ait olmalıdır"


def test_media_track_requires_media_path(client: TestClient):
    session = create_session(client)
    response = client.post(
        f"/api/v1/sessions/{session['id']}/tracks",
        json={"type": "ses", "name": "Eksik dosya"},
    )
    assert response.status_code == 422


def test_unknown_session_returns_not_found(client: TestClient):
    response = client.get("/api/v1/sessions/not-found")
    assert response.status_code == 404


def test_upload_and_probe_wav_media(client: TestClient, tmp_path: Path):
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000)

    session = create_session(client)
    response = client.post(
        f"/api/v1/sessions/{session['id']}/media",
        content=buffer.getvalue(),
        headers={
            "Content-Type": "audio/wav",
            "X-Filename": "ogrenci.wav",
            "X-Track-Name": "%C3%96%C4%9Frenci%20sesi",
            "X-Track-Type": "ses",
            "X-Room": "Oda 1",
        },
    )

    assert response.status_code == 201, response.text
    track = response.json()
    assert track["duration_ms"] == 1000
    assert track["sample_rate"] == 16_000
    assert track["channel_count"] == 1
    assert track["codec_name"] == "pcm_s16le"
    assert len(track["checksum_sha256"]) == 64
    media = client.get(f"/api/v1/tracks/{track['id']}/media")
    assert media.status_code == 200
    assert media.content == buffer.getvalue()


def test_rejects_invalid_media_and_removes_partial_file(client: TestClient):
    session = create_session(client)
    response = client.post(
        f"/api/v1/sessions/{session['id']}/media",
        content=b"not a wave file",
        headers={
            "Content-Type": "audio/wav",
            "X-Filename": "broken.wav",
            "X-Track-Name": "Bozuk ses",
            "X-Track-Type": "ses",
        },
    )
    assert response.status_code == 422
    assert "FFprobe" in response.json()["detail"]


def test_rejects_wrong_media_kind(client: TestClient):
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(b"\x00\x00" * 800)
    session = create_session(client)
    response = client.post(
        f"/api/v1/sessions/{session['id']}/media",
        content=buffer.getvalue(),
        headers={
            "Content-Type": "audio/wav",
            "X-Filename": "audio.wav",
                "X-Track-Name": "Yanl%C4%B1%C5%9F%20video",
            "X-Track-Type": "video",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Video kanalı için görüntü akışı bulunamadı"


def test_video_upload_creates_linked_16k_mono_audio_track(client: TestClient, tmp_path: Path):
    import subprocess

    video = tmp_path / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", str(video),
    ], check=True)
    session = create_session(client)
    response = client.post(
        f"/api/v1/sessions/{session['id']}/media",
        content=video.read_bytes(),
        headers={"Content-Type":"video/mp4","X-Filename":"sample.mp4","X-Track-Name":"Kamera A","X-Track-Type":"video"},
    )
    assert response.status_code == 201, response.text
    video_track = response.json()
    tracks = client.get(f"/api/v1/sessions/{session['id']}/tracks").json()
    audio = next(item for item in tracks if item["type"] == "ses")
    assert audio["parent_track_id"] == video_track["id"]
    assert audio["sample_rate"] == 16000
    assert audio["channel_count"] == 1
    assert audio["codec_name"] == "pcm_s16le"
    assert abs(audio["duration_ms"] - video_track["duration_ms"]) <= 100
    thumbnail = client.get(f"/api/v1/tracks/{video_track['id']}/thumbnail?time_ms=500")
    assert thumbnail.status_code == 200, thumbnail.text
    assert thumbnail.headers["content-type"].startswith("image/jpeg")
    assert len(thumbnail.content) > 100
    filmstrip = client.get(f"/api/v1/tracks/{video_track['id']}/filmstrip")
    assert filmstrip.status_code == 200, filmstrip.text
    assert filmstrip.headers["content-type"].startswith("image/jpeg")
    assert len(filmstrip.content) > 100
    waveform = client.get(f"/api/v1/tracks/{audio['id']}/waveform")
    assert waveform.status_code == 200, waveform.text
    assert waveform.headers["content-type"].startswith("image/png")
    assert len(waveform.content) > 100


def test_transcribes_audio_into_timestamped_segments(client: TestClient):
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000)
    session = create_session(client)
    audio = client.post(
        f"/api/v1/sessions/{session['id']}/media",
        content=buffer.getvalue(),
        headers={
            "Content-Type": "audio/wav", "X-Filename": "test.wav",
            "X-Track-Name": "Test sesi", "X-Track-Type": "ses",
        },
    ).json()

    response = client.post(f"/api/v1/tracks/{audio['id']}/transcriptions")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["model"] == "deterministic-test"
    assert result["output_track"]["type"] == "transkript"
    assert [item["segment_ref"] for item in result["segments"]] == ["S000001", "S000002"]
    assert result["segments"][1]["start_ms"] == 1500

    listed = client.get(f"/api/v1/tracks/{result['output_track']['id']}/segments")
    assert listed.status_code == 200
    assert len(listed.json()) == 2

    indexed = client.post(f"/api/v1/tracks/{result['output_track']['id']}/indexes")
    assert indexed.status_code == 200, indexed.text
    index = indexed.json()
    assert index["chapters"][0]["start_ms"] == 0
    assert index["events"][0]["event_time_ms"] == 1500
    assert index["events"][0]["playback_time_ms"] == 0
    assert index["events"][0]["status"] == "öneri"
    assert index["events"][0]["source"] == "llm"
    assert index["events"][0]["source_segment_ref"] == "S000002"
    assert index["events"][0]["asr_cue"] == "Öğrenci değerlendirme yaptı."
    assert index["events"][0]["evidence_start_ms"] == 1500

    cached = client.post(f"/api/v1/tracks/{result['output_track']['id']}/indexes")
    assert cached.status_code == 200
    assert cached.json()["cached"] is True
    assert cached.json()["index_run_id"] == index["index_run_id"]

    # Bookmark evidence is frozen; later segment drift cannot change the inspector cue.
    from debrief.models import TranscriptSegment
    with client.app.state.session_factory() as db:
        segment = db.get(TranscriptSegment, index["events"][0]["source_segment_id"])
        segment.segment_ref = "S009999"
        segment.text = "Sonradan değişen ve bookmark'a yansımaması gereken metin."
        db.commit()
    frozen = client.get(f"/api/v1/sessions/{session['id']}/events").json()[0]
    assert frozen["source_segment_ref"] == "S000002"
    assert frozen["asr_cue"] == "Öğrenci değerlendirme yaptı."
    assert frozen["event_time_ms"] == 1500

    events = client.get(f"/api/v1/tracks/{index['output_track_id']}/events")
    assert events.status_code == 200
    assert len(events.json()) == 1

    event_id = index["events"][0]["id"]
    approved = client.patch(
        f"/api/v1/events/{event_id}/review",
        json={"status": "onaylı", "reviewed_by": "manuel_test"},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "onaylı"
    assert approved.json()["reviewed_by"] == "manuel_test"
    assert approved.json()["reviewed_at"] is not None

    invalid = client.patch(
        f"/api/v1/events/{event_id}/review",
        json={"status": "öneri", "reviewed_by": "manuel_test"},
    )
    assert invalid.status_code == 422


def test_asr_and_index_can_run_through_background_queue(client: TestClient):
    import io
    import time
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000)
    session = create_session(client, title="Kuyruk testi")
    audio = client.post(
        f"/api/v1/sessions/{session['id']}/media", content=buffer.getvalue(),
        headers={"Content-Type":"audio/wav", "X-Filename":"queue.wav",
                 "X-Track-Name":"Kuyruk sesi", "X-Track-Type":"ses"},
    ).json()

    def wait(job_id):
        for _ in range(100):
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            if job["status"] in {"tamamlandı", "hatalı"}:
                return job
            time.sleep(0.02)
        raise AssertionError("Arka plan işi zamanında tamamlanmadı")

    queued = client.post(f"/api/v1/tracks/{audio['id']}/jobs/asr")
    assert queued.status_code == 202, queued.text
    asr_job = wait(queued.json()["id"])
    assert asr_job["status"] == "tamamlandı", asr_job
    transcript_id = asr_job["result"]["output_track"]["id"]

    index_queued = client.post(f"/api/v1/tracks/{transcript_id}/jobs/index")
    assert index_queued.status_code == 202, index_queued.text
    index_job = wait(index_queued.json()["id"])
    assert index_job["status"] == "tamamlandı", index_job
    assert index_job["result"]["events"][0]["asr_cue"]


def _wav_bytes(seconds: int = 1) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000 * seconds)
    return buffer.getvalue()


def test_two_audio_channels_create_separate_transcripts_and_indexes(client: TestClient):
    session = create_session(client, title="T10 çift ses kanalı")
    outputs = []
    for filename, name in (("ogrenci.wav", "Öğrenci sesi"), ("egitmen.wav", "Eğitmen sesi")):
        media = client.post(
            f"/api/v1/sessions/{session['id']}/media", content=_wav_bytes(),
            headers={"Content-Type":"audio/wav", "X-Filename":filename,
                     "X-Track-Name":name.replace("Ö", "%C3%96").replace("ğ", "%C4%9F").replace("ı", "%C4%B1"), "X-Track-Type":"ses"},
        )
        assert media.status_code == 201, media.text
        transcript = client.post(f"/api/v1/tracks/{media.json()['id']}/transcriptions")
        assert transcript.status_code == 200, transcript.text
        index = client.post(f"/api/v1/tracks/{transcript.json()['output_track']['id']}/indexes")
        assert index.status_code == 200, index.text
        outputs.append((transcript.json()["output_track"]["id"], index.json()["output_track_id"]))
    assert outputs[0][0] != outputs[1][0]
    assert outputs[0][1] != outputs[1][1]


def test_typed_instructor_note_bypasses_asr_and_llm(client: TestClient):
    session = create_session(client, title="T11 eğitmen notu")
    response = client.post(
        f"/api/v1/sessions/{session['id']}/instructor-notes",
        json={"text":"Öğrenci alerji öyküsünü sormayı atladı.",
              "event_time_ms":12340, "instructor_name":"Dr. Aylin"},
    )
    assert response.status_code == 201, response.text
    event = response.json()["event"]
    assert event["source"] == "eğitmen_notu"
    assert event["status"] == "onaylı"
    assert event["event_time_ms"] == 12340
    assert event["playback_time_ms"] == 7340
    assert event["asr_cue"] is None
    assert event["reviewed_by"] == "Dr. Aylin"
    combined = client.get(f"/api/v1/sessions/{session['id']}/events")
    assert combined.status_code == 200
    assert [item["id"] for item in combined.json()] == [event["id"]]
    assert combined.json()[0]["track_name"].endswith("Bookmark")


def test_phase2_timeline_and_manual_bookmark_crud(client: TestClient):
    session = create_session(client, title="Faz 2 timeline")
    created = client.post(
        f"/api/v1/sessions/{session['id']}/bookmarks",
        json={"title":"Oksijen başlandı", "category":"mudahale",
              "event_time_ms":12500, "description":"Eğitmen gözlemi",
              "instructor_name":"Dr. Aylin"},
    )
    assert created.status_code == 201, created.text
    event = created.json()
    assert event["source"] == "eğitmen"
    assert event["status"] == "onaylı"
    assert event["playback_time_ms"] == 7500
    assert event["asr_cue"] is None

    timeline = client.get(f"/api/v1/sessions/{session['id']}/timeline")
    assert timeline.status_code == 200, timeline.text
    assert timeline.json()["session"]["title"] == "Faz 2 timeline"
    assert [item["id"] for item in timeline.json()["events"]] == [event["id"]]

    updated = client.patch(
        f"/api/v1/events/{event['id']}",
        json={"title":"Oksijen desteği başlandı", "category":"guvenlik",
              "event_time_ms":18000, "description":"Düzenlendi"},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["title"] == "Oksijen desteği başlandı"
    assert updated.json()["event_time_ms"] == 18000
    assert updated.json()["playback_time_ms"] == 13000

    deleted = client.delete(f"/api/v1/events/{event['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/sessions/{session['id']}/events").json() == []


def test_phase2_device_events_are_suggestions_and_share_a_device_track(client: TestClient):
    session = create_session(client, title="Cihaz olayları")
    first = client.post(
        f"/api/v1/sessions/{session['id']}/device-events",
        json={"title":"SpO2 alarmı", "category":"cihaz", "event_time_ms":12500,
              "description":"SpO2 yüzde 88", "device_name":"Hasta monitörü",
              "confidence":0.94, "external_event_id":"MON-001"},
    )
    assert first.status_code == 201, first.text
    event = first.json()
    assert event["source"] == "cihaz"
    assert event["status"] == "öneri"
    assert event["confidence"] == 0.94
    assert event["playback_time_ms"] == 7500
    assert event["source_segment_ref"] == "MON-001"
    assert event["asr_cue"] == "SpO2 yüzde 88"

    second = client.post(
        f"/api/v1/sessions/{session['id']}/device-events",
        json={"title":"Nabız alarmı", "event_time_ms":22000,
              "device_name":"Hasta monitörü"},
    )
    assert second.status_code == 201, second.text
    timeline = client.get(f"/api/v1/sessions/{session['id']}/timeline").json()
    assert {item["id"] for item in timeline["events"]} == {
        event["id"], second.json()["id"]
    }
    tracks = client.get(f"/api/v1/sessions/{session['id']}/tracks").json()
    assert len([track for track in tracks if track["type"] == "cihaz"]) == 1
    assert len([track for track in tracks if track["type"] == "indeks"]) == 1


def test_external_device_event_id_is_idempotent(client: TestClient):
    session = create_session(client)
    payload = {
        "title": "Şok uygulandı", "event_time_ms": 12_500,
        "device_name": "Defibrilatör", "external_event_id": "DEFIB-EVENT-000001",
    }
    first = client.post(f"/api/v1/sessions/{session['id']}/device-events", json=payload)
    second = client.post(f"/api/v1/sessions/{session['id']}/device-events", json=payload)
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["id"] == first.json()["id"]
    assert len(client.get(f"/api/v1/sessions/{session['id']}/events").json()) == 1


def test_system_event_is_normalized_and_idempotent(client: TestClient):
    session = create_session(client)
    payload = {
        "title": "Kamera bağlantısı kesildi", "category": "sistem",
        "event_time_ms": 22_000, "description": "Kamera A sinyal kaybı",
        "system_name": "Capture Node 1", "external_event_id": "health-camera-a-22",
    }
    first = client.post(f"/api/v1/sessions/{session['id']}/system-events", json=payload)
    second = client.post(f"/api/v1/sessions/{session['id']}/system-events", json=payload)
    assert first.status_code == 201, first.text
    assert second.json()["id"] == first.json()["id"]
    assert first.json()["source"] == "sistem"
    assert first.json()["playback_time_ms"] == 17_000

    tracks = client.get(f"/api/v1/sessions/{session['id']}/tracks").json()
    assert len([track for track in tracks if track["type"] == "kontrol_listesi"]) == 1
    assert len([track for track in tracks if track["type"] == "indeks"]) == 1


def test_phase2_timeline_uses_latest_index_per_source_track(client: TestClient):
    session = create_session(client, title="Canonical timeline")
    audio = client.post(
        f"/api/v1/sessions/{session['id']}/media", content=_wav_bytes(),
        headers={"Content-Type":"audio/wav", "X-Filename":"source.wav",
                 "X-Track-Name":"Kaynak", "X-Track-Type":"ses"},
    ).json()
    transcript = client.post(f"/api/v1/tracks/{audio['id']}/transcriptions").json()
    source_id = transcript["output_track"]["id"]
    first = client.post(f"/api/v1/tracks/{source_id}/indexes").json()
    second = client.post(f"/api/v1/tracks/{source_id}/indexes").json()
    assert second["cached"] is True
    timeline = client.get(f"/api/v1/sessions/{session['id']}/timeline").json()
    assert len(timeline["events"]) == len(first["events"])
