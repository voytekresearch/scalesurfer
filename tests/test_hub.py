from pathlib import Path

from scalesurfer import fetch_all_models


def test_fetch_all_models_uses_released_hugging_face_paths(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["repo_id"].replace("/", "_") / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    paths = fetch_all_models(fs_versions=7, cache_dir=tmp_path)

    assert set(paths) == {"volume-v7", "stats-v7"}
    assert all(path.is_file() for path in paths.values())
    assert {(call["repo_id"], call["filename"]) for call in calls} == {
        ("rphammonds/scalesurfer-v7", "transunet3d.safetensors"),
        ("rphammonds/scalesurfer-v7", "config.json"),
        ("rphammonds/scalesurfer-stats-v7", "stats_model.safetensors"),
        ("rphammonds/scalesurfer-stats-v7", "config.json"),
    }

