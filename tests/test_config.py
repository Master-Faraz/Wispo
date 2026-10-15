from wispo.config import config_path, load_config, project_root


def test_repo_config_loads_the_stage1_model():
    config = load_config(project_root() / "wispo.yaml")
    assert config.asr.model
    assert config.asr.model_path.endswith(".bin")
    assert config.audio.sample_rate == 16000
    assert config.audio.channels == 1
    assert config.streaming.chunk_ms == 1000
    assert config.asr.device == "auto"


def test_packaged_data_directory_uses_a_writable_settings_copy(monkeypatch, tmp_path):
    data_dir = tmp_path / "wispo-data"
    monkeypatch.setenv("WISPO_DATA_DIR", str(data_dir))

    target = config_path(create=True)

    assert target == data_dir / "wispo.yaml"
    assert target.is_file()
    assert load_config().asr.model == "small.en"
