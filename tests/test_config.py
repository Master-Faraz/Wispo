from whisperflow.config import load_config, project_root


def test_repo_config_loads_the_stage1_model():
    config = load_config(project_root() / "whisperflow.yaml")
    assert config.asr.model in ("tiny.en", "base.en", "small.en")
    assert config.audio.sample_rate == 16000
    assert config.audio.channels == 1
    assert config.streaming.chunk_ms == 1000
    assert config.asr.device == "auto"
