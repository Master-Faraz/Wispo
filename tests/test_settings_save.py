from whisperflow.config import DictionaryEntry, load_config, save_user_settings
from whisperflow.models import installed_models, model_names


def test_save_updates_model_and_dictionary_and_keeps_audio(tmp_path):
    path = tmp_path / "whisperflow.yaml"
    path.write_text(
        "audio:\n  sample_rate: 16000\n  device: auto\nasr:\n  model: tiny.en\n  model_path: models/ggml-tiny.en.bin\n",
        encoding="utf-8",
    )
    save_user_settings(
        path,
        "base.en",
        [DictionaryEntry("github", "GitHub", True), DictionaryEntry("  ", "skip", True)],
    )
    config = load_config(path)
    assert config.asr.model == "base.en"
    assert config.asr.model_path == "models/ggml-base.en.bin"
    assert config.audio.sample_rate == 16000
    assert config.audio.device == "auto"
    assert config.dictionary == (DictionaryEntry("github", "GitHub", True),)
    assert config.hotkey.keys == ("ctrl", "super")


def test_save_stores_the_shortcut(tmp_path):
    path = tmp_path / "whisperflow.yaml"
    path.write_text("asr:\n  model: base.en\n  model_path: models/ggml-base.en.bin\n", encoding="utf-8")
    save_user_settings(path, "base.en", [], ("alt", "space"))
    config = load_config(path)
    assert config.hotkey.keys == ("alt", "space")


def test_installed_models_are_the_files_on_disk():
    names = [name for name, _size in installed_models()]
    assert "base.en" in names
    assert "medium.en" in model_names()
    assert "large-v3-turbo" in model_names()
    assert "large-v3-turbo-q5_0" in model_names()
    assert "distil-small.en" in model_names()
    assert "distil-medium.en" in model_names()
    assert "distil-large-v3" in model_names()
    assert "distil-large-v3.5" in model_names()


def test_save_hardware_and_streaming_settings(tmp_path):
    path = tmp_path / "whisperflow.yaml"
    path.write_text("asr:\n  model: base.en\n  model_path: models/ggml-base.en.bin\n", encoding="utf-8")
    save_user_settings(
        path=path,
        model="base.en",
        entries=[],
        threads=8,
        device="cpu",
        gpu_device=1,
        language="es",
        streaming_settings={"chunk_ms": 500, "length_ms": 3000},
    )
    config = load_config(path)
    assert config.asr.threads == 8
    assert config.asr.device == "cpu"
    assert config.asr.gpu_device == 1
    assert config.asr.language == "es"
    assert config.streaming.chunk_ms == 500
    assert config.streaming.length_ms == 3000
