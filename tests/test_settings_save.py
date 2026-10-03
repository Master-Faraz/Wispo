from whisperflow.config import DictionaryEntry, load_config, save_user_settings
from whisperflow.settings import installed_models


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


def test_installed_models_are_the_files_on_disk():
    names = [name for name, _size in installed_models()]
    assert "base.en" in names
