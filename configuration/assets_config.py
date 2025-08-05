from __future__ import annotations

from pathlib import Path
from typing import Any
from pydantic._internal._utils import deep_update
from pydantic_settings import BaseSettings, SettingsConfigDict, InitSettingsSource
from pydantic_settings.sources import ENV_FILE_SENTINEL, DotenvType

from configuration.pydantic_custom_sources.yaml_config_settings_source import YamlConfigSettingsSource

from src.entities.asset_entity import AssetEntity


class AssetsConfig(BaseSettings):
    assets: list[AssetEntity]

    model_config = SettingsConfigDict(
        env_file="assets.yaml",
        env_file_encoding="utf-8"
    )

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
    ):
        return (
            init_settings,
            env_settings
        )

    def _settings_build_values(
            self,
            init_kwargs: dict[str, Any],
            _case_sensitive: bool | None = None,
            _env_prefix: str | None = None,
            _env_file: DotenvType | None = None,
            _env_file_encoding: str | None = None,
            _env_nested_delimiter: str | None = None,
            _secrets_dir: str | Path | None = None,
            **kwargs: Any,
    ) -> dict[str, Any]:
        env_file = _env_file if _env_file != ENV_FILE_SENTINEL else self.model_config.get('env_file')
        env_file_encoding = (
            _env_file_encoding if _env_file_encoding is not None else self.model_config.get('env_file_encoding')
        )

        init_settings = InitSettingsSource(self.__class__, init_kwargs=init_kwargs)
        yaml_settings = YamlConfigSettingsSource(
            self.__class__,
            yaml_file=env_file,
            yaml_file_encoding=env_file_encoding
        )

        sources = self.settings_customise_sources(
            self.__class__,
            init_settings=init_settings,
            env_settings=yaml_settings,
            dotenv_settings=yaml_settings,
            file_secret_settings=yaml_settings
        )

        if sources:
            return deep_update(*reversed([source() for source in sources]))
        return {}
