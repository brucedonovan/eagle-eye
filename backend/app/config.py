from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Eagle Eye"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_origin: str = "http://localhost:5173"

    data_dir: Path = Path("./data")
    database_url: str = "sqlite+aiosqlite:///./data/eagle_eye.db"
    job_backend: str = "local"
    redis_url: str = "redis://localhost:6379/0"

    osm_user_agent: str = "EagleEye/0.1 (golf-course-vectorization; local-dev)"
    nominatim_url: str = "https://nominatim.openstreetmap.org"
    overpass_url: str = "https://overpass.openstreetmap.fr/api/interpreter"
    overpass_fallbacks: str = (
        "https://overpass-api.de/api/interpreter,"
        "https://overpass.kumi.systems/api/interpreter"
    )

    imagery_max_zoom: int = 18
    imagery_max_tiles: int = 400
    # Small crops around every pin (refine OSM greens + fill gaps). z19 is ~0.3 m in Lisbon.
    imagery_pin_zoom: int = 19
    imagery_pin_radius_m: float = 48.0

    mapbox_token: str = ""
    google_maps_key: str = ""
    planet_api_key: str = ""
    nearmap_api_key: str = ""
    maxar_api_key: str = ""

    segmentation_backend: str = "dual_imagery_fusion"
    segmentation_weights: str = ""

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


settings = Settings()
