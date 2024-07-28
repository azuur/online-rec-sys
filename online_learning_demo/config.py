from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated

from pydantic import BaseModel, ValidationError
from pydantic.functional_validators import AfterValidator
# sudo docker run --name pgvector -e POSTGRES_PASSWORD=pass -e POSTGRES_USER=user -e POSTGRES_DB=db -d ab8dfd51cedf


class PGConfig(BaseSettings):
    host: str
    port: int
    user: str
    password: str | None
    db_name: str

    model_config = SettingsConfigDict(env_prefix="PG_")

    def uri(self):
        return (
            f"postgresql://{self.user}"
            + (f":{self.password}" if self.password is not None else "")
            + f"@{self.host}:{self.port}/{self.db_name}"
        )


def end_with_slash(s: str):
    return s if s.endswith("/") else s + "/"


TerminatedURL = Annotated[str, AfterValidator(end_with_slash)]


class TMDBConfig(BaseSettings):
    api_key: str
    api_read_access_token: str
    find_url: TerminatedURL
    poster_url: TerminatedURL

    model_config = SettingsConfigDict(env_prefix="TMDB_")
