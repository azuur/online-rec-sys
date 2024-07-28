# TODO Use cases to build:
import base64
import json
from logging import getLogger
import logging
from pathlib import Path
import random
from typing import Annotated
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from httpx import AsyncClient
from online_learning_demo.config import PGConfig, TMDBConfig
from psycopg import AsyncConnection

from online_learning_demo.recsys.recsys import (
    OnlineRecSysModel,
    OnlineRecSysRepo,
    TMDBGateway,
)
from fastapi.staticfiles import StaticFiles

logger = getLogger(__file__)
logger.setLevel(logging.DEBUG)

with open(Path(__file__).parent / "static" / "placeholder.jpg", "rb") as f:
    IMG_PLACEHOLDER = f.read()

# Show a mix of recs for user
# GET recs


# Add movies to a batch
# POST feedback
# Should this include auto updating?

# Update user

# because you watched


def get_tmdb_gateway():
    load_dotenv()
    tmdb_config = TMDBConfig()
    httpx_client = AsyncClient()
    return TMDBGateway(httpx_client, tmdb_config)


async def aget_db_conn():
    load_dotenv()
    pg_config = PGConfig()
    return await AsyncConnection.connect(pg_config.uri())


async def get_recsys_repo(aconn: Annotated[AsyncConnection, Depends(aget_db_conn)]):
    return OnlineRecSysRepo(aconn)


async def get_recsys_model():
    return OnlineRecSysModel()


async def get_recs(
    recsys_repo: OnlineRecSysRepo,
    tmdb_gateway: TMDBGateway,
    user_id: int,
    k: int = 20,
    p_random: float = 0.0,
    offset: int = 0,
):
    logger.debug("Getting recommendations for user...")
    n_random = int(p_random * k)
    recs = await recsys_repo.aget_rec_by_user_id(user_id, k - n_random, offset)

    random_recs = []
    if n_random:
        random_recs = await recsys_repo.aget_rec_random_by_user_id(user_id, n_random)

    for r in random_recs:
        i = random.randint(0, len(recs) - 1)
        recs.insert(i, r)

    recs = [
        {"id": r[0], "title": r[1], "synopsis": r[2], "subtitle": r[3], "score": r[4]}
        for r in recs
    ]

    logger.debug("Getting movie posters...")
    movie_ids = [r["id"] for r in recs]
    posters_bytes = await recsys_repo.aget_movie_posters(movie_ids)
    posters_dict = dict(zip(movie_ids, posters_bytes))
    if any(p is None for p in posters_bytes):
        missing_posters_ids = [k for k, v in posters_dict.items() if v is None]
        new_posters_bytes = await tmdb_gateway.aget_movie_posters(missing_posters_ids)
        new_posters_dict = dict(zip(missing_posters_ids, new_posters_bytes))
        await recsys_repo.ainsert_movie_posters(
            ((k, v) for k, v in new_posters_dict.items())
        )
        posters_dict.update(new_posters_dict)

    for k, v in posters_dict.items():
        if v is None:
            posters_dict[k] = IMG_PLACEHOLDER

    for r in recs:
        r["poster_bytes"] = base64.b64encode(posters_dict[r["id"]]).decode("utf-8")
        # syn = r["synopsis"].split()
        # fixed = []
        # for w in syn:
        #     if

    return recs


async def update_user_vec(
    recsys_repo: OnlineRecSysRepo,
    recsys_model: OnlineRecSysModel,
    user_id: int,
    normalized_lr: float = 0.3,
):
    logger.debug("Updating user vector with batch...")
    batch = await recsys_repo.aget_current_batch(user_id)
    user = await recsys_repo.aget_current_user(user_id)

    X = [row[-1] for row in batch]
    Y = [row[2] for row in batch]
    vec = user[0][-1]

    updated_user_emb = recsys_model.calculate_updated(
        vec, (X, Y), normalized_lr=normalized_lr
    )

    new_batch_id = await recsys_repo.amove_current_batch_to_historical(1)

    await recsys_repo.aupdate_user(user_id, new_batch_id, updated_user_emb)
    logger.debug("Updating user vector with batch... SUCCESS")


async def add_to_batch(
    recsys_repo: OnlineRecSysRepo,
    recsys_model: OnlineRecSysModel,
    max_n_in_batch: int,
    user_id: int,
    movie_ids: list[str],
    labels: list[bool],
    normalized_lr: float = 0.2,
):
    logger.debug("Adding movies to batch...")
    await recsys_repo.aadd_to_current_batch(
        user_id=user_id, movie_ids=movie_ids, labels=labels
    )
    logger.debug("Adding movies to batch... SUCCESS")

    n_in_batch = await recsys_repo.aget_current_batch_size(user_id)

    if n_in_batch >= max_n_in_batch:
        await update_user_vec(
            recsys_repo, recsys_model, user_id, normalized_lr=normalized_lr
        )


async def amain_test():
    aconn = await aget_db_conn()
    recsys_repo = await get_recsys_repo(aconn)
    recsys_model = await get_recsys_model()
    tmdb_gateway = get_tmdb_gateway()

    user_id = 1
    k = 15
    max_n_batch = 4

    seq_data = [
        [
            "tt0120382",
            "tt0054518",
            "tt0118819",
            "tt0902272",
            "tt0166924",
            "tt1821549",
            "tt0449059",
            "tt1659337",
            "tt1605783",
        ],
        [True, False, False, False, True, True, True, True, False],
    ]

    for movie_id, label in zip(*seq_data):
        recs = await get_recs(recsys_repo, tmdb_gateway, user_id, k)
        # recs = [(rec[0], rec[1], rec[4], rec[3], rec[2][:100]) for rec in recs]
        print(json.dumps(recs, indent=2))
        await add_to_batch(
            recsys_repo, recsys_model, max_n_batch, user_id, [movie_id], [label]
        )
        _ = input()
    recs = await recsys_repo.aget_rec_by_user_id(user_id, k)
    # recs = [(rec[0], rec[1], rec[4], rec[3], rec[2][:100]) for rec in recs]
    print(json.dumps(recs, indent=2))


app = FastAPI()


MAX_N_BATCH = 1
NORMALIZED_LR = 0.3
templates = Jinja2Templates(directory="online_learning_demo/templates")

app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)


@app.get("/{user_id}")
async def user_home(
    user_id: int,
    request: Request,
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    tmdb_gateway: Annotated[TMDBGateway, Depends(get_tmdb_gateway)],
    limit: int = 15,
    offset: int = 0,
):
    recs = await get_recs(
        recsys_repo=recsys_repo,
        tmdb_gateway=tmdb_gateway,
        user_id=user_id,
        k=limit,
        offset=offset,
    )
    for r in recs:
        r["user_id"] = user_id
    recs[-1]["offset"] = offset + limit
    recs[-1]["limit"] = limit

    return templates.TemplateResponse("home.html", {"movies": recs, "request": request})


@app.get("/next/")
async def _(
    user_id: int,
    request: Request,
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    tmdb_gateway: Annotated[TMDBGateway, Depends(get_tmdb_gateway)],
    limit: int = 15,
    offset: int = 0,
):
    recs = await get_recs(
        recsys_repo=recsys_repo,
        tmdb_gateway=tmdb_gateway,
        user_id=user_id,
        k=limit,
        offset=offset,
    )
    for r in recs:
        r["user_id"] = user_id
    recs[-1]["offset"] = offset + limit
    recs[-1]["limit"] = limit

    return templates.TemplateResponse("rows.html", {"movies": recs, "request": request})


@app.get("/recommendations")
async def _(
    user_id: int,
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    tmdb_gateway: Annotated[TMDBGateway, Depends(get_tmdb_gateway)],
    limit: int = 15,
    offset: int = 0,
):
    return await get_recs(
        recsys_repo=recsys_repo,
        tmdb_gateway=tmdb_gateway,
        user_id=user_id,
        k=limit,
        offset=offset,
    )


@app.post("/batch-feedback/")
async def _(
    user_id: int,
    movie_ids: list[str],
    labels: list[bool],
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    recsys_model: Annotated[OnlineRecSysModel, Depends(get_recsys_model)],
):
    await add_to_batch(
        recsys_repo,
        recsys_model,
        MAX_N_BATCH,
        user_id,
        movie_ids,
        labels,
        normalized_lr=NORMALIZED_LR,
    )


@app.post("/feedback/")
async def _(
    user_id: int,
    movie_id: str,
    label: bool,
    request: Request,
    response: Response,
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    recsys_model: Annotated[OnlineRecSysModel, Depends(get_recsys_model)],
):
    await add_to_batch(
        recsys_repo,
        recsys_model,
        MAX_N_BATCH,
        user_id,
        [movie_id],
        [label],
        normalized_lr=NORMALIZED_LR,
    )

    # response.headers["HX-Refresh"] = "true"

    redirect_url = request.url_for("user_home", user_id=user_id)
    # headers = {"HX-Refresh": "true"}
    return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)


# if __name__ == "__main__":
# asyncio.run(amain_test())
