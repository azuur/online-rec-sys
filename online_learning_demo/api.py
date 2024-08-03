# TODO Use cases to build:
import ast
import base64
import json
from logging import getLogger
import logging
from pathlib import Path
import random
from typing import Annotated
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, Request, Response, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from httpx import AsyncClient
from langchain_openai import OpenAIEmbeddings
import numpy as np
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


async def get_recsys_model(
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
):
    return OnlineRecSysModel(recsys_repo)


class Vectorizer:
    def __init__(self, model: OpenAIEmbeddings) -> None:
        self.model = model

    async def __call__(self, doc: str) -> list[float]:
        return (await self.model.aembed_documents([doc]))[0]


def get_vectorizer():
    model = OpenAIEmbeddings()
    return Vectorizer(model)


async def get_movies(
    query: str | None,
    recsys_repo: OnlineRecSysRepo,
    recsys_model: OnlineRecSysModel,
    tmdb_gateway: TMDBGateway,
    vectorizer: Vectorizer,
    user_id: int,
    k: int = 20,
    p_random: float = 0.0,
    offset: int = 0,
):
    if not query:
        return await get_recs(
            recsys_repo=recsys_repo,
            recsys_model=recsys_model,
            tmdb_gateway=tmdb_gateway,
            user_id=user_id,
            k=k,
            p_random=p_random,
            offset=offset,
        )

    return await get_from_query(
        recsys_repo=recsys_repo,
        recsys_model=recsys_model,
        tmdb_gateway=tmdb_gateway,
        vectorizer=vectorizer,
        user_id=user_id,
        query=query,
        k=k,
        offset=offset,
    )


async def get_from_query(
    recsys_repo: OnlineRecSysRepo,
    recsys_model: OnlineRecSysModel,
    tmdb_gateway: TMDBGateway,
    vectorizer: Vectorizer,
    user_id: int,
    query: str,
    k: int = 20,
    offset: int = 0,
):
    logger.debug("Getting items from search...")

    vec = await vectorizer(query)  # TODO: cache this
    raw_recs = await recsys_repo.aget_rec_from_vector(vec, k, offset)
    recs = [
        {"id": r[0], "title": r[1], "synopsis": r[2], "subtitle": r[3], "score": r[4]}
        for r in raw_recs
    ]
    vecs = [ast.literal_eval(r[-1]) for r in raw_recs]

    model = await recsys_model.aload_model(user_id)
    if model is not None:
        scores = recsys_model.score_content(model, vecs)
        for rec, score in zip(recs, scores):
            rec["score"] = score

    return await _add_movie_posters(recs, recsys_repo, tmdb_gateway)


async def get_recs(
    recsys_repo: OnlineRecSysRepo,
    recsys_model: OnlineRecSysModel,
    tmdb_gateway: TMDBGateway,
    user_id: int,
    k: int = 20,
    p_random: float = 0.0,
    offset: int = 0,
):
    model = await recsys_model.aload_model(user_id)
    if model is None:
        return []
    model_vec = model.coef_[0].tolist()
    breakpoint()
    logger.debug("Getting recommendations for user...")
    n_random = int(p_random * k)
    raw_recs = await recsys_repo.aget_rec_from_vector(model_vec, k - n_random, offset)

    # This happens when a user doesn't exist yet
    if not raw_recs:
        return raw_recs

    random_recs = []
    if n_random:
        random_recs = await recsys_repo.aget_rec_random_by_user_id(user_id, n_random)

    for r in random_recs:
        i = random.randint(0, len(raw_recs) - 1)
        raw_recs.insert(i, r)

    vecs = [ast.literal_eval(r[-1]) for r in raw_recs]

    recs = [
        {"id": r[0], "title": r[1], "synopsis": r[2], "subtitle": r[3], "score": r[4]}
        for r in raw_recs
    ]

    scores = recsys_model.score_content(model, vecs)
    for rec, score in zip(recs, scores):
        rec["score"] = score

    return await _add_movie_posters(recs, recsys_repo, tmdb_gateway)


async def _add_movie_posters(
    recs: list[dict], recsys_repo: OnlineRecSysRepo, tmdb_gateway: TMDBGateway
):
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
):
    logger.debug("Updating user vector with batch...")
    batch = await recsys_repo.aget_current_batch(user_id)

    X = [row[-1] for row in batch]
    Y = [row[2] for row in batch]

    model = await recsys_model.aload_model(user_id)
    model = recsys_model.calculate_updated(model, (X, Y))
    model_bytes = recsys_model.dump_model(model)

    new_batch_id = await recsys_repo.amove_current_batch_to_historical(user_id)

    await recsys_repo.aupdate_user(user_id, new_batch_id, model_bytes)
    logger.debug("Updating user vector with batch... SUCCESS")


async def add_to_batch(
    recsys_repo: OnlineRecSysRepo,
    recsys_model: OnlineRecSysModel,
    max_n_in_batch: int,
    user_id: int,
    movie_ids: list[str],
    labels: list[bool],
):
    logger.debug("Adding movies to batch...")
    await recsys_repo.aadd_to_current_batch(
        user_id=user_id, movie_ids=movie_ids, labels=labels
    )
    logger.debug("Adding movies to batch... SUCCESS")

    batch_labels = [b[2] for b in (await recsys_repo.aget_current_batch(user_id))]
    n_in_batch = len(batch_labels)

    batch_has_both = any(batch_labels) and not all(batch_labels)
    if n_in_batch >= max_n_in_batch and batch_has_both:
        await update_user_vec(recsys_repo, recsys_model, user_id)
        return True
    return False


async def initialize_user(
    recsys_repo: OnlineRecSysRepo, recsys_model: OnlineRecSysModel, user_id: int
):
    if user_id > 100 or user_id < 1:
        raise Exception("We only support 100 users for now")

    batch = await recsys_repo.aget_user_feedback_history(user_id)
    batch = batch or None
    new_model = recsys_model.initialize_new_model(batch)
    new_model_bytes = recsys_model.dump_model(new_model)

    await recsys_repo.ainitialize_user(user_id, new_model_bytes)


async def amain_test():
    aconn = await aget_db_conn()
    recsys_repo = await get_recsys_repo(aconn)
    recsys_model = await get_recsys_model()
    tmdb_gateway = get_tmdb_gateway()
    vectorizer = Vectorizer(OpenAIEmbeddings())

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
        recs = await get_movies(
            None, recsys_repo, recsys_model, tmdb_gateway, vectorizer, user_id, k
        )
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


MAX_N_BATCH = 5
NORMALIZED_LR = 0.3
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)


@app.get("/home/{user_id}/")
async def user_home(
    user_id: int,
    request: Request,
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    recsys_model: Annotated[OnlineRecSysRepo, Depends(get_recsys_model)],
    tmdb_gateway: Annotated[TMDBGateway, Depends(get_tmdb_gateway)],
    vectorizer: Annotated[Vectorizer, Depends(get_vectorizer)],
    limit: int = 15,
    offset: int = 0,
):
    query = None
    recs = await get_movies(
        query=query,
        recsys_repo=recsys_repo,
        recsys_model=recsys_model,
        tmdb_gateway=tmdb_gateway,
        vectorizer=vectorizer,
        user_id=user_id,
        k=limit,
        offset=offset,
    )
    if not recs:
        await initialize_user(recsys_repo, recsys_model, user_id)
        recs = await get_movies(
            query=query,
            recsys_repo=recsys_repo,
            recsys_model=recsys_model,
            tmdb_gateway=tmdb_gateway,
            vectorizer=vectorizer,
            user_id=user_id,
            k=limit,
            offset=offset,
        )

    for i, r in enumerate(recs):
        r["user_id"] = user_id
        r["query"] = query
        r["num"] = i + offset + 1
    recs[-1]["offset"] = offset + len(recs)
    recs[-1]["limit"] = limit

    return templates.TemplateResponse("home.html", {"movies": recs, "request": request})


@app.get("/next/")
async def next(
    user_id: int,
    query: str | None,
    request: Request,
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    recsys_model: Annotated[OnlineRecSysRepo, Depends(get_recsys_model)],
    tmdb_gateway: Annotated[TMDBGateway, Depends(get_tmdb_gateway)],
    vectorizer: Annotated[Vectorizer, Depends(get_vectorizer)],
    limit: int = 15,
    offset: int = 0,
):
    recs = await get_movies(
        query=query,
        recsys_repo=recsys_repo,
        recsys_model=recsys_model,
        tmdb_gateway=tmdb_gateway,
        vectorizer=vectorizer,
        user_id=user_id,
        k=limit,
        offset=offset,
    )
    for i, r in enumerate(recs):
        r["user_id"] = user_id
        r["query"] = query
        r["num"] = i + offset + 1
    recs[-1]["offset"] = offset + len(recs)
    recs[-1]["limit"] = limit

    return templates.TemplateResponse("rows.html", {"movies": recs, "request": request})


@app.get("/recommendations")
async def _(
    user_id: int,
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    recsys_model: Annotated[OnlineRecSysModel, Depends(get_recsys_model)],
    tmdb_gateway: Annotated[TMDBGateway, Depends(get_tmdb_gateway)],
    vectorizer: Annotated[Vectorizer, Depends(get_vectorizer)],
    limit: int = 15,
    offset: int = 0,
):
    return await get_movies(
        query=None,
        recsys_repo=recsys_repo,
        recsys_model=recsys_model,
        tmdb_gateway=tmdb_gateway,
        vectorizer=vectorizer,
        user_id=user_id,
        k=limit,
        offset=offset,
    )


@app.post("/search/")
async def _(
    user_id: Annotated[int, Form()],
    request: Request,
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    recsys_model: Annotated[OnlineRecSysRepo, Depends(get_recsys_model)],
    tmdb_gateway: Annotated[TMDBGateway, Depends(get_tmdb_gateway)],
    vectorizer: Annotated[Vectorizer, Depends(get_vectorizer)],
    query: Annotated[str | None, Form()] = "",
    limit: Annotated[int, Form()] = 15,
    offset: Annotated[int, Form()] = 0,
):
    query = query if query else None
    recs = await get_movies(
        query=query,
        recsys_repo=recsys_repo,
        recsys_model=recsys_model,
        tmdb_gateway=tmdb_gateway,
        vectorizer=vectorizer,
        user_id=user_id,
        k=limit,
        offset=offset,
    )
    for i, r in enumerate(recs):
        r["user_id"] = user_id
        r["query"] = query
        r["num"] = i + offset + 1
    recs[-1]["offset"] = offset + len(recs)
    recs[-1]["limit"] = limit

    return templates.TemplateResponse("rows.html", {"movies": recs, "request": request})


# @app.post("/batch-feedback/")
# async def _(
#     user_id: int,
#     movie_ids: list[str],
#     labels: list[bool],
#     recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
#     recsys_model: Annotated[OnlineRecSysModel, Depends(get_recsys_model)],
# ):
#     await add_to_batch(
#         recsys_repo,
#         recsys_model,
#         MAX_N_BATCH,
#         user_id,
#         movie_ids,
#         labels,
#         normalized_lr=NORMALIZED_LR,
#     )


@app.post("/feedback/")
async def _(
    user_id: int,
    movie_id: str,
    label: bool,
    response: Response,
    recsys_repo: Annotated[OnlineRecSysRepo, Depends(get_recsys_repo)],
    recsys_model: Annotated[OnlineRecSysModel, Depends(get_recsys_model)],
):
    refreshed_preferences = await add_to_batch(
        recsys_repo,
        recsys_model,
        MAX_N_BATCH,
        user_id,
        [movie_id],
        [label],
    )

    # response.headers["HX-Refresh"] = "true"
    if refreshed_preferences:
        redirect_url = (
            app.url_path_for("next") + f"?user_id={user_id}&offset=0&limit=15&query="
        )
        # headers = {"HX-Refresh": "true"}
        return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    response.headers["HX-Reswap"] = "none"


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(Path(__file__).parent / "static" / "favicon.ico")


# if __name__ == "__main__":
# asyncio.run(amain_test())
