import ast
import asyncio
from io import BytesIO
from itertools import repeat
from logging import getLogger
import logging
from typing import Awaitable, Callable, Iterable
from dotenv import load_dotenv
import joblib
import numpy as np
from psycopg import AsyncConnection
import json

from sklearn.linear_model import SGDClassifier
from online_learning_demo.config import PGConfig, TMDBConfig
from httpx import AsyncClient

logger = getLogger(__file__)
logger.setLevel(logging.DEBUG)


class TMDBGateway:
    def __init__(self, httpx_client: AsyncClient, tmdb_config: TMDBConfig):
        self._httpx_client = httpx_client
        self._tmdb_config = tmdb_config

    async def aget_movie_poster(self, movie_id: str):
        try:
            find_url = f"{self._tmdb_config.find_url}{movie_id}?external_source=imdb_id"

            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self._tmdb_config.api_read_access_token}",
            }
            response = await self._httpx_client.get(find_url, headers=headers)
            response = response.json()
            poster_path = response["movie_results"][0]["poster_path"]
            poster_url = f"{self._tmdb_config.poster_url}{poster_path}"
            poster_bytes = await self._httpx_client.get(poster_url, headers=headers)

            return poster_bytes.content
        except Exception as e:
            logger.debug("Poster retrieval failed", exc_info=e)
            return None

    async def aget_movie_posters(self, movie_ids: list[str]):
        semaphore = asyncio.Semaphore(10)
        sleep_secs = 0.2
        operation = AsyncOperationWithSemaphore(
            semaphore,
            operation=self.aget_movie_poster,
            sleep_secs=sleep_secs,
        )
        new_imgs_bytes = await operation(*movie_ids)

        return new_imgs_bytes


def get_tmdb_gateway():
    load_dotenv()
    tmdb_config = TMDBConfig()
    httpx_client = AsyncClient()
    return TMDBGateway(httpx_client, tmdb_config)


async def aget_db_conn():
    load_dotenv()
    pg_config = PGConfig()
    return await AsyncConnection.connect(pg_config.uri())


class OnlineRecSysRepo:
    def __init__(self, aconn: AsyncConnection):
        self.aconn = aconn

    # async def aget_rec_by_movie_id(self, movie_id: str, k: int = 5, offset: int = 0):
    #     sql = """
    #     WITH thisvec AS (
    #         SELECT vec
    #         FROM movies_all
    #         WHERE id = %s
    #         LIMIT 1
    #     )
    #     SELECT
    #         movies_all.id,
    #         movies_all.title,
    #         movies_all.synopsis,
    #         movies_all.tags,
    #         (2 - (movies_all.vec <=> thisvec.vec)) / 2 AS score
    #     FROM movies_all, thisvec
    #     WHERE movies_all.id <> %s
    #     ORDER BY score DESC
    #     OFFSET %s
    #     LIMIT %s;
    #     """
    #     async with self.aconn.cursor() as cur:
    #         res = await cur.execute(sql, (movie_id, movie_id, offset, k))
    #         res = await res.fetchall()
    #         return res

    async def aget_model_bytes(self, user_id: int):
        sql = """
        SELECT id, sklearn_model 
        FROM users_current_v1
        WHERE id = %s
        LIMIT 1
        """

        async with self.aconn.cursor() as cur:
            result = await cur.execute(sql, (user_id,))
            result = await result.fetchall()

        if result:
            return next(iter(result))[1]
        return None

    async def aget_rec_by_user_id(self, user_id: str, k: int = 5, offset: int = 0):
        sql = """
        WITH thisvec AS (
            SELECT id, vec 
            FROM users_current 
            WHERE id = %(user_id)s
            LIMIT 1
        ),
        already_watched AS (
            SELECT movie_id
            FROM batches_current
            WHERE user_id = %(user_id)s
            UNION
            SELECT movie_id
            FROM batches_history
            WHERE user_id = %(user_id)s
        )
        SELECT
            movies_all.id,
            movies_all.title,
            movies_all.synopsis,
            movies_all.tags, 
            (2 - (movies_all.vec <=> thisvec.vec)) / 2 AS score,
            movies_all.vec
        FROM movies_all, thisvec
        WHERE movies_all.id NOT IN (SELECT movie_id FROM already_watched)
        ORDER BY score DESC
        OFFSET %(offset)s
        LIMIT %(limit)s;
        """
        async with self.aconn.cursor() as cur:
            res = await cur.execute(
                sql, {"user_id": user_id, "limit": k, "offset": offset}
            )
            res = await res.fetchall()
            return res

    async def aget_rec_random_by_user_id(
        self, user_id: str, k: int = 5, seed: int = None
    ):
        sql = f"""
        WITH thisvec AS (
            SELECT vec 
            FROM users_current 
            WHERE id = %(user_id)s
            LIMIT 1
        ),
        already_watched AS (
            SELECT movie_id
            FROM batches_current
            WHERE user_id = %(user_id)s
            UNION
            SELECT movie_id
            FROM batches_history
            WHERE user_id = %(user_id)s
        )
        SELECT
            movies_all.id,
            movies_all.title,
            movies_all.synopsis,
            movies_all.tags, 
            (2 - (movies_all.vec <=> thisvec.vec)) / 2 AS score,
            movies_all.vec
        FROM movies_all TABLESAMPLE BERNOULLI(1){f' REPEATABLE({seed})' if seed else ''}, thisvec
        WHERE movies_all.id NOT IN (SELECT movie_id FROM already_watched)
        LIMIT %(limit)s;
        """
        async with self.aconn.cursor() as cur:
            res = await cur.execute(sql, {"user_id": user_id, "limit": k})
            res = await res.fetchall()
            return res

    async def aget_rec_from_vector(self, vec: list[float], k: int = 5, offset: int = 0):
        sql = """
        SELECT
            movies_all.id,
            movies_all.title,
            movies_all.synopsis,
            movies_all.tags, 
            (2 - (movies_all.vec <=> %(vec)s)) / 2 AS score,
            movies_all.vec
        FROM movies_all
        ORDER BY score DESC
        OFFSET %(offset)s
        LIMIT %(limit)s;
        """
        async with self.aconn.cursor() as cur:
            res = await cur.execute(
                sql, {"vec": str(vec), "limit": k, "offset": offset}
            )
            res = await res.fetchall()
            return res

    async def aadd_to_current_batch(self, user_id: int, movie_ids: str, labels: bool):
        sql = """
        INSERT INTO batches_current (user_id, movie_id, label)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, movie_id) DO NOTHING
        """
        async with self.aconn.cursor() as cur:
            await cur.executemany(sql, zip(repeat(user_id), movie_ids, labels))
        await self.aconn.commit()
        return None

    async def aget_current_batch_size(self, user_id: int):
        sql = """
        SELECT COUNT(*) FROM batches_current WHERE user_id = %s
        """
        async with self.aconn.cursor() as cur:
            res = await cur.execute(sql, (user_id,))
            res = await res.fetchall()
            if not res:
                return None
            return res[0][0]

    async def aget_current_batch(self, user_id: int):
        sql = """
        WITH batch AS
        (SELECT user_id, movie_id, label FROM batches_current WHERE user_id = %s)
        SELECT batch.user_id, batch.movie_id, batch.label, movies_all.vec
        FROM batch
        JOIN movies_all
        ON batch.movie_id = movies_all.id
        """
        async with self.aconn.cursor() as cur:
            res = await cur.execute(sql, (user_id,))
            res = await res.fetchall()
        return [(*r[:-1], ast.literal_eval(r[-1])) for r in res]

    async def aget_current_user(self, user_id: int):
        sql = """
        SELECT id, vec
        FROM users_current 
        WHERE id = %s
        """
        async with self.aconn.cursor() as cur:
            res = await cur.execute(sql, (user_id,))
            res = await res.fetchall()
        return [(*r[:-1], ast.literal_eval(r[-1])) for r in res]

    async def amove_current_batch_to_historical(self, user_id: int):
        insert_sql = """
        WITH nextbatchid AS (SELECT nextval('batches_history_batch_id_seq') as id)      
        INSERT INTO batches_history (batch_id, user_id, movie_id, label)
        SELECT nextbatchid.id, user_id, movie_id, label
        FROM batches_current, nextbatchid
        WHERE user_id = %s
        RETURNING batch_id;
        """
        delete_sql = """
        DELETE FROM batches_current
        WHERE user_id = %s;
        """
        async with self.aconn.cursor() as cur:
            result = await cur.execute(insert_sql, (user_id,))
            result = await result.fetchone()
            await cur.execute(delete_sql, (user_id,))
        await self.aconn.commit()
        return result[0]

    async def aupdate_user(
        self, user_id: int, batch_id: int | None, model_bytes: bytes
    ):
        insert_sql = """
        INSERT INTO users_history_v1 (id, sklearn_model, next_batch_id)
        SELECT id, sklearn_model, %(batch_id)s
        FROM users_current_v1
        WHERE id = %(user_id)s;
        """
        update_sql = """
        UPDATE users_current_v1
        SET sklearn_model = %(model_bytes)s :: BYTEA, updated_at = NOW()
        WHERE id = %(user_id)s;
        """
        async with self.aconn.cursor() as cur:
            await cur.execute(insert_sql, {"user_id": user_id, "batch_id": batch_id})
            await cur.execute(
                update_sql,
                {"user_id": user_id, "model_bytes": model_bytes},
            )
        await self.aconn.commit()

    async def aget_movie_posters(self, movie_ids: list[str]) -> list[bytes | None]:
        select_sql = """
        SELECT movie_id, img_bytes
        FROM movies_posters
        WHERE movie_id = ANY(%(movie_ids)s)
        """
        async with self.aconn.cursor() as cur:
            result = await cur.execute(select_sql, {"movie_ids": movie_ids})
            result = await result.fetchall()

        result = {k: v for (k, v) in result}
        return [result.get(k) for k in movie_ids]

        # if set(result.keys()) == set(movie_ids):
        #     return [result[k] for k in movie_ids]

        # missing_img_ids = [k for k in movie_ids if k not in result]
        # new_imgs_bytes = await self._aget_movie_posters_external(missing_img_ids)
        # new_imgs = dict(zip(missing_img_ids, new_imgs_bytes))
        # await self._ainsert_movie_posters(((k, v) for k, v in new_imgs.items()))

        # result.update(new_imgs)

    async def ainsert_movie_posters(
        self, movie_ids_and_images: Iterable[tuple[str, bytes | None]]
    ):
        insert_sql = """
        INSERT INTO movies_posters (movie_id, img_bytes)
        VALUES (%s, %s)
        """
        async with self.aconn.cursor() as cur:
            await cur.executemany(
                insert_sql, (t for t in movie_ids_and_images if t[1] is not None)
            )
        await self.aconn.commit()

    async def ainitialize_user(self, user_id: int, model_bytes: bytes):
        async with self.aconn.cursor() as cur:
            await cur.execute(
                "INSERT INTO users_current_v1 (id, sklearn_model) VALUES (%s, %s) "
                "ON CONFLICT (id) DO UPDATE SET sklearn_model = EXCLUDED.sklearn_model",
                (user_id, model_bytes),
            )
        await self.aconn.commit()

    async def aget_user_feedback_history(self, user_id: int):
        sql = """
        WITH hist as (
            SELECT label, movie_id FROM batches_history WHERE user_id = %s
        )
        SELECT hist.label, movies_all.vec
        FROM hist
        JOIN movies_all
        ON hist.movie_id = movies_all.id
        """
        async with self.aconn.cursor() as cur:
            res = await cur.execute(sql, (user_id,))
            res = await res.fetchall()
            if not res:
                return None
            return list(zip(*[(mid, ast.literal_eval(vec)) for mid, vec in res]))


class OnlineRecSysModel:
    def __init__(self, recsys_repo: OnlineRecSysRepo) -> None:
        self.recsys_repo = recsys_repo

    async def aload_model(self, user_id: int):
        model_bytes = await self.recsys_repo.aget_model_bytes(user_id)
        if not model_bytes:
            return None
        model_buffer = BytesIO(model_bytes)
        model: SGDClassifier = joblib.load(model_buffer)
        return model

    def calculate_updated(
        self,
        model: SGDClassifier,
        batch: tuple[list[list[float]], list[bool]],
    ):
        X = np.array(batch[0])
        y = np.array(batch[1]).astype(int)
        model.partial_fit(X, y)
        return model

    def score_content(self, model: SGDClassifier, batch: list[list[float]]):
        X = np.array(batch)
        return model.predict_proba(X)[:, 1].tolist()

    def dump_model(self, model: SGDClassifier):
        bytes_io = BytesIO()
        joblib.dump(model, bytes_io)
        bytes_io.seek(0)
        return bytes_io.read()

    def initialize_new_model(
        self, batch: tuple[list[list[float]], list[bool]] | None = None
    ):
        breakpoint()
        if batch is not None:
            X = (1 / np.sqrt(1536)) * np.random.randn(100, 1536)
            y = [0, 1] + np.random.randint(2, size=98).tolist()
        else:
            X = np.array(batch[0])
            y = np.array(batch[1]).astype(int)

        # maybe fiddle with params
        model = SGDClassifier(
            loss="log_loss", alpha=1, learning_rate="constant", eta0=0.1
        )
        model.fit(X, y)
        return model


class AsyncOperationWithSemaphore[K, V]:
    def __init__(
        self,
        semaphore: asyncio.Semaphore,
        operation: Callable[[K], Awaitable[V]],
        sleep_secs: float,
    ):
        self._semaphore = semaphore
        self.operation = operation
        self.sleep_secs = sleep_secs

    async def _aoperation(self, inpt: K):
        async with self._semaphore:
            res = await self.operation(inpt)
            await asyncio.sleep(self.sleep_secs)
            return res

    async def __call__(self, *inpts: K):
        return await asyncio.gather(*(self._aoperation(i) for i in inpts))


async def amain():
    aconn = await aget_db_conn()
    async with aconn.cursor() as cur:
        res = await cur.execute("SELECT * FROM movies_all LIMIT 1")
        res = await res.fetchall()
        print(str(res)[:100])

    recsys_repo = OnlineRecSysRepo(aconn=aconn)

    result = await recsys_repo.aget_rec_by_movie_id(res[0][0])
    print(json.dumps([str(x)[:100] for x in result], indent=2))

    result = await recsys_repo.aget_rec_from_vector(res[0][-1])
    print(json.dumps([str(x)[:100] for x in result], indent=2))

    result = await recsys_repo.aget_rec_by_user_id(1)
    print(json.dumps([str(x)[:100] for x in result], indent=2))

    result = await recsys_repo.aget_rec_random_by_user_id(1)
    print(json.dumps([str(x)[:100] for x in result], indent=2))

    # await recsys_repo.aadd_to_current_batch(1, "tt0120382", True)  # The Truman Show
    # await recsys_repo.aadd_to_current_batch(1, "tt0054518", False)  # Avengers
    cur_batch = await recsys_repo.aget_current_batch(1)
    print(json.dumps([str(x)[:100] for x in cur_batch], indent=2))

    cur_user = await recsys_repo.aget_current_user(1)
    print(json.dumps([str(x)[:100] for x in cur_user], indent=2))

    cur_batch_X = [row[-1] for row in cur_batch]
    cur_batch_Y = [row[2] for row in cur_batch]
    cur_user_vec = cur_user[0][-1]

    recsys_model = OnlineRecSysModel()
    updated_user_emb = recsys_model.calculate_updated(
        cur_user_vec, (cur_batch_X, cur_batch_Y)
    )

    print(str(updated_user_emb)[:100])

    new_batch_id = await recsys_repo.amove_current_batch_to_historical(1)
    print(new_batch_id, "\n\n")
    result = await recsys_repo.aget_current_batch_size(1)
    print(result)

    await recsys_repo.aupdate_user(1, new_batch_id, updated_user_emb)

    result = await recsys_repo.aget_current_user(1)
    print(json.dumps([str(x)[:100] for x in result], indent=2))

    await recsys_repo.aadd_to_current_batch(
        1, ["tt0120382", "tt0054518"], [True, False]
    )  # The Truman Show, Avengers

    result = await recsys_repo.aget_current_batch(1)
    print(json.dumps([str(x)[:100] for x in result], indent=2))

    result = await recsys_repo.aget_current_batch_size(1)
    print(result)

    result = await recsys_repo.aget_rec_by_movie_id("tt0120382")
    print(json.dumps([str(x)[:100] for x in result], indent=2))

    # model = OpenAIEmbeddings()
    # initial_user_emb = (
    #     await model.aembed_documents(["atmospheric, storytelling, drama"])
    # )[0]

    # await recsys_repo.aupdate_user(1, new_batch_id, initial_user_emb)

    result = await recsys_repo.aget_current_user(1)
    print(json.dumps([str(x)[:100] for x in result], indent=2))


async def amain():
    tmdb_gateway = get_tmdb_gateway()
    aconn = await aget_db_conn()
    async with aconn.cursor() as cur:
        res = await cur.execute("SELECT * FROM movies_all LIMIT 1")
        res = await res.fetchall()
        print(str(res)[:100])

    recsys_repo = OnlineRecSysRepo(aconn=aconn)

    movie_ids = [
        "tt0055058",
        "tt0082951",
        "tt0094812",
        "tt0039302",
    ]

    imgs_bytes = await recsys_repo.aget_movie_posters(movie_ids)
    print(imgs_bytes)

    imgs_bytes = await tmdb_gateway.aget_movie_posters(movie_ids)
    print(imgs_bytes)

    await recsys_repo.ainsert_movie_posters(zip(movie_ids, imgs_bytes))
    imgs_bytes = await recsys_repo.aget_movie_posters(movie_ids)
    print(imgs_bytes)

    # with open("test.jpg", "wb") as f:
    #     f.write(imgs_bytes[0])

    # if set(result.keys()) == set(movie_ids):
    #     return [result[k] for k in movie_ids]

    # missing_img_ids = [k for k in movie_ids if k not in result]
    # new_imgs_bytes = await self._aget_movie_posters_external(missing_img_ids)
    # new_imgs = dict(zip(missing_img_ids, new_imgs_bytes))
    # await self._ainsert_movie_posters(((k, v) for k, v in new_imgs.items()))

    # result.update(new_imgs)


async def amain():
    aconn = await aget_db_conn()
    recsys_repo = OnlineRecSysRepo(aconn=aconn)
    recsys_model = OnlineRecSysModel(recsys_repo=recsys_repo)

    model = await recsys_model.aload_model(1)
    breakpoint()


if __name__ == "__main__":
    asyncio.run(amain())
