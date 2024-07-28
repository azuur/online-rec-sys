import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from online_learning_demo.config import PGConfig
from psycopg import AsyncConnection

load_dotenv()
pg_config = PGConfig()


processed_data_path = (
    Path(__file__).parent.parent.parent
    / "data"
    / "processed"
    / "mpst_full_data_processed.parquet"
)

embedding_data_path = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "embeddings.jsonl"
)


def load_data():
    df = pd.read_parquet(processed_data_path)

    with open(embedding_data_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            cid, emb = list(json.loads(line).items())[0]

            df_row = df[df["imdb_id"] == cid].iloc[0]

            yield {
                "id": cid,
                "title": df_row["title"],
                "synopsis": df_row["plot_synopsis"],
                "tags": df_row["tags"],
                "vec_input": df_row["text_to_embed"],
                "vec": emb,
            }


async def amain():
    aconn = await AsyncConnection.connect(pg_config.uri())

    # await register_vector_async(aconn)
    create_ext_sql = "CREATE EXTENSION IF NOT EXISTS vector;"
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS movies_all (
        id CHAR(9),
        title VARCHAR,
        synopsis VARCHAR,
        tags VARCHAR,
        vec_input VARCHAR,
        vec VECTOR(1536),
        PRIMARY KEY (id)
    );
    """
    create_posters_table_sql = """
    CREATE TABLE IF NOT EXISTS movies_posters (
        movie_id CHAR(9),
        img_bytes BYTEA,
        FOREIGN KEY (movie_id) REFERENCES movies_all(id)
    );
    """
    create_index_sql = "CREATE INDEX ON movies_all USING hnsw (vec vector_cosine_ops);"
    count_table_sql = "select count(*) from movies_all;"

    async with aconn.cursor() as cur:
        # await cur.execute(create_ext_sql)
        # await cur.execute(create_table_sql)
        # await cur.execute(create_posters_table_sql)
        # await cur.execute(create_index_sql)
        n = await cur.execute(count_table_sql)
        n = await n.fetchall()
        n = n[0][0]
        print(f"n={n}")

    if n > 0:
        raise Exception("wait...")

    async with aconn.cursor() as cur:
        for row in tqdm(load_data()):
            await cur.execute(
                "INSERT INTO movies_all "
                "(id, title, synopsis, tags, vec_input, vec) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    row["id"],
                    row["title"],
                    row["synopsis"],
                    row["tags"],
                    row["vec_input"],
                    row["vec"],
                ),
            )
    await aconn.commit()


if __name__ == "__main__":
    asyncio.run(amain())
