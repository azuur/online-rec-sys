import asyncio
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from online_learning_demo.config import PGConfig
from psycopg import AsyncConnection

load_dotenv()
pg_config = PGConfig()


async def amain():
    aconn = await AsyncConnection.connect(pg_config.uri())

    # await register_vector_async(aconn)
    create_users_table_sql = """
    CREATE TABLE IF NOT EXISTS users_current (
        id SERIAL PRIMARY KEY,
        vec VECTOR(1536),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """

    create_batches_table_sql = """
    CREATE TABLE IF NOT EXISTS batches_current (
        user_id INTEGER,
        movie_id CHAR(9),
        label BOOL,
        added_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (user_id, movie_id),
        FOREIGN KEY (user_id) REFERENCES users_current(id),
        FOREIGN KEY (movie_id) REFERENCES movies_all(id)
    )
    """

    create_users_history_table_sql = """
    CREATE TABLE IF NOT EXISTS users_history (
        id INTEGER,
        vec VECTOR(1536),
        next_batch_id INTEGER,
        moved_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (id, moved_at),
        FOREIGN KEY (id) REFERENCES users_current(id)
    )
    """

    create_batches_history_table_sql = """
    CREATE TABLE IF NOT EXISTS batches_history (
        batch_id SERIAL,
        user_id INTEGER,
        movie_id CHAR(9),
        label BOOL,
        added_at TIMESTAMP DEFAULT NOW(),
        FOREIGN KEY (user_id) REFERENCES users_current(id),
        FOREIGN KEY (movie_id) REFERENCES movies_all(id)
    )
    """

    model = OpenAIEmbeddings()
    initial_user_emb = (
        await model.aembed_documents(
            ["atmospheric, storytelling, drama, family drama, manchester"]
        )
    )[0]

    async with aconn.cursor() as cur:
        # s = "DROP TABLE users_history"
        # await cur.execute(s)
        await cur.execute(create_users_table_sql)
        await cur.execute(create_batches_table_sql)
        await cur.execute(create_users_history_table_sql)
        await cur.execute(create_batches_history_table_sql)

        user_id = 1
        await cur.execute(
            "INSERT INTO users_current (id, vec) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET vec = EXCLUDED.vec",
            (user_id, initial_user_emb),
        )
    await aconn.commit()


if __name__ == "__main__":
    asyncio.run(amain())
