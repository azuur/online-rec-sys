import asyncio
from io import BytesIO
from dotenv import load_dotenv
import joblib
from langchain_openai import OpenAIEmbeddings
import numpy as np
from sklearn.linear_model import SGDClassifier
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

    create_users_table_v1_sql = """
    CREATE TABLE IF NOT EXISTS users_current_v1 (
        id SERIAL PRIMARY KEY,
        sklearn_model BYTEA,
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

    create_users_history_table_v1_sql = """
    CREATE TABLE IF NOT EXISTS users_history_v1 (
        id INTEGER,
        sklearn_model BYTEA,
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
        await cur.execute(create_users_table_v1_sql)
        await cur.execute(create_users_history_table_v1_sql)

        user_id = 1
        # await cur.execute(
        #     "INSERT INTO users_current (id, vec) VALUES (%s, %s) "
        #     "ON CONFLICT (id) DO UPDATE SET vec = EXCLUDED.vec",
        #     (user_id, initial_user_emb),
        # )

        X = np.random.randn(10, 1536)
        y = np.random.randint(2, size=10)

        model = SGDClassifier(loss="log_loss")
        model.partial_fit(X, y, classes=[0, 1])

        bytes_io = BytesIO()
        joblib.dump(model, bytes_io)
        bytes_io.seek(0)

        model_bytes = bytes_io.read()

        await cur.execute(
            "INSERT INTO users_current_v1 (id, sklearn_model) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET sklearn_model = EXCLUDED.sklearn_model",
            (user_id, model_bytes),
        )
    await aconn.commit()


if __name__ == "__main__":
    asyncio.run(amain())
