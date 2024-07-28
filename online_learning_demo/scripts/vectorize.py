import asyncio
import inspect
import json
from logging import getLogger
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from langchain_core.embeddings import Embeddings
from tqdm.asyncio import tqdm_asyncio

logger = getLogger(__file__)

raw_data_path = (
    Path(__file__).parent.parent.parent / "data" / "raw" / "mpst_full_data.csv"
)

processed_data_path = (
    Path(__file__).parent.parent.parent
    / "data"
    / "processed"
    / "mpst_full_data_processed.parquet"
)

embedding_data_path = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "embeddings.jsonl"
)

load_dotenv()


def make_embedding_input(row: dict):
    return inspect.cleandoc(
        f"""
        *{row['title']}*
        Tags: {row['tags']}
        Synopsis: {row['plot_synopsis']}
        """
    )


async def embed_with_semaphore(
    model: Embeddings, semaphore: asyncio.Semaphore, docs: list[str], timeout=2
):
    async with semaphore:
        try:
            result = await model.aembed_documents(docs)
            await asyncio.sleep(timeout)
            return result
        except Exception as e:
            logger.warning("Embedding failed.", exc_info=e)
            return [None for _ in range(len(docs))]


async def amain():
    load_dotenv()

    data = pd.read_csv(raw_data_path)

    data["text_to_embed"] = data.apply(make_embedding_input, axis=1)

    data.to_parquet(processed_data_path)

    model = OpenAIEmbeddings()
    semaphore = asyncio.Semaphore(20)

    batch_size = 5
    tasks = [
        embed_with_semaphore(
            model,
            semaphore,
            data["text_to_embed"].iloc[pos : (pos + batch_size)].tolist(),
            timeout=5,
        )
        for pos in range(0, len(data), batch_size)
    ]

    embeddings = await tqdm_asyncio.gather(*tasks)
    embeddings = [e for batch in embeddings for e in batch]
    embeddings = [{k: v} for k, v in zip(data["imdb_id"].tolist(), embeddings)]

    with open(embedding_data_path, "w") as f:
        for entry in embeddings:
            json.dump(entry, f)
            f.write("\n")


if __name__ == "__main__":
    asyncio.run(amain())
