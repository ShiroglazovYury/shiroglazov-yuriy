import random
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


SEED = 993
MODEL_NAME = "intfloat/e5-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
MAX_LENGTH = 512


def set_seed(seed = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_html(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_test_dataframe(path):
    df = pd.read_csv(path)
    required_cols = [
        "id",
        "query",
        "product_title",
        "product_description",
        "product_bullet_point",
    ]
    text_cols = ["query", "product_title", "product_description", "product_bullet_point"]
    for col in text_cols:
        df[col] = df[col].fillna("")
        if col in ["product_title", "product_description", "product_bullet_point"]:
            df[col] = df[col].apply(strip_html)
    return df


def build_product_text(df):
    title = df["product_title"].astype(str)
    bullet = df["product_bullet_point"].astype(str)
    desc = df["product_description"].astype(str)
    return (title + ". " + bullet + ". " + desc).str.strip()


def average_pool(last_hidden_states, attention_mask):
    masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    backbone: AutoModel,
    prefix: str,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
):
    all_embeddings: list[Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {prefix.strip()}s"):
        batch_texts = [prefix + t for t in texts[start : start + batch_size]]
        batch = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        outputs = backbone(**batch)
        emb = average_pool(outputs.last_hidden_state, batch["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu())
    if not all_embeddings:
        hidden_size = backbone.config.hidden_size
        return np.zeros((0, hidden_size), dtype=np.float32)
    return torch.cat(all_embeddings, dim=0).numpy()


def generate_scores(test_df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    backbone.eval()

    queries = test_df["query"].astype(str).tolist()
    product_texts = build_product_text(test_df).astype(str).tolist()

    query_embs = encode_texts(queries, tokenizer, backbone, prefix="query: ")
    product_embs = encode_texts(product_texts, tokenizer, backbone, prefix="passage: ")

    scores = np.sum(query_embs * product_embs, axis=1)
    return scores.astype(float)


def save_submission(test_df, scores, path):
    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "prediction": scores,
        }
    )
    submission.to_csv(path, index=False)


def main():
    set_seed()
    test_path = "data/test.csv"
    test_df = prepare_test_dataframe(test_path)
    scores = generate_scores(test_df)
    save_submission(test_df, scores, "results/submission.csv")


if __name__ == "__main__":
    main()