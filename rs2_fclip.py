# rs2_fclip.py

import os
import numpy as np
import pandas as pd

from fashion_clip.fashion_clip import FashionCLIP

DATA_PATH = "dataset_100k.csv"

USER_COL = "user_id"
ITEM_COL = "asin"
PARENT_ASIN_COL = "parent_asin"
RATING_COL = "rating"
ITEM_TITLE_COL = "item_title"
DESCRIPTION_COL = "description"
FEATURES_COL = "features"
STORE_COL = "store"

RATING_THRESHOLD = 4.0

EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "fashionclip_text_only.npy")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[[USER_COL, ITEM_COL, PARENT_ASIN_COL, RATING_COL,
             ITEM_TITLE_COL, DESCRIPTION_COL, FEATURES_COL, STORE_COL]]

    df[ITEM_TITLE_COL] = df[ITEM_TITLE_COL].fillna("")
    df[DESCRIPTION_COL] = df[DESCRIPTION_COL].fillna("")
    df[FEATURES_COL] = df[FEATURES_COL].astype(str).fillna("")
    df[STORE_COL] = df[STORE_COL].fillna("")

    df["combined_text"] = df.apply(prepare_item_text, axis=1)

    item_texts = (
        df.groupby(ITEM_COL)
          .agg({
              "combined_text": "first",
              PARENT_ASIN_COL: "first",
              ITEM_TITLE_COL: "first",
          })
          .reset_index()
    )

    return df, item_texts


def prepare_item_text(row):
    parts = []

    if row[ITEM_TITLE_COL]:
        parts.append(str(row[ITEM_TITLE_COL]))

    if row[DESCRIPTION_COL]:
        parts.append(str(row[DESCRIPTION_COL]))

    if row[FEATURES_COL] and row[FEATURES_COL] != "nan":
        features_text = str(row[FEATURES_COL])
        if len(features_text) < 500:
            parts.append(features_text)

    if row[STORE_COL]:
        parts.append(f"by {row[STORE_COL]}")

    return " ".join(parts)


# FASHIONCLIP EMBEDDINGS

def extract_fashionclip_text_embeddings(item_texts, batch_size=32):
    print("Loading FashionCLIP model...")
    fclip = FashionCLIP("fashion-clip")

    texts = item_texts["combined_text"].tolist()
    n_items = len(texts)

    print(f"Extracting text embeddings for {n_items} items...")
    embeddings = []

    for i in range(0, n_items, batch_size):
        if i % (batch_size * 10) == 0:
            print(f"  Processed {i}/{n_items}...")
        batch_texts = texts[i: i + batch_size]
        try:
            batch_emb = fclip.encode_text(batch_texts, batch_size=batch_size)
            embeddings.extend(batch_emb)
        except Exception as e:
            print(f"  Failed batch at {i}: {e}")
            embeddings.extend([np.zeros(512)] * len(batch_texts))

    embeddings = np.vstack(embeddings)
    print("Embedding extraction complete.", embeddings.shape)
    return embeddings


def load_or_build_embeddings(item_texts):
    if os.path.exists(EMBEDDINGS_PATH):
        print(f"Loading cached FashionCLIP embeddings from {EMBEDDINGS_PATH}...")
        embeddings = np.load(EMBEDDINGS_PATH)
        print("Loaded embeddings with shape:", embeddings.shape)
        return embeddings

    print("No cached embeddings found. Extracting FashionCLIP embeddings")
    embeddings = extract_fashionclip_text_embeddings(item_texts, batch_size=32)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    return embeddings


def build_user_profile(user_id, df, asin_to_index, item_embeddings,
                       rating_threshold=RATING_THRESHOLD):
    user_rows = df[df[USER_COL] == user_id]
    if user_rows.empty:
        return None, None

    liked = user_rows[user_rows[RATING_COL] >= rating_threshold]
    if liked.empty:
        return None, None

    liked_asins = [a for a in liked[ITEM_COL].values if a in asin_to_index]
    if not liked_asins:
        return None, None

    liked_indices = [asin_to_index[a] for a in liked_asins]
    liked_embs = item_embeddings[liked_indices]

    weights = liked[RATING_COL].values.astype(np.float32)
    weights = weights / weights.sum()

    user_profile = weights @ liked_embs

    # L2 norm
    norm = np.linalg.norm(user_profile)
    if norm > 0:
        user_profile = user_profile / norm

    return user_profile, set(user_rows[ITEM_COL].values)


def recommend_for_user(user_id, df, asin_to_index, index_to_asin, item_embeddings,
                       rating_threshold=RATING_THRESHOLD, top_k=10):
    user_profile, seen_asins = build_user_profile(
        user_id, df, asin_to_index, item_embeddings, rating_threshold
    )

    if user_profile is None:
        return []

    item_norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    item_embeddings_norm = item_embeddings / (item_norms + 1e-8)

    sims = item_embeddings_norm @ user_profile
    ranked_indices = np.argsort(-sims)

    recommendations = []
    for idx in ranked_indices:
        asin = index_to_asin[idx]
        if asin in seen_asins:
            continue
        score = float(sims[idx])
        recommendations.append((asin, score))
        if len(recommendations) >= top_k:
            break

    return recommendations


# EVALUATION: RECALL@K + SERENDIPITY@K

def evaluate_rs2(df, item_embeddings, asin_to_index, index_to_asin,
                 k=5, rating_threshold=RATING_THRESHOLD,
                 max_users=1000, random_state=42):
    rng = np.random.default_rng(random_state)

    pos_df = df[df[RATING_COL] >= rating_threshold]
    user_pos_counts = pos_df[USER_COL].value_counts()
    eligible_users = user_pos_counts[user_pos_counts >= 3].index.to_numpy()

    if len(eligible_users) == 0:
        return {"recall_at_k": 0.0, "serendipity_at_k": 0.0}

    if len(eligible_users) > max_users:
        eval_users = rng.choice(eligible_users, size=max_users, replace=False)
    else:
        eval_users = eligible_users

    hits = 0
    total = 0
    serendipity_sum = 0.0

    # Normalize item embeddings once
    item_norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    items_norm = item_embeddings / (item_norms + 1e-8)

    for user in eval_users:
        user_rows = df[df[USER_COL] == user]
        liked_items = user_rows[user_rows[RATING_COL] >= rating_threshold][ITEM_COL].values

        if len(liked_items) < 2:
            continue

        test_item = liked_items[0]
        train_items = liked_items[1:]

        train_indices = [asin_to_index[a] for a in train_items if a in asin_to_index]
        if not train_indices:
            continue

        # User vector = mean of train embeddings
        user_vec = items_norm[train_indices].mean(axis=0)
        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)

        sims = items_norm @ user_vec

        if k < len(sims):
            candidate_idx = np.argpartition(-sims, k)[:k * 2]
            candidate_scores = sims[candidate_idx]
            sorted_idx = candidate_idx[np.argsort(-candidate_scores)]
            ranked = sorted_idx
        else:
            ranked = np.argsort(-sims)

        recs = []
        train_set = set(train_items)
        for idx in ranked:
            asin = index_to_asin[idx]
            if asin in train_set:
                continue
            recs.append(asin)
            if len(recs) >= k:
                break

        if len(recs) == 0:
            continue

        hit = test_item in recs
        if hit:
            hits += 1

            if test_item in asin_to_index:
                test_idx = asin_to_index[test_item]
                test_vec = items_norm[test_idx]

                train_vecs = items_norm[train_indices]
                sims_test_train = train_vecs @ test_vec
                max_sim_seen = float(sims_test_train.max()) if len(sims_test_train) > 0 else 0.0
                unexpectedness = 1.0 - max_sim_seen
            else:
                unexpectedness = 0.0
        else:
            unexpectedness = 0.0

        serendipity_sum += unexpectedness
        total += 1

    if total == 0:
        return {"recall_at_k": 0.0, "serendipity_at_k": 0.0}

    recall_at_k = hits / total
    serendipity_at_k = serendipity_sum / total

    return {"recall_at_k": recall_at_k, "serendipity_at_k": serendipity_at_k}

def build_rs2_model():
    df, item_texts = load_data()
    item_embeddings = load_or_build_embeddings(item_texts)

    index_to_asin = dict(enumerate(item_texts[ITEM_COL].values))
    asin_to_index = {asin: idx for idx, asin in index_to_asin.items()}

    item_meta = (
        df.groupby(ITEM_COL)
          .agg(
              item_title=(ITEM_TITLE_COL, "first"),
              mean_rating=(RATING_COL, "mean"),
              rating_count=(RATING_COL, "count"),
          )
    )

    return {
        "df": df,
        "item_embeddings": item_embeddings,
        "asin_to_index": asin_to_index,
        "index_to_asin": index_to_asin,
        "item_meta": item_meta,
    }
