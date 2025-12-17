import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "dataset_100k.csv"

USER_COL = "user_id"
ITEM_COL = "asin"
RATING_COL = "rating"
ITEM_TITLE_COL = "item_title"
DESCRIPTION_COL = "description"

RATING_THRESHOLD = 4.0


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df[[USER_COL, ITEM_COL, RATING_COL, ITEM_TITLE_COL, DESCRIPTION_COL]]

    # Basic cleaning
    df[ITEM_TITLE_COL] = df[ITEM_TITLE_COL].fillna("")
    df[DESCRIPTION_COL] = df[DESCRIPTION_COL].fillna("")

    # Combined text for each row
    df["combined_text"] = (
        df[ITEM_TITLE_COL].astype(str) + " " +
        df[DESCRIPTION_COL].astype(str)
    )

    return df


def build_item_tfidf(df, max_features=30000):
    item_texts = (
        df.groupby(ITEM_COL)["combined_text"]
          .apply(lambda texts: " ".join(texts))
          .reset_index()
          .rename(columns={"combined_text": "item_text"})
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
    )
    item_tfidf = vectorizer.fit_transform(item_texts["item_text"])

    index_to_asin = dict(enumerate(item_texts[ITEM_COL].values))
    asin_to_index = {asin: idx for idx, asin in index_to_asin.items()}

    return item_tfidf, item_texts, asin_to_index, index_to_asin


def build_user_profile(user_id, df, asin_to_index, item_tfidf,
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
    item_matrix = item_tfidf[liked_indices]

    # Weighted average by rating
    weights = liked[RATING_COL].values.astype(np.float32)
    weights = weights / weights.sum()

    user_vec_dense = weights @ item_matrix.toarray()
    user_vec = csr_matrix(user_vec_dense.reshape(1, -1))

    return user_vec, set(user_rows[ITEM_COL].values)


def recommend_for_user(user_id, df, asin_to_index, index_to_asin, item_tfidf,
                       rating_threshold=RATING_THRESHOLD, top_k=10):
    user_vec, seen_asins = build_user_profile(
        user_id, df, asin_to_index, item_tfidf, rating_threshold
    )

    if user_vec is None:
        return []

    sims = cosine_similarity(user_vec, item_tfidf).flatten()
    ranked_indices = np.argsort(-sims)

    recommendations = []
    for idx in ranked_indices:
        asin = index_to_asin[idx]
        if asin in seen_asins:
            continue
        score = sims[idx]
        recommendations.append((asin, float(score)))
        if len(recommendations) >= top_k:
            break

    return recommendations


# EVALUATION: RECALL@K + SERENDIPITY@K

def evaluate_rs1(df, item_tfidf, asin_to_index, index_to_asin,
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

    # Precompute item vectors row-wise
    item_matrix = item_tfidf

    for user in eval_users:
        user_rows = df[df[USER_COL] == user]
        liked_items = user_rows[user_rows[RATING_COL] >= rating_threshold][ITEM_COL].values

        if len(liked_items) < 2:
            continue

        test_item = liked_items[0]
        train_items = liked_items[1:]

        # Indices for train items
        train_indices = [asin_to_index[a] for a in train_items if a in asin_to_index]
        if not train_indices:
            continue

        # User vector for ranking: mean of train items
        user_vec = item_matrix[train_indices].mean(axis=0)
        user_vec = csr_matrix(user_vec)

        sims = cosine_similarity(user_vec, item_matrix).flatten()

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

            # Compute serendipity contribution from the test item
            if test_item in asin_to_index:
                test_idx = asin_to_index[test_item]
                test_vec = item_matrix[test_idx]

                train_vecs = item_matrix[train_indices]
                # similarities between test item and train items
                sims_test_train = cosine_similarity(test_vec, train_vecs).flatten()
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


def build_rs1_model():
    df = load_data()
    item_tfidf, item_texts, asin_to_index, index_to_asin = build_item_tfidf(df)

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
        "item_tfidf": item_tfidf,
        "asin_to_index": asin_to_index,
        "index_to_asin": index_to_asin,
        "item_meta": item_meta,
    }
