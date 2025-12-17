import argparse

from rs1_tfidf import build_rs1_model, recommend_for_user as recommend_rs1, RATING_THRESHOLD as RS1_THRESH
from rs2_fclip import build_rs2_model, recommend_for_user as recommend_rs2, RATING_THRESHOLD as RS2_THRESH


def print_recommendations(user_id, model_name, recs, item_meta, rating_threshold):
    print(f"RECOMMENDATIONS")
    print(f"Model: {model_name}")
    print(f"User:  {user_id}")

    print("\nUser's liked items (rating >= {:.1f}):".format(rating_threshold))
    df = item_meta._df if hasattr(item_meta, "_df") else None

    print("\nTop recommendations:")
    print(f"{'Rank':<5} {'ASIN':<15} {'Title':<60} {'Score':<8}")
    for rank, (asin, score) in enumerate(recs, 1):
        if asin in item_meta.index:
            title = str(item_meta.loc[asin, "item_title"])[:57]
        else:
            title = "Unknown title"
        print(f"{rank:<5} {asin:<15} {title:<60} {score:<8.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI for RS1 (TF-IDF) and RS2 (FashionCLIP) recommender systems."
    )
    parser.add_argument("--model", choices=["rs1", "rs2"], required=True,
                        help="Which recommender to use: rs1 (TF-IDF) or rs2 (FashionCLIP).")
    parser.add_argument("--user", required=True,
                        help="User ID to get recommendations.")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of recommendations to return.")

    args = parser.parse_args()

    if args.model == "rs1":
        print("Loading RS1 model (TF-IDF)...")
        rs1 = build_rs1_model()
        df = rs1["df"]
        if args.user not in df["user_id"].astype(str).values:
            print(f"User {args.user} not found in dataset.")
            return
        recs = recommend_rs1(
            user_id=args.user,
            df=df,
            asin_to_index=rs1["asin_to_index"],
            index_to_asin=rs1["index_to_asin"],
            item_tfidf=rs1["item_tfidf"],
            rating_threshold=RS1_THRESH,
            top_k=args.top_k,
        )
        print_recommendations(args.user, "RS1 (TF-IDF)", recs, rs1["item_meta"], RS1_THRESH)

    else:
        print("Loading RS2 model (FashionCLIP)...")
        rs2 = build_rs2_model()
        df = rs2["df"]
        if args.user not in df["user_id"].astype(str).values:
            print(f"User {args.user} not found in dataset.")
            return
        recs = recommend_rs2(
            user_id=args.user,
            df=df,
            asin_to_index=rs2["asin_to_index"],
            index_to_asin=rs2["index_to_asin"],
            item_embeddings=rs2["item_embeddings"],
            rating_threshold=RS2_THRESH,
            top_k=args.top_k,
        )
        print_recommendations(args.user, "RS2 (FashionCLIP)", recs, rs2["item_meta"], RS2_THRESH)


if __name__ == "__main__":
    main()
