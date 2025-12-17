from rs1_tfidf import build_rs1_model, evaluate_rs1, RATING_THRESHOLD as RS1_THRESH
from rs2_fclip import build_rs2_model, evaluate_rs2, RATING_THRESHOLD as RS2_THRESH


def main():
    print("Building RS1 (TF-IDF) model...")
    rs1 = build_rs1_model()

    print("\nBuilding RS2 (FashionCLIP) model...")
    rs2 = build_rs2_model()

    print("EVALUATION: Recall@K and Serendipity@K")

    k_values = [5, 10]

    for k in k_values:
        print(f"K = {k}")

        rs1_metrics = evaluate_rs1(
            df=rs1["df"],
            item_tfidf=rs1["item_tfidf"],
            asin_to_index=rs1["asin_to_index"],
            index_to_asin=rs1["index_to_asin"],
            k=k,
            rating_threshold=RS1_THRESH,
            max_users=1000,
        )

        rs2_metrics = evaluate_rs2(
            df=rs2["df"],
            item_embeddings=rs2["item_embeddings"],
            asin_to_index=rs2["asin_to_index"],
            index_to_asin=rs2["index_to_asin"],
            k=k,
            rating_threshold=RS2_THRESH,
            max_users=1000,
        )

        print(f"\nRS1 (TF-IDF):")
        print(f"  Recall@{k}:       {rs1_metrics['recall_at_k']:.4f}")
        print(f"  Serendipity@{k}:  {rs1_metrics['serendipity_at_k']:.4f}")

        print(f"\nRS2 (FashionCLIP):")
        print(f"  Recall@{k}:       {rs2_metrics['recall_at_k']:.4f}")
        print(f"  Serendipity@{k}:  {rs2_metrics['serendipity_at_k']:.4f}")

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
