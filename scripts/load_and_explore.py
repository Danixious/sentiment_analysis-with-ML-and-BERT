import os
import pandas as pd

def load_reviews(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    reviews = [line.strip() for line in lines if line.strip()]
    print(f" Loaded {len(reviews)} non-empty reviews.")
    return reviews

def preview_data(reviews, n=10):
    print("\n Sample Reviews:\n")
    for i, review in enumerate(reviews[:n], start=1):
        print(f"{i}. {review}")

def analyze_lengths(reviews):
    lengths = [len(review.split()) for review in reviews]
    df = pd.DataFrame(lengths, columns=["word_count"])
    
    print("\n Review Length Stats:")
    print(df.describe())

def remove_duplicates(reviews):
    before = len(reviews)
    reviews = list(set(reviews))
    after = len(reviews)
    print(f"\n Removed {before - after} duplicate reviews.")
    return reviews

if __name__ == "__main__":
    path = "data/testdata.txt"

    print(" Loading Data...")
    reviews = load_reviews(path)

    reviews = remove_duplicates(reviews)

    preview_data(reviews)
    analyze_lengths(reviews)
