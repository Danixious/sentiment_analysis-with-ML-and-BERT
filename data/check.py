import pandas as pd
import os

txt_path = "testdata.txt"
csv_path = "labeled_data.csv"


if not os.path.exists(txt_path):
    print(f" File '{txt_path}' not found.")
else:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("'testdata.txt' is empty.")
    else:
        df = pd.DataFrame({
            "Text": lines,
            "Sentiment": "",
            "Review": ""
        })

        df.to_csv(csv_path, index=False)
        print(f" '{csv_path}' created with {len(df)} entries.")

