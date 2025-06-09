import pandas as pd


with open("testdata.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()


df = pd.DataFrame({
    "Text": [line.strip() for line in lines],      
    "Sentiment": "",                               
    "Review": ""                                    
})


df.to_csv("labeled_data.csv", index=False)

print("✅ labeled_data.csv created from testdata.txt with additional columns.")
