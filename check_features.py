import pickle
from ml.feature_extraction import extract_features

model = pickle.load(open("ml/model/phishing_model.pkl", "rb"))

names = ["url_length","hostname_length","path_length","dots","hyphens","underscores",
         "slashes","question_marks","equals","at_symbols","ampersands","percent_symbols",
         "uses_https","uses_ip","subdomains","suspicious_words","has_at","double_slash",
         "shortener","digits","digit_ratio","query_params","entropy"]

test_urls = [
    "https://www.google.com",
    "https://www.southbankmosaics.com",
    "https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset",
    "http://www.teramill.com",
]

for url in test_urls:
    feats = extract_features(url)
    proba = model.predict_proba([feats])[0]
    print("URL:", url)
    for n, v in zip(names, feats):
        print(f"  {n}: {v}")
    print(f"  --> P(phishing)={proba[0]:.4f}  P(legit)={proba[1]:.4f}")
    print()
