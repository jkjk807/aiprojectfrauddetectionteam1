import os
import base64
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import json
import math
import numpy as np
import pandas as pd
from dateutil import parser as dateparser
from sklearn.metrics.pairwise import cosine_distances


endpoint = os.getenv("ENDPOINT_URL", "https://cimb-hackathon-team1-resource.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
      
# Initialize Azure OpenAI client with Entra ID authentication
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2025-01-01-preview",
)


# =========================
# 1) POC CONFIG
# =========================
INPUT_CSV = "transactions.csv"
ACCOUNT_REPORT = "account_fraud_report.csv"
TX_REPORT = "transactions_flagged.csv"

# Tunables (start here, tweak after you see results)
COSINE_THRESHOLD = 0.30        # per-transaction anomaly distance threshold
FRAUD_ACCOUNT_RATIO = 0.20     # if >20% txs anomalous, mark account suspicious
MAX_BOOST_FACTOR = 1.8         # if a single tx exceeds COSINE_THRESHOLD*factor, mark account suspicious too
BATCH_SIZE = 64                # embedding batch size

# =========================
# 2) HELPERS
# =========================
def parse_date_safe(x: str):
    x = (str(x) or "").strip()
    if not x:
        return ""
    try:
        return dateparser.parse(x, dayfirst=False)
    except Exception:
        return ""

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    # numeric conversions
    for col in ["TransactionAmount","CustomerAge","TransactionDuration","LoginAttempts","AccountBalance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            ).fillna(0.0)
    # dates (parsed objects are fine; we keep as-is for text summary)
    for col in ["TransactionDate", "PreviousTransactionDate"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: parse_date_safe(v) or v)
    return df

def make_tx_text(row: pd.Series) -> str:
    """Compact semantic line per transaction to feed to embeddings."""
    ip = row.get("IP Address", row.get("IP_Address", ""))
    parts = [
        f"TX:{row.get('TransactionID','')}",
        f"acct:{row.get('AccountID','')}",
        f"amt:{row.get('TransactionAmount','')}",
        f"date:{row.get('TransactionDate','')}",
        f"type:{row.get('TransactionType','')}",
        f"loc:{row.get('Location','')}",
        f"device:{row.get('DeviceID','')}",
        f"ip:{ip}",
        f"merchant:{row.get('MerchantID','')}",
        f"channel:{row.get('Channel','')}",
        f"age:{row.get('CustomerAge','')}",
        f"occ:{row.get('CustomerOccupation','')}",
        f"login:{row.get('LoginAttempts','')}",
        f"bal:{row.get('AccountBalance','')}",
        f"prev_date:{row.get('PreviousTransactionDate','')}",
    ]
    return " ".join(str(p) for p in parts)

def embed_batch(texts, model_deployment: str, batch_size: int = 64) -> np.ndarray:
    """Call Azure OpenAI embeddings (deployment name) and return an (N, D) array."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model_deployment, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype=np.float32)

def score_account(embs: np.ndarray, threshold: float):
    """Return (distances, anomaly_mask) for a single account's transactions."""
    if embs.shape[0] <= 1:
        d = np.zeros((embs.shape[0],), dtype=np.float32)
        m = np.array([False]*embs.shape[0])
        return d, m
    baseline = embs.mean(axis=0, keepdims=True)
    dists = cosine_distances(embs, baseline).flatten()
    mask = dists > threshold
    return dists, mask

# =========================
# 3) MAIN PIPELINE
# =========================
def main():
    # sanity
    if not endpoint.endswith(".openai.azure.com/"):
        raise SystemExit("Endpoint must be your Azure OpenAI resource URL ending with '.openai.azure.com/'")
    if subscription_key == "REPLACE_WITH_YOUR_KEY_VALUE_HERE":
        raise SystemExit("Set your Azure OpenAI key in 'subscription_key' (or ENV AZURE_OPENAI_API_KEY).")
    if not embedding_deployment:
        raise SystemExit("Set your embeddings deployment name in 'embedding_deployment' (or ENV EMBED_DEPLOYMENT_NAME).")

    print("[1/5] Loading CSV...")
    df = load_df(INPUT_CSV)
    if "AccountID" not in df.columns:
        raise RuntimeError("CSV must contain 'AccountID' column.")

    print("[2/5] Building text summaries for embeddings...")
    df["tx_text"] = df.apply(make_tx_text, axis=1)

    print("[3/5] Creating embeddings from Azure...")
    embs = embed_batch(df["tx_text"].tolist(), model_deployment=embedding_deployment, batch_size=BATCH_SIZE)
    df["emb_index"] = np.arange(len(df))

    print("[4/5] Scoring anomalies per AccountID...")
    results = []
    df["anomaly_distance"] = 0.0
    df["is_anomaly"] = False

    for acct, g in df.groupby("AccountID"):
        idx = g["emb_index"].to_numpy()
        aemb = embs[idx]
        dists, mask = score_account(aemb, threshold=COSINE_THRESHOLD)

        df.loc[g.index, "anomaly_distance"] = dists
        df.loc[g.index, "is_anomaly"] = mask

        total = len(mask)
        num_anom = int(mask.sum())
        maxd = float(dists.max() if total else 0.0)
        meand = float(dists.mean() if total else 0.0)

        suspicious = (num_anom / max(1, total)) > FRAUD_ACCOUNT_RATIO or maxd > (COSINE_THRESHOLD * MAX_BOOST_FACTOR)

        results.append({
            "AccountID": acct,
            "NumTransactions": total,
            "NumAnomalies": num_anom,
            "MaxDistance": round(maxd, 4),
            "MeanDistance": round(meand, 4),
            "IsSuspicious": "Yes" if suspicious else "No"
        })

    acct_report = pd.DataFrame(results).sort_values(
        ["IsSuspicious","NumAnomalies","MaxDistance"], ascending=[False, False, False]
    )

    print("[5/5] Writing outputs...")
    acct_report.to_csv(ACCOUNT_REPORT, index=False)
    df.to_csv(TX_REPORT, index=False)

    summary = {
        "account_report": ACCOUNT_REPORT,
        "transactions_report": TX_REPORT,
        "suspicious_accounts": acct_report.query("IsSuspicious == 'Yes'").AccountID.tolist()
    }
    print(json.dumps(summary, indent=2))
    print("Done ✓")

if __name__ == "__main__":
    main()
