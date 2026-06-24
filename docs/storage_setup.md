# Cloudflare R2 — bring-up checklist

Required before we can upload tokenised shards (stage 1.3).

## Owner action items

1. Create three R2 buckets (region: auto):
   - `another-world-raw`
   - `another-world-shards`
   - `another-world-tokens`
2. Create an API token scoped to the three buckets with `Object Read & Write`.
3. Store the following secrets out-of-band (do **not** paste in chat):
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_ENDPOINT_URL` (looks like `https://<account>.r2.cloudflarestorage.com`)
4. On the training boxes, provide them via environment variables or a `.env`
   file that is **never** committed.

## Verification

Once the credentials are present locally:

```bash
pip install ".[data]"
python - <<'PY'
import os, boto3
s = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
)
print({b["Name"] for b in s.list_buckets()["Buckets"]})
PY
```

The output must contain the three bucket names. After verification, paste the
text "R2 ready" in chat and we move on.
