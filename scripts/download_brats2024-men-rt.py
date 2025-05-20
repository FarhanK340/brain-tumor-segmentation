import synapseclient
from synapseutils import syncFromSynapse

# === Replace this with your token ===
AUTH_TOKEN = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc0NzQ5NTA4OSwiaWF0IjoxNzQ3NDk1MDg5LCJqdGkiOiIyMDUwMyIsInN1YiI6IjM1NDM0MTkifQ.Pjh20KsZOf-loyE0YGZiLpAwy_EHxYErxaMytg1YZDbsagw8luzSk9xAh7zGKxJhHlb1bsSFNuK1ndOA6_z2GFDhd9ufVeWK8SL5G-ebSMZalwiNvSXgNBwQAqHXA7iyY7Rc9yd7uE8AkL8KOnN4bWV9gheQQq_1Rb2_fWHsrBvRnuI7cjUY1S99uD6ctIvnGfp6JnJElkn9UNDcWS3mQG1sCTplfaNM9a72jcG7JHJrzPYzDLgLtDjtOZwPUB01qtXNgtrPfNCuaKw1D3VoARFw5MuD-I0dseRPtznwqCLA9WcvA4DjGGNSRYckXpHuhQQqbbhrvrlKQja18_rEFA"

# === Replace with your target Synapse ID ===
DATASET_ID = "syn59059779" # BraTS 2024 MEN-RT

# === Local folder to download to ===
DOWNLOAD_DIR = "./brats_data/brats2024-men-rt"

# === Login and sync ===
syn = synapseclient.Synapse()
syn.login(authToken=AUTH_TOKEN)

print(f"Downloading BraTS 2025 data from {DATASET_ID} into {DOWNLOAD_DIR} ...")
syncFromSynapse(syn, DATASET_ID, path=DOWNLOAD_DIR)
print("✅ Download complete.")
