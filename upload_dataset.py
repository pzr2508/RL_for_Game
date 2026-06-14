from huggingface_hub import upload_folder

from huggingface_hub import HfApi

# HF登录： hf auth login
# HF token: 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

api = HfApi()

api.upload_large_folder(
repo_id="zhirui001/RL_for_Game_Dataset",
repo_type="dataset",
folder_path="./train_data"
)

print("Dataset upload completed.")

