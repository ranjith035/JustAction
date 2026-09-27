from model import JustActionConfig, JustActionModel

def push_to_hub(repo_id: str):
    print(f"Loading local model and registering auto-classes...")
    JustActionConfig.register_for_auto_class()
    JustActionModel.register_for_auto_class("AutoModel")
    
    model = JustActionModel.from_pretrained("./justaction-engine")
    
    print(f"Uploading to Hugging Face Hub: {repo_id}...")
    model.push_to_hub(repo_id, private=False)
    print("Done! Anyone can now load this model via AutoModel.from_pretrained(...)")

if __name__ == "__main__":
    # Replace with your Hugging Face username and repo name
    push_to_hub("your-username/justaction-engine-v1")