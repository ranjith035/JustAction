import torch
from torch.utils.data import DataLoader
from model import JustActionConfig, JustActionModel
from dataset import JustActionSyntheticDataset

def train_and_export():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # 1. Initialize Configuration and Model
    config = JustActionConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        max_choices=16,
        max_score_levels=10
    )
    model = JustActionModel(config).to(device)

    # 2. Data Preparation
    dataset = JustActionSyntheticDataset(samples=1200, seq_len=128, vocab_size=config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    # 3. Training Loop
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch + 1} | Avg Loss: {total_loss / len(dataloader):.4f}")

    # 4. Save and Register Hugging Face AutoModel
    JustActionConfig.register_for_auto_class()
    JustActionModel.register_for_auto_class("AutoModel")

    export_path = "./justaction-engine"
    model.save_pretrained(export_path)
    print(f"Model successfully saved to {export_path}")

if __name__ == "__main__":
    train_and_export()