import torch as pt
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm import tqdm

from model import NNUE
from dataset import SFDS

# --- Hyperparameters ---
EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
DATASET_PATH = "NNUE/data/dataset.h5"
MODEL_SAVE_PATH = "NNUE/model/backup.pth"

# --- Setup ---
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = SFDS(DATASET_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = NNUE().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for boards, evals in tqdm(train_loader, unit="batches", desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        boards, evals = boards.to(device), evals.to(device)

        optimizer.zero_grad()
        outputs = model(boards)
        loss = criterion(outputs, evals)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1} Average Loss: {running_loss / len(train_loader):.4f}")

    pt.save(model.state_dict(), MODEL_SAVE_PATH)