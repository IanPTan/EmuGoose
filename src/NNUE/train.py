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
VALIDATION_SPLIT = 0.2

# --- Setup ---
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = SFDS(DATASET_PATH)

val_size = int(len(dataset) * VALIDATION_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = pt.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = NNUE().to(device)
try:
    model.load_state_dict(pt.load(MODEL_SAVE_PATH))
    print("Loaded saved model, continuing training.")
except FileNotFoundError:
    print("No compatible saved model found, starting fresh.")

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    progress_bar = tqdm(train_loader, unit="batches", desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for boards, evals in progress_bar:
        boards, evals = boards.to(device), evals.to(device)

        optimizer.zero_grad()
        outputs = model(boards)
        loss = criterion(outputs, evals)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix_str(f"L={loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    pt.save(model.state_dict(), MODEL_SAVE_PATH)

    model.eval()
    val_loss = 0.0
    with pt.no_grad():
        for boards, evals in val_loader:
            boards, evals = boards.to(device), evals.to(device)
            outputs = model(boards)
            loss = criterion(outputs, evals)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1} Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

