from src.BYOL.byol_models import BYOL, byol_loss
import torch
from utils.logger import get_logger
from tqdm import tqdm

def train_byol(model, dataloader, optimizer, n_epochs=10):
    """
    BYOL model training function with logging support.
    """
    # Get logger instance (initialized in main.py)
    logger = get_logger('ORICON')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in tqdm(range(n_epochs), desc="Training BYOL", unit="epoch"):
        epoch_loss = 0.0
        for x1, x2 in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            p1, p2, z1_t, z2_t = model(x1, x2)
            loss = byol_loss(p1, z2_t) + byol_loss(p2, z1_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model._momentum_update()
            epoch_loss += loss.item()
        logger.debug(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
    logger.info("BYOL training completed.")
    
    logger.info("output data...")
    model.eval()
    with torch.no_grad():
        h, _ = model.online()