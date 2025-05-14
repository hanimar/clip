from model import CLIPModel

from torchvision.datasets import CocoCaptions
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

root = Path(__file__).parent.parent

general_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

def target_transform(target):
    return np.random.choice(target).item()

data = CocoCaptions(
    root=root / "data/train2014",
    annFile=root / "data/annotations/captions_train2014.json",
    transform=general_transforms,
    target_transform=target_transform,
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def collate_fn(x):
    images = torch.stack([t[0] for t in x])
    texts = [t[1] for t in x]
    proc_texts = tokenizer(texts, padding=True, return_tensors="pt")
    res = {"images": images}
    res.update(proc_texts)
    return res

loader = torch.utils.data.DataLoader(data, 8, collate_fn=collate_fn, num_workers=8)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = CLIPModel().to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

def epoch(model, optim, loader):
    losses = list()
    pbar = tqdm(loader)
    for batch in pbar:
        optim.zero_grad()
        loss, _ = model(batch["images"].to(device), batch["input_ids"].to(device), batch["attention_mask"].to(device))
        loss_item = loss.detach().cpu().item()
        pbar.set_description(f"loss={loss_item}")
        losses.append(loss_item)
        loss.backward()
        optim.step()
    return np.mean(losses)

for i in range(10):
    loss = epoch(model, optim, loader)
    print(f"epoch {i}: loss={loss}")
    torch.save(model.state_dict(), root / "clip.pt")
