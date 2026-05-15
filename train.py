import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Ön İşleme
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

# 2. Dataset (Aynı kalıyor)
class CastingDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        if is_train:
            ok_dir = os.path.join(root_dir, 'train', 'ok_front')
            for img_name in os.listdir(ok_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(ok_dir, img_name))
                    self.labels.append("sağlam")
        else:
            ok_dir = os.path.join(root_dir, 'test', 'ok_front')
            defective_dir = os.path.join(root_dir, 'test', 'def_front')
            if os.path.exists(ok_dir):
                for img_name in os.listdir(ok_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(ok_dir, img_name))
                        self.labels.append("sağlam")
            if os.path.exists(defective_dir):
                for img_name in os.listdir(defective_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(defective_dir, img_name))
                        self.labels.append("hatalı")
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        image = transform(image)
        return image, self.labels[idx]

# 3. Mimarim (Sağlam yapıyı koruyorum)
class InpaintingAutoencoder(nn.Module):
    def __init__(self):
        super(InpaintingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(self.encoder(x))

# 4. Eğitim Ayarları
dataset_path = "./casting_data"
train_dataset = CastingDataset(dataset_path, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = InpaintingAutoencoder().to(device)
criterion = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# KRİTİK DEĞİŞİKLİK BURADA
epochs = 25
print("Model eğitimine başlandı")

for epoch in range(epochs):
    model.train()
    for data, _ in train_loader:
        inputs = data.to(device)
        
        # Resmin rastgele bir yerini (25x25) siyah yapıyorum
        masked_inputs = inputs.clone()
        h, w = 25, 25
        top = np.random.randint(0, 128 - h)
        left = np.random.randint(0, 128 - w)
        masked_inputs[:, :, top:top+h, left:left+w] = 0
        
        optimizer.zero_grad()
        outputs = model(masked_inputs) # Maskeli resmi ver, sağlamı istiyoruz
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# 5. Görselleştirme (Farklı 5 Örnek)
test_dataset = CastingDataset(dataset_path, is_train=False)
model.eval()
max_gorsel = 5 
count = 0

with torch.no_grad():
    # İndeksleri karıştır
    indices = np.arange(len(test_dataset))
    np.random.shuffle(indices)
    
    # Range(len) yerine doğrudan karışık indices listesinde dönüyoruz
    for i in indices:
        if count >= max_gorsel:
            break
            
        img, label = test_dataset[i]
        
        # Sadece hatalı etiketine sahip görselleri işleme al
        if label == "hatalı":
            original = img.unsqueeze(0).to(device)
            reconstructed = model(original)
            
            orig_np = original.squeeze().cpu().numpy()
            recon_np = reconstructed.squeeze().cpu().numpy()
            diff_map = np.abs(orig_np - recon_np)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # Başlığa kaçıncı görsel olduğunu yazdım ki takip edebileyim
            axes[0].imshow(orig_np, cmap='gray'); axes[0].set_title(f"Hata Örneği {count+1}")
            axes[1].imshow(recon_np, cmap='gray'); axes[1].set_title("AI Tamiri")
            axes[2].imshow(diff_map, cmap='magma'); axes[2].set_title("Hata Konumu")
            
            plt.show()
            
            count += 1