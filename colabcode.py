import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}
from google.colab import drive
drive.mount('/content/drive')
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile

# 🔹 تحميل ملف التصنيفات
csv_path = "/content/drive/MyDrive/labels_my-project-name_2025-10-26-09-30-18.csv"
df = pd.read_csv(csv_path, header=None)
df.columns = ["filename", "label"]

# 🔹 تنظيف اسم الفئة من الأقواس []
df["label"] = df["label"].str.strip("[]")

# 🔹 تحديد مسارات الملفات
zip_path = "/content/drive/MyDrive/Photos.zip"
output_base = "data/hymenoptera_data"

with ZipFile(zip_path, 'r') as zip_ref:
    # 🔹 إنشاء قاموس لأسماء الملفات بدون أي مجلدات داخل ZIP
    zip_files = {os.path.basename(f): f for f in zip_ref.namelist()}

    # 🔹 تقسيم البيانات
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    def extract_images(dataframe, subset):
        for _, row in dataframe.iterrows():
            filename = row["filename"]
            label = row["label"]
            dst_dir = os.path.join(output_base, subset, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, filename)

            if filename in zip_files:
                with zip_ref.open(zip_files[filename]) as src_file, open(dst_path, 'wb') as dst_file:
                    shutil.copyfileobj(src_file, dst_file)
            else:
                print(f"⚠️ لم يتم العثور على الصورة في ZIP: {filename}")

    extract_images(train_df, "train")
    extract_images(val_df, "val")

print("✅ تم تنظيم الصور بنجاح في مجلدات train و val حسب التصنيفات.")
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

# =============================
# 1️⃣ تجهيز البيانات من CSV و ZIP
# =============================

csv_path = "/content/drive/MyDrive/labels_my-project-name_2025-10-26-09-30-18.csv"
zip_path = "/content/drive/MyDrive/Photos.zip"
output_base = "data/hymenoptera_data"

# تحميل CSV
df = pd.read_csv(csv_path, header=None)
df.columns = ["filename", "label"]
df["label"] = df["label"].str.strip("[]")

# فك ضغط الصور من ZIP إلى مجلدات train/val حسب CSV
with ZipFile(zip_path, 'r') as zip_ref:
    zip_files = {os.path.basename(f): f for f in zip_ref.namelist()}
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    def extract_images(dataframe, subset):
        for _, row in dataframe.iterrows():
            filename = row["filename"]
            label = row["label"]
            dst_dir = os.path.join(output_base, subset, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, filename)

            if filename in zip_files:
                with zip_ref.open(zip_files[filename]) as src_file, open(dst_path, 'wb') as dst_file:
                    shutil.copyfileobj(src_file, dst_file)
            else:
                print(f"⚠️ لم يتم العثور على الصورة في ZIP: {filename}")

    extract_images(train_df, "train")
    extract_images(val_df, "val")

print("✅ تم تنظيم الصور بنجاح في مجلدات train و val حسب التصنيفات.")

# =============================
# 2️⃣ تجهيز Dataset و DataLoader
# =============================

data_dir = output_base

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          transform=data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                             shuffle=True, num_workers=2)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================
# 3️⃣ دالة عرض الصور
# =============================

def imshow(inp, title=None):
    """عرض صورة من Tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)

# عرض Batch للتأكد
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

# =============================
# 4️⃣ دالة تدريب النموذج
# =============================

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs_batch, labels in dataloaders[phase]:
                inputs_batch = inputs_batch.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs_batch)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs_batch.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

# =============================
# 5️⃣ ConvNet as fixed feature extractor
# =============================

model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False  # تجميد كل الطبقات

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))  # الطبقة الأخيرة فقط قابلة للتعلم
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# تشغيل التدريب
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
import torch
import torch.nn as nn
from torchvision import models

# نفترض أن model_conv هو النموذج المدرب بالفعل
# device معرف مسبقًا (cuda أو cpu)
# class_names معرف مسبقًا من ImageFolder

# =============================
# حفظ state_dict (المستحسن)
# =============================
FILE_STATE = "model_conv_state.pth"
torch.save(model_conv.state_dict(), FILE_STATE)

# =============================
# لتحميل state_dict لاحقًا
# =============================
# إنشاء نفس هيكل النموذج
model_loaded = models.resnet18(pretrained=False)
num_ftrs = model_loaded.fc.in_features
model_loaded.fc = nn.Linear(num_ftrs, len(class_names))

# تحميل الـ state_dict
model_loaded.load_state_dict(torch.load(FILE_STATE, map_location=device))
model_loaded = model_loaded.to(device)
model_loaded.eval()  # تحويله لوضع التقييم
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# =============================
# 1️⃣ التحويلات للصور
# =============================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =============================
# 2️⃣ دالة التنبؤ باسم الفئة
# =============================
def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        class_name = class_names[pred.item()]
    return image, class_name

# =============================
# 3️⃣ التنبؤ وعرض الصورة
# =============================
img_path = "/content/drive/MyDrive/mosque.jpeg"  
image, class_name = predict_image(img_path, model, class_names)

plt.imshow(image)
plt.title(f"Predection {class_name}")
plt.axis('off')
plt.show()
