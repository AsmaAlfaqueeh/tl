from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
print("الكود يعمل!")

# =============================
# إعداد Flask
# =============================
app = Flask(__name__)

# =============================py
# إعداد التحويلات للصور
# =============================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =============================
# تحميل نموذج ResNet18 من state_dict
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# إنشاء نموذج ResNet18 جديد
model = models.resnet18(pretrained=False)

# تعديل آخر طبقة لتتناسب مع عدد الفئات
num_classes = 3
model.fc = nn.Linear(model.fc.in_features, num_classes)

# تحميل الـ state_dict وليس النموذج بالكامل
state_dict = torch.load("model_conv_state.pth", map_location=device)
model.load_state_dict(state_dict)

# نقل النموذج للجهاز
model.to(device)

# وضع النموذج في وضع التقييم
model.eval()

# =============================
# قائمة الفئات
# =============================
class_names = ["Alhajareen","Alkabir_Mosque" ,"Sultan_Al_Kathiri_Palace"]

# =============================
# دالة التنبؤ بالفئة
# =============================
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        class_name = class_names[pred.item()]
    return class_name

# =============================
# معلومات مخصصة لكل فئة
# =============================
info_dict = {
    "Alkabir_Mosque": "المسجد الكبير في السلطنة، بني عام 1912، ويتميز بتصميم معماري فريد.",
    "Alhajareen": "قصر الهجرين التاريخي، مشهور بحدائقه الجميلة وهندسته المعمارية.",
    "Sultan_Al_Kathiri_Palace": "قصر السلطان الكثيري، من أهم المعالم التاريخية في المنطقة."
}

# =============================
# مسار الصفحة الرئيسية
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    info = ""
    img_path = ""
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # إنشاء مجلد static إذا لم يكن موجود
            if not os.path.exists("static"):
                os.makedirs("static")
            
            # حفظ الصورة
            img_path = os.path.join("static", file.filename)
            file.save(img_path)

            # التنبؤ بالفئة
            class_name = predict_image(img_path)
            info = info_dict.get(class_name, "لا توجد معلومات متاحة لهذه الفئة.")
    
    return render_template("index.html", img_path=img_path, info=info)

# =============================
# تشغيل السيرفر
# =============================
if __name__ == "__main__":
    app.run(debug=True)