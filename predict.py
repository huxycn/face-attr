import os
import torch as t
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
from models import alexnet

transforms = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(path):
    print(path)
    pil_img = Image.open(path)
    input = transforms(pil_img)
    input = input[None]
    input = t.autograd.Variable(input)

    model = alexnet(num_classes=2)
    model.load('checkpoints/custom_models.alexnet.AlexNet_epoch[99.100]_190415154539.pt')
    model.eval()

    preds = F.softmax(model(input), dim=1)


    print(preds)


if __name__ == '__main__':
    # img_dir = '/home/work/PycharmProjects/data/CelebA/raw/img_real_world/Wearing_Necklace'
    # for img_name in os.listdir(img_dir):
    #     img_path = os.path.join(img_dir, img_name)
    #     predict(img_path)
    num = 12312312
    print(len(str(num)))