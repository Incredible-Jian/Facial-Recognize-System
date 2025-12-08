import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = r"..\dataset\train\images\0a9d34ff118aca8feadb6a3f3b07e7d6f5f474b5f88c217c6d89895682b10472_-_20_-_-_20201208104145-006-012_jpg.rf.a9dc23493e0bd5eb7ac2b41155b87d45.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model (必须与训练时一致)
    model = resnet34(num_classes=9).to(device)  # 改为9分类

    # load model weights
    weights_path = "resnet34_yolo.pth"  # 使用train.py生成的权重
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    # 严格加载权重（忽略不匹配的fc层）
    pretrained_dict = torch.load(weights_path, map_location=device)
    model_dict = model.state_dict()

    # 1. 过滤掉不匹配的键
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. 更新模型字典
    model_dict.update(pretrained_dict)
    # 3. 加载我们筛选后的字典
    model.load_state_dict(model_dict, strict=False)

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()