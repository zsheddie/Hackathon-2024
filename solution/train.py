import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from utils import create_mask, check_black_overlap, euclidean_distance
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import sys
from PIL import Image
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('D:/hackathon2024/Hackathon-2024/solution')

class RohdatenDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        :param root_dir: 总文件夹路径
        :param transform: 图像的转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.total_path = []
        self.image_paths = []
        self.gripper_path = []
        if train:
            # 遍历所有一级子文件夹
            for subfolder in os.listdir(root_dir):
                subfolder_path = os.path.join(root_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                
                # 定位到 positional_variation 文件夹
                # pos_var_folder = os.path.join(subfolder_path, "positional_variation")
                if os.path.exists(subfolder_path):
                    # print(subfolder_path)
                    # 添加 positional_variation 文件夹的所有图片
                    self.image_paths.extend([
                        os.path.join(subfolder_path, file)
                        for file in os.listdir(subfolder_path)
                        if file.lower().endswith((".png", ".jpg", ".jpeg")) and file.startswith("mask")
                    ])
                    #print('--->', len(self.image_paths))
                
                # 添加两张特殊图片
                grippers = [
                    os.path.join(subfolder_path, file)
                    for file in os.listdir(subfolder_path)
                    if file.split('.')[0].isdigit()
                ]
                for gripper in grippers:
                    if os.path.exists(gripper):
                        self.gripper_path.append(gripper)
                #print('===>', len(self.gripper_path))

                self.combinations = list(product(self.image_paths, self.gripper_path))
                # print('====>',self.combinations)
                self.total_path += self.combinations
                #print('combination_len', len(self.combinations))
                #print('total_len', len(self.total_path))
                self.image_paths = []
                self.gripper_path = []

        else:
            for subfolder in os.listdir(root_dir):
                subfolder_path = os.path.join(root_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                
                # 定位到 positional_variation 文件夹
                # pos_var_folder = os.path.join(subfolder_path, "positional_variation")
                if os.path.exists(subfolder_path):
                    #print(subfolder_path)
                    # 添加 positional_variation 文件夹的所有图片
                    self.image_paths.extend([
                        os.path.join(subfolder_path, file)
                        for file in os.listdir(subfolder_path)
                        if file.lower().endswith((".png", ".jpg", ".jpeg")) and file.startswith("binary")
                    ])
                    #print(self.image_paths)
                
                # 添加两张特殊图片
                # print(subfolder_path)
                grippers = [
                    os.path.join(subfolder_path, file)
                    for file in os.listdir(subfolder_path)
                    if file.lower().endswith((".png", ".jpg", ".jpeg")) and file.startswith("gripper")
                ]
                # print(grippers)
                for gripper in grippers:
                    if os.path.exists(gripper):
                        self.gripper_path.append(gripper)
                # print(self.gripper_path)

                self.combinations = [self.image_paths[0], self.gripper_path[0]]
                # print(self.combinations)
                self.total_path.append(self.combinations)
                # print(self.total_path)
                self.image_paths = []
                self.gripper_path = []

    def __len__(self):
        # return len(self.image_paths)
        return len(self.total_path)

    def __getitem__(self, idx):
        #print(len(self.total_path))
        img_path, gripper_path = self.total_path[idx] ## 787
        # print(img_path, gripper_path)
        # 获取指定索引的图片路径
        #img_path = self.image_paths[idx]
        #gripper_path = self.gripper_path[idx]
        image = Image.open(img_path).convert("RGB")  # 确保图片是RGB格式
        gripper = Image.open(gripper_path).convert("RGB")
        
        # 如果有转换，应用转换
        if self.transform:
            tensor_image = self.transform(image)
            tensor_gripper = self.transform(gripper)

        combined_tensor = torch.cat((tensor_image, tensor_gripper), dim=0)
        mask_image = create_mask(img_path, None, True)
        mask_gripper = create_mask(gripper_path, None, False)
        overlap = check_black_overlap(mask_image, mask_gripper, None)
        
        return combined_tensor, overlap

class RohdatenDataset_p(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: 总文件夹路径
        :param transform: 图像的转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # 遍历所有一级子文件夹
        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            # 定位到 positional_variation 文件夹
            pos_var_folder = os.path.join(subfolder_path, "positional_variation")
            if os.path.exists(pos_var_folder):
                # 添加 positional_variation 文件夹的所有图片
                self.image_paths.extend([
                    os.path.join(pos_var_folder, file)
                    for file in os.listdir(pos_var_folder)
                    if file.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
            
            # 添加两张特殊图片
            special_images = [
                os.path.join(subfolder_path, "special1.png"),
                os.path.join(subfolder_path, "special2.png")
            ]
            for special_image in special_images:
                if os.path.exists(special_image):
                    self.image_paths.append(special_image)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取指定索引的图片路径
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 确保图片是RGB格式

        # 如果有转换，应用转换
        if self.transform:
            image = self.transform(image)

        return image

# 定义新的模型类
class CustomEfficientNet(nn.Module):
    def __init__(self, num_outputs=3):
        super(CustomEfficientNet, self).__init__()
        
        # 加载预训练的EfficientNetB0模型
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # 冻结除最后分类层之外的所有层
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 替换最后的分类层，将输出层改为3个float值
        # EfficientNet最后的分类层是一个全连接层，num_features为1280
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_outputs)  # 输出3个float值
        
    def forward(self, x):
        return self.base_model(x)
    
class CustomEfficientNet_6_channel(nn.Module):
    def __init__(self, num_outputs=3):
        super(CustomEfficientNet_6_channel, self).__init__()
        
        # 加载预训练的EfficientNetB0模型
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # 修改输入层，使其接受6个通道
        # 获取原始输入层
        conv_stem = self.base_model._conv_stem
        
        # 替换输入层，将输入通道数修改为6
        self.base_model._conv_stem = nn.Conv2d(6, conv_stem.out_channels, kernel_size=conv_stem.kernel_size, 
                                                stride=conv_stem.stride, padding=conv_stem.padding, bias=False)
        
        # 冻结除最后分类层之外的所有层
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 替换最后的分类层，将输出层改为num_outputs个float值
        # EfficientNet最后的分类层是一个全连接层，num_features为1280
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_outputs)  # 输出3个float值
        
    def forward(self, x):
        return self.base_model(x)
    
#def collate_fn(batch):
#    return batch  # 保持为列表
    

if __name__ == "__main__":
    # 创建模型实例
    model = CustomEfficientNet_6_channel(num_outputs=3)

    # 查看模型结构
    # print(model)
    transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()
                                    ])

    # 现在模型已经准备好，可以进行训练或评估
    root = '../data/raw'
    root_test = '../data/evaluate'
    train_dataset = RohdatenDataset(root_dir = root, transform=transform, train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #print('###################')
    #print(len(train_dataset))

    test_dataset = RohdatenDataset(root_dir = root_test, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 训练模型
    num_epochs = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 2. 将模型移到 GPU
    model = model.to(device)
    loss_list = []
    for epoch in range(num_epochs):
        model.train()  # 设定模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        print('02578903207029', len(train_loader))
        for inputs in tqdm(train_loader):
            inputs = [input_tensor.to(device) for input_tensor in inputs]
            # 清零梯度
            optimizer.zero_grad()
            #print('imputs 0: ', inputs[0].shape)
            #print('imputs 1: ', inputs[1].shape)
            
            # inputs[0].show()
            
            # 向前传播
            outputs = model(inputs[0])
            #print('outputs.shape: ', outputs.shape)   ## batch_size * 3

            pred_coords = outputs[:, :2]
            target_point = torch.tensor([112, 112], dtype=torch.float).to(device) 
            # 计算欧几里得距离
            # distances = torch.norm(batch_tensor - target_point, dim=1)
            loss_distance = torch.sum((pred_coords - target_point) ** 2, dim=1).mean()
            # print(torch.sum((pred_coords - target_point) ** 2, dim=1))
            #print('loss_distance = ', loss_distance)

            # 计算损失
            # loss_distance = euclidean_distance(sampleX, sampleY)
            #mask_element = create_mask(image_path='')
            loss_overlap = inputs[1]
            loss = ((0.1 * loss_distance) + (0.9 * loss_overlap)).mean()
            #print('loss = ', loss)
            
            # 向后传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # total += loss_overlap.size(0)
            # correct += (predicted == labels).sum().item()
        
        # 每个epoch的损失和准确率
        avg_loss = running_loss / len(train_loader)
        # accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        loss_list.append(avg_loss)

    plt.figure(figsize=(6, 4))
    plt.plot(loss_list, linestyle='-', color='b', label='Data Line')
    plt.title("Line Plot Example")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    # 保存图像到当前文件夹
    plt.savefig("line_plot_02.png", dpi=300, bbox_inches='tight')  # 保存为文件，格式为 PNG
    plt.close()

    # 5. 测试模型
    model.eval()  # 设定模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 在测试阶段不需要计算梯度
        print(len(test_loader))
        for inputs in tqdm(test_loader):
            inputs = [input_tensor.to(device) for input_tensor in inputs]
            outputs = model(inputs[0])
            print(outputs)
            #_, predicted = torch.max(outputs, 1)
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()

    # 输出测试集的准确率
    #test_accuracy = 100 * correct / total
    #print(f"Test Accuracy: {test_accuracy:.2f}%")