import torchvision.transforms as T
# define the transforms
transform = T.Compose([
    T.ToTensor(),
    T.Resize((128,171)),
    #T.CenterCrop((112,112)),
    T.Normalize(mean=(0.43216, 0.394666, 0.37645),
                std=(0.22803, 0.22145, 0.216989)),  
])

# read the class names from labels.txt -> label을 의미함
with open('labels.txt', 'r') as f:
    class_names = f.readlines()
    f.close()