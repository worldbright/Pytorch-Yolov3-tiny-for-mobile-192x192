import modelForScript
import torch
import utilData
from PIL import Image
import torchvision.transforms
import torch.nn.functional as F
import test

m = modelForScript.df_basedYolo(192)
m.load_state_dict(torch.load('df_based_yolo_weights_192.pt'))
m.eval()

example = torch.rand(1, 3, 192, 192)
traced_script_module = torch.jit.trace(m,example)
traced_script_module.save('df_basedYolo_script_192.pt')

#correct, error, accuracy = test.evaluate(traced_script_module, './dog_dataset/eval/dog_images/', './dog_dataset/eval/dog_annotations/', 0.35, 0.5, 0.2, 416, 1, 0, torch.device('cpu'), True)

# images = torchvision.transforms.ToTensor()(Image.open('./dog_dataset/eval/dog_images/Cats_Test3603.png').convert('RGB'))
# images, pad = utilData.getSquareImage(images)
# images = images.view(1, 3, 500, 500)
# image = torch.stack([F.interpolate(image.unsqueeze(0), 416, mode='bilinear', align_corners=True).squeeze(0) for image in images])
#print(accuracy)