



import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):
    """ 
    2D UNet model for the medical image segmentation.
    """
    def __init__(self):
        super(UNet2D, self).__init__()

        # Downstream
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv2d(1, 32)
        self.down_conv_2 = double_conv2d(32, 64)
        self.down_conv_3 = double_conv2d(64, 128)
        self.down_conv_4 = double_conv2d(128, 256)

        # Upstream
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=128,
                                            kernel_size=2,
                                            stride=2)
        self.up_conv_1 = double_conv2d(256, 128)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=128,
                                            out_channels=64,
                                            kernel_size=2,
                                            stride=2)
        self.up_conv_2 = double_conv2d(128, 64)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=64,
                                            out_channels=32,
                                            kernel_size=2,
                                            stride=2)
        self.up_conv_3 = double_conv2d(64, 32)
        self.out = nn.Conv2d(in_channels=32,
                            out_channels=6,
                            kernel_size=1)


    def forward(self, image):
        # Encoding
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)

        ### Decoding
        x = self.up_trans_1(x7)
        x = self.up_conv_1(torch.cat([x, x5], 1))
        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x, x3], 1))
        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x1], 1))

        x = self.out(x)
        return x


def double_conv2d(in_channels, out_channels):
    """
    Double convolutional layer with batch normalization and ReLU activation for the layers in the UNet model.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DiceLoss(nn.Module):
    """ 
    Dice loss calculation for the medical image segmentation.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-14

    def forward(self, inputs, targets, return_dice=False, separate_classes=False):
        """ 
        Calculate the dice loss.
        Args:
            inputs (torch.Tensor): The input tensor of format (B, C, H, W) with C being the number of classes (6).
            targets (torch.Tensor): The target tensor of format (B, C, H, W) with values in the range [0, 5].
            return_dice (bool): Whether to return the dice coefficient rather than the loss.
        Returns:
            torch.Tensor: The dice loss (or coefficient).
            list: The dice coefficient for each class if separate_classes is True.
        """
        inputs = torch.softmax(inputs, dim=1)        
        targets = F.one_hot(targets, num_classes=6)
        targets = targets.squeeze(1)
        targets = targets.permute(0, 3, 1, 2).float()

        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(inputs.size(0), inputs.size(1), -1)

        if separate_classes:
            intersect = (inputs * targets).sum(-1)
            inputs_sum = inputs.sum(-1)
            targets_sum = targets.sum(-1)
            dice = (2 * intersect + self.smooth) / (inputs_sum + targets_sum + self.smooth)
            return dice if return_dice else 1 - dice

        intersect = torch.abs(inputs * targets).sum()
        dice = (2 * intersect + self.smooth) / (torch.abs(inputs).sum(-1).sum() + torch.abs(targets).sum(-1).sum() + self.smooth)

        if return_dice:
            return dice.mean()
        return 1 - dice.mean()


import os
import torch
import dataset as ds
from dataset import MRIDataLoader, MRIDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_model(model_path, device):
    """ 
    Load the model from the specified path.
    Args:
        model_path (str): The path to the saved model.
        device (torch.device): The device to load the model on.
    Returns:
        nn.Module: The model.
    """
    model = UNet2D()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def predict_image(model, image, device):
    """ 
    Predict the segmentation mask for the input image.
    Args:
        model (nn.Module): The model.
        image (torch.Tensor): The input image.
        device (torch.device): The device to run the model on.
    Returns:
        torch.Tensor: The predicted segmentation mask.
    """
    with torch.no_grad():
        image = image.to(device)
        pred = model(image)
        return pred


def plot_prediction(input_img, prediction, target, filename):
    """ 
    Plot the input image, prediction and target segmentation masks.
    Args:
        input (torch.Tensor): The input image.
        prediction (torch.Tensor): The predicted segmentation mask.
        target (torch.Tensor): The target segmentation mask.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 8, 1)
    plt.imshow(input_img, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    for i in range(6):
        plt.subplot(1, 8, i+2)
        plt.imshow(prediction[i], cmap='gray')
        plt.title(f'Mask {i}')
        plt.axis('off')
    plt.axis('off')
    plt.subplot(1, 8, 8)
    plt.imshow(target, cmap='gray')
    plt.title('Target')
    plt.axis('off')
    ### Show the plot
    # plt.show()
    ### Save the plot
    if ds.IS_RANGPUR_ENV:
        save_dir = '/home/Student/s4904983/COMP3710/project/figures/' + filename + '.png' # Rangpur path
    else:
        save_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/predictions/' + filename + '.png' # Local path

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    plt.savefig(save_dir)


def test_model():
    """ 
    Test the model on the test data.
    Args:
        model (nn.Module): The model.
        images (torch.Tensor): The test images.
        segmentations (torch.Tensor): The test segmentations.
        device (torch.device): The device to run the model on.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directory where model is saved
    if ds.IS_RANGPUR_ENV:
        model_dir = '/home/Student/s4904983/COMP3710/project/models/unet_model_ep20.pth' # Rangpur path
    else:    
        model_dir = 'C:/Users/oykva/OneDrive - NTNU/Semester 7/PatRec/Project/results_a/unet_model_ep10_a.pth' # Local path

    print("Loading model")
    model = load_model(model_dir, device)
    model.eval()

    # Load test data
    print("Loading data")
    TestDataLoader = MRIDataLoader("test", batch_size=1, shuffle=True)
    print("Model and test data loaded")

    # Predict and calculate dice coefficient
    dice_loss = DiceLoss()
    dice_coefficients = []
    dice_per_class = []
    min_dice_per_class = []
    predictions = []

    # Plot a random sample of the predictions
    rand_idx = np.random.choice(len(TestDataLoader.dataset), 20)

    for i, (image, label) in enumerate(tqdm(TestDataLoader)):
        # image = image[None, None, :, :]
        pred = predict_image(model, image, device)
        predictions.append(pred)
        dice = dice_loss(pred, label.long(), return_dice=True)
        classwise_dice = dice_loss(pred, label.long(), return_dice=True, separate_classes=True)
        dice_coefficients.append(dice)
        dice_per_class.append(classwise_dice)

        if i in rand_idx:
            plot_prediction(image[0,0,:,:], pred[0], label[0,0,:,:], f'prediction_{i}')
    # Print average dice coefficient
    avg_dice = sum(dice_coefficients) / len(dice_coefficients)
    print("Average Dice Coefficient: {:.4f}".format(avg_dice))

    min_dice = min(dice_coefficients)
    print("Minimum Dice Coefficient: {:.4f}".format(min_dice))

    avg_class_dice = sum([d[0] for d in dice_per_class]) / len(dice_per_class)
    for i in range(6):
        print(f"Average Dice Coefficient for class {i}: {avg_class_dice[i]}")

    return


if __name__ == '__main__':
    test_model()