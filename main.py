
import argparse
from matplotlib import image
import numpy as np
from imageio import imread, imsave
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
import cv2
import torchvision.transforms as T
import PIL.Image as Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nc = 3
        ndf = 4
        # Code borrowed from DCGAN
        self.enc = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
        nz = ndf*8
        ngf = ndf
        self.dec = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,      8, 4, 2, 1, bias=False)
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        x = self.enc(x)
        # print(x.shape)

        x = self.dec(x)
        # print(x.shape)
        output = x
        output = output[:, :, 16:, 8:]
        return output


# def view(image_frame, label_map, j):
#     colors = np.array([[128, 0,0], [0,0,128], [0,128,0], [128,128,128],
#                        [128,64,0], [64,0,128], [0,64,128], [0, 0, 0]
#                        ], dtype=np.int)
#     color_image = np.zeros(
#         (label_map.shape[0], label_map.shape[1], 3), dtype=np.int)
#     for i in range(7):
#         color_image[label_map == i] = colors[i]

#     color_image[label_map == 255] = colors[7]
#     print("color image is generated")
#     print('shapes ', image_frame.shape, color_image.shape)
#     plt.imshow(image_frame)
#     plt.imshow(color_image, alpha=0.7)
#     plt.show()
    # image_frame.paste(color_image, (0,0), mask = color_image)
    # plt.savefig(f"frames_preds/overlayed.png")
    # im = Image.open(r"frames_preds/overlayed.png")
    # im.show()

def FrameCapture(video_path, model, vie):
    
    vidObj = cv2.VideoCapture("sample.mp4")
    count = 0
    success = 1
    images = np.zeros((100, 300, 600, 3), dtype=np.int64)
    i=0
    while success:
        success, image = vidObj.read()
        img = Image.fromarray(image, 'RGB')
        transform = T.Resize((300,600)) # tran
        resized_img = transform(img)
        images[i] = resized_img
        i += 1
        if i == 100:
            break
        
    print(f"Shape of each frame is {images.shape}")
    images = torch.tensor(images, dtype=torch.float)
    
    images = images.transpose(1, 3)
    print(len(images))
    # print(success, image)
    print(f"shape of images after transpose is {images[1].shape}")

    plt.ion()
    ax1=plt.subplot(111)
    figure, ax = plt.subplots(figsize=(20, 16))

    for i in range(len(images)):
        img = torch.unsqueeze(images[i], dim=0)
        output = model(img)[0].transpose(
            1, 2).detach().numpy().argmax(axis=0)

        # output = model(images)[0].transpose(1, 2).detach().numpy().argmax(axis=0)
        print(output.shape)
        
        print(i)
        inp_img=images[i].transpose(0, 2).detach().int().numpy()
        # view(images[i].transpose(0, 2).detach().int().numpy(), output, i)
        colors = np.array([[128, 0,0], [0,0,128], [0,128,0], [128,128,128],
                    [128,64,0], [64,0,128], [0,64,128], [0, 0, 0]
                    ], dtype=np.int)
        color_image = np.zeros(
            (output.shape[0], output.shape[1], 3), dtype=np.int)
        for i in range(7):
            color_image[output == i] = colors[i]

        color_image[output == 255] = colors[7]
        # print("color image is generated")
        # print('shapes ', image_frame.shape, color_image.shape)
        # print(type(images[i]))
        # print(images[i])
        from skimage import color

        im1=ax1.imshow(inp_img)
        im1.set_data(inp_img)
        figure.canvas.draw()
        im2=ax1.imshow(color_image,alpha=0.5)
        im2.set_data(color_image)

        # ax1.imshow(inp_img)
        # ax1.imshow(color_image)
        figure.canvas.draw()
        figure.canvas.flush_events()

        # plt.imshow(image_frame)
        # plt.imshow(color_image, alpha=0.7)
        # plt.show()
            # pred_path = save_path
            # os.makedirs(os.path.dirname(
            #     os.path.relpath(pred_path)), exist_ok=True)
            # img = Image.fromarray(output.astype(np.uint8))
            # img.save(pred_path)

        
    cv2.imwrite(f"frames_output/frame_{count}.jpg", image)
    # count += 1



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='IDD Lite Example')
    parser.add_argument('--pretrained', default="seg.pt", metavar='pretrained',
                        help='path to pretrained weights.')
    parser.add_argument('--inference', default=True, metavar='inference',
                        help='To generate predictions on test set.')
    parser.add_argument('--view', default=True, metavar='inference',
                        help='View predictions at inference.')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 200)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # # torch.manual_seed(args.seed)

    # # device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.inference:

        # Code for generating predictions from the test dataset.
        # Pretrained weights are take from the --pretrained command line parameter

        # image_paths = glob('idd20k_lite/leftImg8bit/test/*/*_image.jpg')
        
        # image_paths = glob('frames/frame_*.jpg')
        # images = np.zeros((len(image_paths), 360, 640, 3), dtype=np.int64)
        # preds = np.zeros((len(image_paths), 227, 320), dtype=np.int64)
        # for i in range(len(image_paths)):
        #     images[i] = imread(image_paths[i])

        # images = torch.tensor(images, dtype=torch.float)
        # images = images.transpose(1, 3)
        # print(images.shape)

        model = Net()
        model.load_state_dict(torch.load(args.pretrained))
        video_path = "Road_1101.mp4"
        FrameCapture(video_path, model, vie=True)
        # for i in range(len(image_paths)):
        #     img = torch.unsqueeze(images[i], dim=0)
        #     output = model(img)[0].transpose(
        #         1, 2).detach().numpy().argmax(axis=0)

            # if args.view:
            #     view(images[i].transpose(0, 2).detach().int().numpy(), output)
            #     pred_path = image_paths[i].replace(
            #         'idd20k_lite', 'preds').replace('leftImg8bit/test/', '').replace('_image.jpg', '_label.png')
            #     os.makedirs(os.path.dirname(
            #         os.path.relpath(pred_path)), exist_ok=True)
            #     img = Image.fromarray(output.astype(np.uint8))
            #     img.save(pred_path)

    else:

        # Code for training on the train split of the dataset.

        image_paths = glob('idd20k_lite/leftImg8bit/train/*/*_image.jpg')
        label_paths = [p.replace('leftImg8bit', 'gtFine').replace(
            '_image.jpg', '_label.png') for p in image_paths]

        images = np.zeros((1403, 227, 320, 3), dtype=np.int)
        labels = np.zeros((1403, 227, 320), dtype=np.int)

        for i in range(1403):
            images[i] = imread(image_paths[i])
            labels[i] = imread(label_paths[i])
            labels[i][labels[i] == 255] = 7

        images = torch.tensor(images, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, images, labels, optimizer, epoch)
            scheduler.step()
            torch.save(model.state_dict(), "seg.pt")


if __name__ == '__main__':
    main()
