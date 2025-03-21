import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
print(sys.path)

def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv


# function to crop an image
def crop_img(input_tensor, target_tensor):
    target_size = target_tensor.size()[2]
    input_size = input_tensor.size()[2]
    delta = input_size - target_size
    delta = delta // 2
    return input_tensor[:,:,delta:input_size-delta,delta:input_size-delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv1_encoder = double_conv(1, 64)
        self.conv2_encoder = double_conv(64, 128)
        self.conv3_encoder = double_conv(128, 256)
        self.conv4_encoder = double_conv(256, 512)
        self.conv5_encoder = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv1_decoder = double_conv(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv2_decoder = double_conv(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv3_decoder = double_conv(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv4_decoder = double_conv(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=2) # Change the number

    def forward(self, image):

        # Checking the size of input
        print("The shape of input image is", image.shape)

        ########## encoder part of the UNet ###########

        # first convolution block
        x1 = self.conv1_encoder(image) # x1 should be copied for skip connection in the decoder layer
        print("After first conv2d layer:")
        print(f"x1", x1.shape)
        x1_max = self.maxpool(x1)
        print("After first maxpooling layer: ")
        print(f"x1_max", x1_max.shape)

        # second convolution block
        x2 = self.conv2_encoder(x1_max) # x2 should be copied for skip connection in the decoder layer
        print("After second conv2d layer:")
        print(f"x2", x2.shape)
        x2_max = self.maxpool(x2)
        print("After second maxpooling layer: ")
        print(f"x2_max", x2_max.shape)

        # third convolution block
        x3 = self.conv3_encoder(x2_max)  # x3 should be copied for skip connection in the decoder layer
        print("After third conv2d layer:")
        print(f"x3", x3.shape)
        x3_max = self.maxpool(x3)
        print("After third maxpooling layer: ")
        print(f"x3_max", x3_max.shape)

        # fourth convolution block
        x4 = self.conv4_encoder(x3_max)  # x4 should be copied for skip connection in the decoder layer
        print("After fourth conv2d layer:")
        print(f"x4", x4.shape)
        x4_max = self.maxpool(x4)
        print("After fourth maxpooling layer: ")
        print(f"x4_2", x4_max.shape)

        # fifth convolution block or the one without maxpooling layer
        x5 = self.conv5_encoder(x4_max)
        print("After fifth conv2d layer:")
        print(f"x5", x5.shape)

        # return x5

        ######## decoder part of the UNet #########
        print("########The decoder part begins#######")
        # first decoder block
        x_decoder_1 = self.up_transpose_1(x5)
        print("After applying the transposed convolution on x5")
        print(f"x_decoder_1", x_decoder_1.shape)
        cropped_x4 = crop_img(x4, x_decoder_1)
        print(f"The shape of cropped_x4 is", cropped_x4.shape)
        concat_x_decoder_1 = self.conv1_decoder(torch.cat([cropped_x4,x_decoder_1], dim=1)) # concatenation is performed followed by double convolution
        # temp_concat = torch.cat([cropped_x4, x_decoder_1], dim=1)
        # print("The shape of concat_x_decoder_1 before double conv is", temp_concat.shape)
        print("The shape of concat_x_decoder_1 after double conv is", concat_x_decoder_1.shape)

        # second decoder block
        x_decoder_2 = self.up_transpose_2(concat_x_decoder_1)
        print(f"The shape of x_decoder_2 is", x_decoder_2.shape)
        cropped_x3 = crop_img(x3, x_decoder_2)
        print(f"The shape of cropped_x3 is", cropped_x3.shape)
        concat_x_decoder_2 = self.conv2_decoder(torch.cat([cropped_x3, x_decoder_2], dim=1))
        print("The shape of concat_x_decoder_2 after double conv is", concat_x_decoder_2.shape)

        # third decoder block
        x_decoder_3 = self.up_transpose_3(concat_x_decoder_2)
        print(f"The shape of x_decoder_3 is", x_decoder_3.shape)
        cropped_x2 = crop_img(x2, x_decoder_3)
        print(f"The shape of cropped_x4 is", cropped_x2.shape)
        concat_x_decoder_3 = self.conv3_decoder(torch.cat([cropped_x2, x_decoder_3], dim=1))
        print("The shape of concat_x_decoder_3 after double conv is", concat_x_decoder_3.shape)

        # fourth decoder block
        x_decoder_4 = self.up_transpose_4(concat_x_decoder_3)
        print(f"The shape of x_decoder_4 is", x_decoder_4.shape)
        cropped_x1 = crop_img(x1, x_decoder_4)
        print(f"The shape of cropped_x1 is", cropped_x1.shape)
        concat_x_decoder_4 = self.conv4_decoder(torch.cat([cropped_x1, x_decoder_4], dim=1))
        print("The shape of concat_x_decoder_4 after double conv is", concat_x_decoder_4.shape)



if __name__ == "__main__":
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    output = model(image)
    #print(output.shape)
