import argparse
from pathlib import Path
import warnings
import math
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import net
from function import coral
from function import random_projection_style_transfer
import matplotlib.pyplot as plt
import time

# Test command example
# python test.py --content input/content/5.jpg --style input/style/178.jpg --output lzz_1101


# Set compression ratio and number of groups here
compress_ratio = 1
group_nums = 1


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    # Timing for encoding (VGG feature extraction)
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        encode_start = torch.cuda.Event(enable_timing=True)
        encode_end = torch.cuda.Event(enable_timing=True)
        encode_start.record()
    else:
        encode_start = time.time()  # Start CPU timing

    # Extract content and style features (encoding process)
    content_f = vgg(content)
    style_f = vgg(style)
    
    # End timing and calculate time consumption
    if use_cuda:
        encode_end.record()
        torch.cuda.synchronize()  # Synchronize GPU
        encode_time = encode_start.elapsed_time(encode_end)  # Unit: milliseconds
    else:
        encode_end = time.time()
        encode_time = (encode_end - encode_start) * 1000  # Convert to milliseconds
    print(f"Encoding module (VGG feature extraction) time consumption: {encode_time:.2f} ms")


    # A = content_f.cpu()
    # # print(A.size())
    # A = A.reshape(-1, A.shape[-1])
    # print(A.size())
    # plt.hist(A)  # Using 50 bins
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Tensor A')
    # plt.grid(True)
    # plt.show()
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        # Replace with random_projection_style_transfer
        base_feat = random_projection_style_transfer(content_f, style_f, compress_ratio, group_nums)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        # Replace with random_projection_style_transfer
        feat = random_projection_style_transfer(content_f, style_f, compress_ratio, group_nums)

    feat = feat * alpha + content_f * (1 - alpha)

    # Calculate decoding time
    if use_cuda:
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        decode_start.record()
    else:
        decode_start = time.time()

    output = decoder(feat)

    if use_cuda:
        decode_end.record()
        torch.cuda.synchronize()
        decode_time = decode_start.elapsed_time(decode_end)
    else:
        decode_time = (time.time() - decode_start) * 1000
    print(f"[Decoding module] Time consumption: {decode_time:.2f} ms")
    return output


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('11111111111111111111111111111111111111111', device)

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))).unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            # Total timing for main process
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()

            output = style_transfer(vgg, decoder, content, style, args.alpha, interpolation_weights)
            
            total_end.record()
            torch.cuda.synchronize()
            total_time = total_start.elapsed_time(total_end)
            print(f"Total main process time consumption: {total_time:.2f} ms")

        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha)
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))