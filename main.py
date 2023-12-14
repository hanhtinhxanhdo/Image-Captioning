from models import EncoderCNN_Resnet50, EncoderCNN_VGG19, DecoderRNN
from Data_loader import *

import os, sys
from pycocotools.coco import COCO
import urllib, zipfile, subprocess

import nltk
nltk.download('punkt')
import torch 
from torchvision import transforms
import torch.utils.data as data
import torch.nn as nn

import matplotlib.pyplot as plt 
import skimage.io as io 
import numpy as np, math, pickle

from collections import Counter

########################################################
print('*' * 100)
if torch.cuda.is_available():
    print("GPU is available (Torch).")
    _device = torch.device("cuda")
    print("Current GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. Using CPU.")
    _device = torch.device("cpu")

print('*' * 100, '\n')
########################################################

# # Specify the directory name
# directory_name = 'opt'

# # Create the directory if it doesn't exist
# os.makedirs(directory_name, exist_ok=True)

# # Change the current working directory to the newly created directory
# os.chdir(directory_name)

# # Specify the repository URL
# repository_url = 'https://github.com/cocodataset/cocoapi.git'

# # Construct the git clone command
# git_clone_command = ['git', 'clone', repository_url]

# # Run the git clone command using subprocess
# try:
#     subprocess.run(git_clone_command, check=True)
#     print(f"Repository cloned successfully to {os.path.abspath(directory_name)}")
# except subprocess.CalledProcessError as e:
#     print(f"Error cloning repository: {e}")


# # Change the current working directory
# os.chdir('opt/cocoapi')

# # Download the annotation
# annotations_trainval2014 = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
# image_info_test2014 = 'http://images.cocodataset.org/annotations/image_info_test2014.zip'

# urllib.request.urlretrieve(annotations_trainval2014, filename='annotations_trainval2014.zip')
# urllib.request.urlretrieve(image_info_test2014, filename='image_info_test2014.zip')


# # os.chdir('opt/cocoapi')
# # Extract 'annotations_trainval2014.zip'
# with zipfile.ZipFile('annotations_trainval2014.zip', 'r') as zip_ref:
#     zip_ref.extractall('opt/cocoapi')

# # Remove 'annotations_trainval2014.zip'
# try:
#     os.remove('annotations_trainval2014.zip')
#     print('annotations_trainval2014.zip removed')
# except Exception as e:
#     print(f"Error removing annotations_trainval2014.zip: {e}")

# # Extract 'image_info_test2014.zip'
# with zipfile.ZipFile('image_info_test2014.zip', 'r') as zip_ref:
#     zip_ref.extractall('opt/cocoapi')

# # Remove 'image_info_test2014.zip'
# try:
#     os.remove('image_info_test2014.zip')
#     print('image_info_test2014.zip removed')
# except Exception as e:
#     print(f"Error removing image_info_test2014.zip: {e}")


os.chdir('opt/cocoapi/annotations')
# initialize COCO API for instance annotations
dataType = 'val2014'
instances_annFile = 'instances_{}.json'.format(dataType)
print(instances_annFile)
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = 'captions_{}.json'.format(dataType)
coco_caps = COCO(captions_annFile)

# get image ids 
ids = list(coco.anns.keys())

#Pick a random annotation id and display img of that annotation  :
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']
print(url)
I = io.imread(url)
plt.imshow(I)

# Display captions for that annotation id :
ann_ids = coco_caps.getAnnIds(img_id)
print('Number of annotations i.e captions for the image: ', ann_ids)
print()
anns = coco_caps.loadAnns(ann_ids)
coco_caps.showAnns(anns)



# Get the current working directory
current_directory = os.getcwd()
print(f"\nCurrent directory: {current_directory}")

# Move to the parent directory
os.chdir('..')

# Get the updated current working directory
updated_directory = os.getcwd()
print(f"Updated directory: {updated_directory}")


# os.chdir('opt/cocoapi')

# train2014 = 'http://images.cocodataset.org/zips/train2014.zip'
# test2014 = 'http://images.cocodataset.org/zips/test2014.zip'
# val2014 = 'http://images.cocodataset.org/zips/val2014.zip'

# urllib.request.urlretrieve(train2014, 'train2014')
# urllib.request.urlretrieve(test2014, 'test2014')
# urllib.request.urlretrieve(val2014, 'val2014')


# with zipfile.ZipFile('train2014', 'r' ) as zip_ref:
#   zip_ref.extractall('images')

# try:
#   os.remove( 'train2014')
#   print('zip removed')
# except:
#   None


# with zipfile.ZipFile('test2014', 'r' ) as zip_ref:
#   zip_ref.extractall('images')

# try:
#   os.remove( 'test2014' )
#   print('zip removed')
# except:
#   None

# with zipfile.ZipFile('val2014', 'r' ) as zip_ref:
#     zip_ref.extractall('images')

# try:
#     os.remove('val2014')
#     print('zip removed')
# except:
#     None

print('\n', '=' * 100)
print('\tStep 1: Explore the DataLoader\n')

current_directory = os.getcwd()
print(f"\nCurrent directory: {current_directory}")

# Move to the parent directory
os.chdir('..')
os.chdir('..')

# Get the updated current working directory
updated_directory = os.getcwd()
print(f"Updated directory: {updated_directory}")

# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Set the minimum word count threshold.
vocab_threshold = 8

# Specify the batch size.
batch_size = 128

# Obtain the data loader.
data_loader_train = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False,
                         cocoapi_loc = 'opt')



# Exploring the dataloader now :

sample_caption = 'A person doing a trick xxxx on a rail while riding a skateboard.'
sample_tokens = nltk.tokenize.word_tokenize(sample_caption.lower() )

sample_caption = []
start_word  = data_loader_train.dataset.vocab.start_word
end_word = data_loader_train.dataset.vocab.end_word
sample_tokens.insert(0, start_word)
sample_tokens.append(end_word)
sample_caption.extend([data_loader_train.dataset.vocab(token) for token in sample_tokens])


sample_caption = torch.Tensor(sample_caption).long()
print('Find Below the Sample tokens and the idx values of those tokens in word2idx', '\n')
print(sample_tokens) 
print(sample_caption)

print('Find index values for words below \n')
print('Start idx {}, End idx {}, unknown idx {}'.format(0, 1, 2))

# Lets check word2idx in vocb 
print('First few vocab', dict(list(data_loader_train.dataset.vocab.word2idx.items())[:10]))
# Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary: ', len(data_loader_train.dataset.vocab))


print('\n', '=' * 100)
print('\tStep 2: Use the Data Loader to Obtain Batches\n')

counter = Counter(data_loader_train.dataset.caption_lengths)
lengths = sorted( counter.items(), key = lambda pair: pair[1], reverse=True )
for val, count in lengths:
    print('value %2d count %5d' % (val, count))
    if count < 10000: 
        break

# Randomly sample a caption length, and sample indices with that length.
indices = data_loader_train.dataset.get_train_indices()
print('Sample Indices:', indices )

# Create and assign a batch sampler to retrieve a batch with the sampled indices.
sampler = data.sampler.SubsetRandomSampler(indices)
data_loader_train.batch_sampler.sampler = sampler 

# obtain images, caption :
images , captions = next(iter(data_loader_train))
print(images.shape, captions.shape)


print('\n', '=' * 100)
print('\tStep 3: Experiment with the CNN Encoder\n')

# specify dim of image embedding
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 256 
encoder = EncoderCNN_Resnet50(embed_size)
encoder.to(device)
images =  images.to(device) # images from step2 
features = encoder(images)

print(type(features), features.shape, images.shape)
assert(type(features) == torch.Tensor), 'Encoder output should be pytorch tensor'
assert(features.shape[0] == batch_size) & (features.shape[1] == embed_size), "The shape of the encoder output is incorrect."


print('\n', '=' * 100)
print('\tStep 4: Implement the RNN Decoder\n')

log_file = "training_log_resnet50.txt" # name of file with saved training loss and perplexity
embed_size = 256
hidden_size = 512
num_layers = 1 
num_epochs = 1
print_every = 50
save_every = 1 
vocab_size = len(data_loader_train.dataset.vocab)
total_step = math.ceil(len(data_loader_train.dataset.caption_lengths) / data_loader_train.batch_sampler.batch_size)

# Initializing the encoder and decoder
encoder = EncoderCNN_Resnet50(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Defining the loss function
criterion = (nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss())

lr = 0.001
all_params = list(decoder.parameters()) + list(encoder.embed.parameters())
optimizer = torch.optim.Adam(params=all_params, lr = lr)


# device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = 'checkpoint_resnet50'
os.makedirs(model_save_path, exist_ok=True)


# Open the training log file.
f = open(log_file, "w")

for epoch in range(1, num_epochs + 1):
    for i_step in range(1, total_step + 1):
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader_train.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader_train.batch_sampler.sampler = new_sampler

        # Obtain the batch.
        images, captions = next(iter(data_loader_train))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)

        # Zero the gradients.
        encoder.zero_grad()
        decoder.zero_grad()

        # Passing the inputs through the CNN-RNN model
        features = encoder(images)
        outputs = decoder(features, captions)

        # Calculating the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        #         # Uncomment to debug
        #         print(outputs.shape, captions.shape)
        #         # torch.Size([bs, cap_len, vocab_size]) torch.Size([bs, cap_len])

        #         print(outputs.view(-1, vocab_size).shape, captions.view(-1).shape)
        #         # torch.Size([bs*cap_len, vocab_size]) torch.Size([bs*cap_len])

        # Backwarding pass
        loss.backward()

        # Updating the parameters in the optimizer
        optimizer.step()

        # Getting training statistics
        stats = (
            f"Epoch [{epoch}/{num_epochs}], Step [{i_step}/{total_step}], "
            f"Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):.4f}"
        )

        # Print training statistics to file.
        f.write(stats + "\n")
        f.flush()

        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print("\r" + stats)

        # Save the weights.
        if epoch % save_every == 0:
            torch.save(encoder.state_dict(), os.path.join(model_save_path, 'encoderdata_{}.pkl'.format(epoch)))
            torch.save(decoder.state_dict(), os.path.join(model_save_path, 'decoderdata_{}.pkl'.format(epoch)))

# Close the training log file.
f.close()
    