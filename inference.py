import torch, numpy, PIL.Image, torchvision.transforms
from semseg_model.models import ModelBuilder, SegmentationModule
from utils.visualize import visualize_result
import os

class SegmentModel():
    def __init__(self) -> None:
        #Download weight
        self.download_weigth()
        #Check device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch='resnet50dilated',
            fc_dim=2048,
            weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
        net_decoder = ModelBuilder.build_decoder(
            arch='ppm_deepsup',
            fc_dim=2048,
            num_class=150,
            weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
            use_softmax=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        self.segmentation_module.eval()
        self.segmentation_module.to(self.device)

    def download_weigth(self):
        if not os.path.exists('ckpt/ade20k-resnet50dilated-ppm_deepsup'):
            os.system('bash download_weight.sh')

    def get_prediction(self, image_path):
        # Load and normalize one image as a singleton tensor batch
        pil_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        ])
        pil_image = PIL.Image.open(image_path).convert('RGB')
        img_original = numpy.array(pil_image)
        img_data = pil_to_tensor(pil_image)
        singleton_batch = {'img_data': img_data[None].to('cpu')}
        output_size = img_data.shape[1:]

        # Run the segmentation at the highest resolution.
        with torch.no_grad():
            scores = self.segmentation_module(singleton_batch, segSize=output_size)
            
        # Get the predicted scores for each pixel
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()
        seg_image = visualize_result(img_original, pred)
        return seg_image