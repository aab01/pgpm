# NOT CURRENTLY USED, BUT WILL HAVE TO BE....
import os
import numpy as np
import torch
import torch.nn as nn
from .model import UNet1

class Unet1Fracture:
    
    def __init__(self, params, model_dir='..', e=5):
        
        self.model_path = os.path.join(model_dir,params['path'])
        self.patch_w = params['patch_width']
        self.patch_h = params['patch_height']
        self.stride_w = params['stride_horizontal']
        self.stride_h = params['stride_vertical']
        self.n_channels = params['n_channels']
        self.n_classes = params['n_classes']
        self.e = e
        
    def initialize(self):
        
        self.device = ("cuda" if torch.cuda.is_available() else "cpu" )
        self.net = UNet1(n_channels=self.n_channels, n_classes=self.n_classes)
        try:
            self.net.load_state_dict(torch.load(self.model_path))
        except RuntimeError:
            self.net = nn.DataParallel(self.net)
            self.net.load_state_dict(torch.load(self.model_path))
        self.net.to(self.device)
        
        self.net.eval()
        
    
    def predict_proba(self, image):
        
        assert image.dtype.type == np.uint8, 'image format should be uint8 (PIL Image)' 
        
        image = image/255   # normalize between 0-1 - consired standard scaling instead
        img_expand = self.expand_image(image)
        
        img_prob = np.zeros(img_expand.shape[:2]+(self.n_classes,))
        # img_sum is used to count pixel oberlapping during for sliding patch prediction
        img_sum = np.zeros(img_expand.shape[:2])


        for i_h in range(self.n_h+1):

            start_h = i_h*self.stride_h
            end_h = start_h+self.patch_h

            for i_w in range(self.n_w+1):

                start_w = i_w*self.stride_w
                end_w = start_w+self.patch_w
                
                patch = img_expand[start_h:end_h,start_w:end_w]
                # mirror patch contours before prediction - U-Net overlap-tile strategy
                patch = input_filled_mirroring(patch, e=self.e)

                if len(patch.shape)==2:
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
                if len(patch.shape)==3:
                    patch_tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float()
                
                pred = self.net(patch_tensor.to(self.device))
                
                prob_predict = torch.sigmoid(pred).squeeze().detach().cpu().numpy()
                prob_predict = np.transpose(prob_predict, (1,2,0))
                
                # crop patch contours after prediction
                img_prob[start_h:end_h,start_w:end_w,:] += prob_predict[self.e:-self.e,self.e:-self.e,:]
                img_sum[start_h:end_h,start_w:end_w] += 1
                
        img_sum = np.dstack([img_sum]*self.n_classes)
        avg_prob = img_prob/img_sum
        avg_prob = avg_prob[:self.img_h,:self.img_w,:] # crop to original image size
        
        return avg_prob
    
    def predict_image(self, image):
        
        prob_predict = self.predict_proba(image)
        mask = np.argmax(prob_predict, axis=2)
        
        return mask
