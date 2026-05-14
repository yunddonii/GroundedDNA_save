import torch
import torchvision
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, encode_length, encoder_type="linear", hidden_nodes=None, finetune = False, batch_norm=False, **kwargs):
        super(VGG16, self).__init__()
        self.backbone = torchvision.models.vgg16(pretrained=True)
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:6])
        
        for param in self.backbone.parameters():
            param.requires_grad = finetune
        
        if hidden_nodes is None:
            hidden_nodes = []
                    
        in_dims = [4096] + hidden_nodes
        out_dims = hidden_nodes + [encode_length]

        if encoder_type == "linear":
            encoder_layers = []
            self.n_layers_in_block = 3
            for in_dim, out_dim in zip(in_dims, out_dims):
                encoder_layers.append( nn.Linear(in_dim, out_dim))
                if out_dim != encode_length:
                    encoder_layers.append(nn.ReLU())
                    encoder_layers.append(nn.Dropout()) #

                    
        elif encoder_type == "cafe":
            encoder_layers = [Cafe(4096, in_dims[0]//2, encode_length)]
            self.n_layers_in_block = 1
            
        else:
            raise NotImplementedError(f"{encoder_type} is not defined")
        
                
        if batch_norm:
            encoder_layers.append(nn.BatchNorm1d(encode_length, momentum=0.1))

        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        # for param in self.encoder_layers.parameters():
        #     print(param)

    def forward(self, x, featmap_requires_grad = False):
        feat = self.backbone.features(x)
        
        if featmap_requires_grad:
            feat.requires_grad = True
            
        x = feat.view(x.size(0), -1)
        x = self.backbone.classifier(x)
        
        u = x

        encoder_block_outputs = []
        for i, layer in enumerate(self.encoder_layers):
            u = layer(u)
            
            is_block_output = (i % self.n_layers_in_block) == (self.n_layers_in_block -1)
            
            if is_block_output:
                encoder_block_outputs.append(u)
        
        out = {}
        out['featuremap'] = feat
        out['backbone_last_output'] = x
        out['continuous_code'] = u

        out['encoder_block_outputs'] = encoder_block_outputs

        return out
    
    
    
class ResnetImgNet(nn.Module):
    def __init__(self, encode_length, encoder_type="linear", hidden_nodes=None, finetune = False, batch_norm=False, **kwargs):
        super(ResnetImgNet, self).__init__()
        
        self.backbone = torchvision.models.resnet18(pretrained=True) 

        for param in self.backbone.parameters():
            param.requires_grad = finetune
        
        if hidden_nodes is None:
            hidden_nodes = []
                    
        in_dims = [self.backbone.fc.in_features] + hidden_nodes
        out_dims = hidden_nodes + [encode_length]

        if encoder_type == "linear":
            encoder_layers = []
            self.n_layers_in_block = 3
            for in_dim, out_dim in zip(in_dims, out_dims):
                encoder_layers.append( nn.Linear(in_dim, out_dim))
                if out_dim != encode_length:
                    encoder_layers.append(nn.ReLU())
                    encoder_layers.append(nn.Dropout()) #
            
                    
        elif encoder_type == "cafe":
            encoder_layers = [Cafe(in_dims[0], in_dims[0]//2, encode_length)]
            self.n_layers_in_block = 1            
                    

        else:
            raise NotImplementedError(f"{encoder_type} is not defined")

        del self.backbone.fc
        
        if batch_norm:
            encoder_layers.append(nn.BatchNorm1d(encode_length, momentum=0.1))
            
        self.encoder_layers = nn.Sequential(*encoder_layers) 
        
    def forward(self, x, featmap_requires_grad = False):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feat = self.backbone.layer4(x)

        if featmap_requires_grad:
            feat.requires_grad = True
            
        x = self.backbone.avgpool(feat)
        x = torch.flatten(x, 1)
        
        u = x
        encoder_block_outputs = []
        for i, layer in enumerate(self.encoder_layers):
            u = layer(u)
            
            is_block_output = (i % self.n_layers_in_block) == (self.n_layers_in_block -1)
            
            if is_block_output:
                encoder_block_outputs.append(u)
        
        out = {}
        out['featuremap'] = feat
        out['backbone_last_output'] = x
        out['continuous_code'] = u

        out['encoder_block_outputs'] = encoder_block_outputs


        return out


class ViT(nn.Module):
    def __init__(self, encode_length, hidden_nodes=None, encoder_type = "linear", finetune = False, batch_norm=False, **kwargs) -> None:
        super().__init__()
        # Vision Transfomer Base Model
        self.backbone = torchvision.models.vit_b_16(pretrained=True) # weights="IMAGENET1K_V1"
        del self.backbone.heads # head: Linear(768, n_class)
        
        for param in self.backbone.parameters():
            param.requires_grad = finetune
        
        if hidden_nodes is None:
            hidden_nodes = []
                    
        in_dims = [768] + hidden_nodes
        out_dims = hidden_nodes + [encode_length]


        if encoder_type == "linear":
            encoder_layers = []
            self.n_layers_in_block = 3
            for in_dim, out_dim in zip(in_dims, out_dims):
                encoder_layers.append( nn.Linear(in_dim, out_dim))
                if out_dim != encode_length:
                    encoder_layers.append(nn.ReLU())
                    encoder_layers.append(nn.Dropout()) #
            
                    
        elif encoder_type == "cafe":
            encoder_layers = [Cafe(in_dims[0], in_dims[0]//2, encode_length)]
            self.n_layers_in_block = 1
            
        else:
            raise NotImplementedError(f"{encoder_type} is not defined")
                
        
                
        if batch_norm:
            encoder_layers.append(nn.BatchNorm1d(encode_length, momentum=0.1))

        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        
    
    def forward(self, x, featmap_requires_grad = False):
        
        # Reshape and permute the input tensor
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        feat = self.backbone.encoder(x)

        if featmap_requires_grad:
            raise NotImplementedError

        # Classifier "token" as used by standard language architectures
        feat = feat[:, 0]

        # Continuous Code Generation
        u = feat
        encoder_block_outputs = []
        for i, layer in enumerate(self.encoder_layers):
            u = layer(u)
            
            is_block_output = (i % self.n_layers_in_block) == (self.n_layers_in_block -1)
            
            if is_block_output:
                encoder_block_outputs.append(u)
        
        out = {}
        out['featuremap'] = feat
        out['backbone_last_output'] = feat
        out['continuous_code'] = u

        out['encoder_block_outputs'] = encoder_block_outputs


        return out


class Cafe(nn.Module):
    def __init__(self, input_dim, feature_dim, code_length) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, feature_dim)
        self.fc_2 = nn.Linear(input_dim, feature_dim)
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
        self.hash_layer = nn.Linear(feature_dim, code_length)
        
        dict_fc = self.fc_1.state_dict()
        self.fc_2.load_state_dict(dict_fc)
        
    def forward(self, x):
        
        x1 = self.fc_1(x)
        x2 = self.fc_2(x)

        concept_selector1 = torch.tanh(x1)
        concept_selector2 = torch.tanh(x2)

        alpha = concept_selector2 * concept_selector1

        x1 = x1 * alpha
        x2 = x2 * alpha
        x = torch.cat((x1, x2), dim=1)
        
        x = self.fc(x)
        
        # direct_feature = x
        
        x = torch.tanh(x)

        hash_codes = self.hash_layer(x)
        hash_codes = torch.tanh(hash_codes)
        
        return hash_codes #, direct_feature
        
    
    


backbone_dict = {"VGG16": VGG16, "ResNet": ResnetImgNet, "ViT": ViT} 





