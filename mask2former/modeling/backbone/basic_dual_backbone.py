from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec, build_backbone
from torch import nn
from detectron2.config import configurable, CfgNode
from typing import Optional, Union, Callable, List, Dict
from torch import Tensor

@BACKBONE_REGISTRY.register()
class BasicDualBackbone(Backbone):
    @configurable
    def __init__(self,
                 backbone1: Backbone,
                 backbone2: Backbone,
                 combination: Optional[Union[str, Callable]] = None,
                 modalities: Optional[List[str]] = ['image', 'depth']):
        """ Basic class for dual backbones. 

        This is a wrapper around Detectron2 backbone and supports using multiple backbones for multiple modalities.\
        It simply combines the outputs of the two backbones with the summation, concatenation, etc.

        Args:
            backbone1 (Backbone): The backbone for the first modality in ``modalities``
            backbone2 (Backbone): The backbone for the second modality in ``modalities``
            combination (Union[str, Callable]): The combination method, e.g. summation. If it's None, then the output of\
                both backbones will be returned. Should take two Dict[str, Tensor] and return one Dict[str, Tensor].\
                Defaults to None
            modalities: List of modalities. The first modality will be passed to the first backbone and so on.\
                Defaults to ['image', 'depth']
        """        
        super().__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        # To complete
        self.combination = combination if isinstance(combination, Callable) else\
            _get_combination_methods(combination)
        self.modalities = modalities
        
        self._out_features = self.backbone1._out_features
        self._out_feature_channels = self.backbone1._out_feature_channels
        self._out_feature_strides = self.backbone1._out_feature_strides
        
    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            Dict[str, Tensor]: dict[str->Tensor]: names and the corresponding features.
        """        
        out1 = self.backbone1(x[self.modalities[0]])
        out2 = self.backbone2(x[self.modalities[1]])
        assert list(out1) == list(out2), "Output features of the two backbone should have the same names"
        return self.combination(out1, out2)
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        """ Create backbone from the config.

        If cfg.MODEL.SAME_BACKBONE is set to True, then the two backbones will have the same architecture\
            , otherwise, it assumes that the configuration for each backbone is in the cfg with postfix __{1,2}.

        Args:
            cfg (_type_): _description_
            input_shape (_type_): _description_
        """    
        assert cfg.MODEL.BACKBONE.BASE, "You must provide a base backbone in ``cfg.model.BACKBONE.BASE``"
        same_backbone = not isinstance(cfg.MODEL.BACKBONE.BASE, list)
        
        backbones = []
        for i, modality in enumerate(cfg.MODEL.INPUT_MODALITIES):
            cfg_ = cfg.clone()
            cfg_.defrost()
            if same_backbone:
                cfg_.MODEL.BACKBONE.NAME = cfg.MODEL.BACKBONE.BASE
            else:
                cfg_.MODEL.BACKBONE.NAME = cfg.MODEL.BACKBONE.BASE[i]
            backbone_cfg_ = _modality_specific_config(cfg_, modality) 
            cfg_.merge_from_list(backbone_cfg_)
            cfg_.freeze()
            backbones.append(build_backbone(cfg_, input_shape))
        
        return {
            'backbone1': backbones[0],
            'backbone2': backbones[1],
            'combination': cfg.MODEL.COMBINATION,
            'modalities': cfg.MODEL.INPUT_MODALITIES
        }
        
# TODO: add documentation
def _get_combination_methods(method: str):
    if method == 'sum':
        return lambda out1, out2: {k: out1.get(k) + out2.get(k) for k in out1.keys()}
    
    
# TODO: add documentation 
def _modality_specific_config(old_dict, name_to_remove):
    listed_config = _nested_dict_to_list(old_dict)
    listed_new_configs = []
    for key, value in listed_config:
        if name_to_remove in key:
            listed_new_configs.append(key.replace('_' + name_to_remove, ''))
            listed_new_configs.append(value)
    return listed_new_configs

def _nested_dict_to_list(d, keys=[]):
    result = []
    for k, v in d.items():
        if isinstance(v, dict):
            result.extend(_nested_dict_to_list(v, keys + [k]))
        else:
            result.append(('.'.join(keys + [k]), v))
    return result


if __name__ == '__main__':
    import detectron2
    cfg = detectron2.config.get_cfg()
    print(cfg.keys())