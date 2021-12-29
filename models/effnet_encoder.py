from tensorflow.keras import Model
from tensorflow.keras.applications import efficientnet

class EffnetEncoder:
    ''' EfficientNet based encoder '''
    def __init__(self, base_name: str, input_shape: tuple[int]) -> None:
        self.encoder_name = f'Effnet{base_name}Encoder'
        self.input_shape = input_shape
        self.layer_names = [
            'block2a_expand_activation',
            'block3a_expand_activation',
            'block4a_expand_activation',
            'block6a_expand_activation',
            'block7a_project_conv'
        ]
        
        # Download our pretrained base
        self.base_model = self.load_pretrained_base(base_name)
    
    def load_pretrained_base(self, base_name: str) -> Model:
        model_mapping = {
            'B0' : efficientnet.EfficientNetB0(include_top=False, input_shape=self.input_shape, weights='imagenet'),
            'B1' : efficientnet.EfficientNetB1(include_top=False, input_shape=self.input_shape, weights='imagenet')
        }
        
        return model_mapping[base_name]
        
    def build_encoder(self, trainable: bool = False) -> Model:
        base_model_outputs = [self.base_model.get_layer(name).output for name in self.layer_names]
        encoder = Model(inputs=self.base_model.input, outputs=base_model_outputs, name=self.encoder_name)
        encoder.trainable = trainable
        
        return encoder