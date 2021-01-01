import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D, LeakyReLU, Conv2D
from tensorflow.keras import Sequential, Model, Input


class PA(Model):
    def __init__(self, n_filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = Conv2D(n_filters, (1, 1), strides=(1, 1), activation='sigmoid')
    
    def __call__(self, X, training=True):
        return self.conv(X) * X
        

class SCPA(Model):
    def __init__(self, n_filters, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inp_conv_br_a = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False)
        # top sub-branch
        self.conv1_br_a = Conv2D(n_filters, (1, 1), strides=(1, 1), activation='sigmoid', use_bias=False)

        # bottom sub-branch
        self.conv2_br_a = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False)
        
        self.conv3_br_a = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False)
        
        self.inp_conv_br_b = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False)
        self.conv2_br_b = Conv2D(n_filters, (3, 3), strides=(1, 1), dilation_rate=1, use_bias=False)

        self.outp_conv = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False)
        
        self.lrelu = LeakyReLU(0.2)
        self.one_padd = ZeroPadding2D(padding=1)
        
        
    def __call__(self, X, training=True):
        br_a_inputs = self.lrelu(self.inp_conv_br_a(X))
        
        # PA-Conv
        br_a_outputs = self.lrelu(self.conv3_br_a(self.conv1_br_a(br_a_inputs) * self.conv2_br_a(br_a_inputs)))
        
        br_b_inputs = self.lrelu(self.inp_conv_br_b(X))
        
        br_b_outputs = self.conv2_br_b(self.one_padd(br_b_inputs))

        return self.outp_conv(tf.concat((br_a_outputs, br_b_outputs), axis=3)) + X
    
    def call(self, inputs, training=True):
        return self(inputs, training=training)


class UPA():
    def __init__(self, n_filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # try using transposed convolution if this doesn't work
        self.upsampling = UpSampling2D(size=(2, 2), interpolation='nearest')
        self.conv1 = Conv2D(n_filters, (3, 3), strides=(1, 1))
        self.pix_att = PA(n_filters)
        self.conv2 = Conv2D(n_filters, (3, 3), strides=(1, 1))
        
        self.lrelu = LeakyReLU(0.2)
        self.one_padd = ZeroPadding2D(padding=1)
        
    def __call__(self, X, training=True):
        outputs = self.upsampling(X)
        outputs = self.lrelu(self.one_padd(self.conv1(outputs)))
        outputs = self.lrelu(self.pix_att(outputs))
        outputs = self.lrelu(self.one_padd(self.conv2(outputs)))
        return self.lrelu(self.conv2(self.one_padd(outputs)))
        
    def call(self, inputs, training=True):
        return self(inputs, training=training)
    
    
class PixelAttentionSRNetwork(Model):
    def __init__(self, input_shape, feat_extr_n_filters, upsamp_n_filters, n_blocks, scale=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inp_conv = Conv2D(feat_extr_n_filters, (3, 3), strides=(1, 1), input_shape=input_shape)
        
        self.scpa_trunk = Sequential(
            (Input(shape=(*input_shape[1:-1], feat_extr_n_filters)),) + 
             tuple(SCPA(feat_extr_n_filters) for _ in range(n_blocks))
        )

        self.trunk_conv = Conv2D(feat_extr_n_filters, (3, 3), strides=(1, 1))
        
        self.scale_factor = scale
        self.upsampling_2x = UPA(upsamp_n_filters)
        
        if scale == 4:
            self.upsampling_4x = UPA(upsamp_n_filters)
        
        self.outp_conv = Conv2D(3, (3, 3), strides=(1, 1))
        
        self.one_padd = ZeroPadding2D(padding=1)
    
    def __call__(self, X, training=True):
        outputs = self.inp_conv(self.one_padd(X))
        
        outputs += self.trunk_conv(self.one_padd(self.scpa_trunk(outputs)))

        outputs = self.upsampling_2x(outputs)
        
        if self.scale_factor == 4:
            outputs = self.upsampling_4x(outputs)
            
        outputs = self.outp_conv(self.one_padd(outputs))

        return outputs + tf.image.resize(X, tuple(dim * self.scale_factor for dim in X.shape[1:3]), method='bilinear')
        
    def call(self, inputs, training=True):
        return self(inputs, training=training)
