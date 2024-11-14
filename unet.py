import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model



def conv_block(input, filters):
    x = Conv2D(filters, 3, activation='relu', padding='same')(input)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    return x


def build_multimodal_unet(input_shape):
    # Wejście dla danych optycznych (Sentinel-2)
    optical_input = Input(shape=input_shape, name="optical_input")
    o1 = conv_block(optical_input, 64)
    p1 = MaxPooling2D((2, 2))(o1)
    o2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(o2)
    o3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(o3)
    o4 = conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(o4)
    o5 = conv_block(p4, 1024)

    # Wejście dla danych radarowych (Sentinel-1)
    radar_input = Input(shape=input_shape, name="radar_input")
    r1 = conv_block(radar_input, 64)
    p1_r = MaxPooling2D((2, 2))(r1)
    r2 = conv_block(p1_r, 128)
    p2_r = MaxPooling2D((2, 2))(r2)
    r3 = conv_block(p2_r, 256)
    p3_r = MaxPooling2D((2, 2))(r3)
    r4 = conv_block(p3_r, 512)
    p4_r = MaxPooling2D((2, 2))(r4)
    r5 = conv_block(p4_r, 1024)

    # Połączenie obu ścieżek danych
    fusion = concatenate([o5, r5])

    # Dekoder
    u4 = UpSampling2D((2, 2))(fusion)
    u4 = concatenate([u4, o4, r4])
    u4 = conv_block(u4, 512)

    u3 = UpSampling2D((2, 2))(u4)
    u3 = concatenate([u3, o3, r3])
    u3 = conv_block(u3, 256)

    u2 = UpSampling2D((2, 2))(u3)
    u2 = concatenate([u2, o2, r2])
    u2 = conv_block(u2, 128)

    u1 = UpSampling2D((2, 2))(u2)
    u1 = concatenate([u1, o1, r1])
    u1 = conv_block(u1, 64)

    # Wyjście
    output = Conv2D(1, (1, 1), activation="sigmoid")(u1)

    # Definiowanie modelu
    model = Model(inputs=[optical_input, radar_input], outputs=output)
    return model



input_shape = (512, 512, 3)  # 3 kanały dla danych Sentinel-2 i Sentinel-1

model = build_multimodal_unet(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


