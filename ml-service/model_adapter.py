import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEG_SIZE = (384, 384)


def decoder_block(x, skip, filters, name):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name=name + "_up",
    )(x)

    x = layers.Concatenate(
        name=name + "_concat"
    )([x, skip])

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=name + "_conv1",
    )(x)

    x = layers.BatchNormalization(
        name=name + "_bn1"
    )(x)

    x = layers.Activation(
        "relu",
        name=name + "_relu1"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=name + "_conv2",
    )(x)

    x = layers.BatchNormalization(
        name=name + "_bn2"
    )(x)

    x = layers.Activation(
        "relu",
        name=name + "_relu2"
    )(x)

    return x


def build_unet():
    """
    Reconstruct the exact lesion-segmentation architecture used
    for the final IDRiD model.

    Input:
        (384, 384, 3), float image in the 0-255 range.

    Output:
        (384, 384, 1), sigmoid DR-lesion probability map.

    The saved .weights.h5 file must be loaded after this model
    is constructed.
    """

    base = keras.applications.MobileNetV2(
        input_shape=(*SEG_SIZE, 3),
        include_top=False,

        # We already possess the final trained encoder weights
        # inside idrid_lesion_unet_final.weights.h5.
        # No ImageNet download is required for deployment.
        weights=None,
    )

    skip_names = [
        "block_1_expand_relu",
        "block_3_expand_relu",
        "block_6_expand_relu",
        "block_13_expand_relu",
        "block_16_project",
    ]

    down_stack = keras.Model(
        inputs=base.input,
        outputs=[
            base.get_layer(name).output
            for name in skip_names
        ],
        name="mobilenetv2_encoder",
    )

    # Trainability does not affect inference, but keeping the
    # encoder frozen is appropriate for the deployed model.
    down_stack.trainable = False

    inputs = keras.Input(
        shape=(*SEG_SIZE, 3),
        name="seg_fundus",
    )

    # This preprocessing layer was part of the trained model.
    # Website code must therefore pass pixels in the 0-255 range.
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_preprocess",
    )(inputs)

    features = down_stack(
        x,
        training=False,
    )

    s1, s2, s3, s4, bottleneck = features

    x = decoder_block(
        bottleneck,
        s4,
        256,
        "dec4",
    )

    x = decoder_block(
        x,
        s3,
        128,
        "dec3",
    )

    x = decoder_block(
        x,
        s2,
        64,
        "dec2",
    )

    x = decoder_block(
        x,
        s1,
        32,
        "dec1",
    )

    x = layers.Conv2DTranspose(
        24,
        3,
        strides=2,
        padding="same",
        activation="relu",
        name="final_up",
    )(x)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="lesion_probability",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="IDRiD_Lesion_UNet",
    )

    return model