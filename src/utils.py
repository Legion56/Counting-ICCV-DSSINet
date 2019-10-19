def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    if input_length is None:
        return None
    assert padding == "same"
    output_length = input_length
    return (output_length + stride - 1) // stride

def same_padding_length(input_length, filter_size, stride, dilation=1):
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    output_length = (input_length + stride - 1) // stride
    pad_length = max(0, (output_length - 1) * stride + dilated_filter_size - input_length)
    return pad_length

def compute_output_shape(input_shape, filters, kernel_size, padding, strides, dilation):
    space = input_shape[1:]
    new_space = []
    for i in range(len(space)):
        new_dim = conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=padding,
            stride=strides[i],
            dilation=dilation[i])
        new_space.append(new_dim)
    return (filters,) + tuple(new_space)

def compute_same_padding2d(input_shape, kernel_size, strides, dilation):
    space = input_shape[2:]
    assert len(space) == 2, "{}".format(space)
    new_space = []
    new_input = []
    for i in range(len(space)):
        pad_length = same_padding_length(
            space[i],
            kernel_size[i],
            stride=strides[i],
            dilation=dilation[i])
        new_space.append(pad_length)
        new_input.append(pad_length % 2)
    return tuple(new_space), tuple(new_input)
