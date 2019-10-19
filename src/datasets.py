import socket


hostname = socket.gethostname()
SHANG_PATH = ''
UCFEC_PATH = ''
WORLD_PATH = ''
TRANCOS_PATH = ''
UCF_PATH = ''
UCSD_PATH = ''

datasets = {
    'shanghaiA': {
        "density_method": "adaptive",
        "density_method_config": {'downsize':32},

        "train_image_path": SHANG_PATH + '/part_A_final/train_data/images',
        "train_label_path": SHANG_PATH + '/part_A_final/train_data/ground_truth',
        "test_image_path": SHANG_PATH + '/part_A_final/test_data/images',
        "test_label_path": SHANG_PATH + '/part_A_final/test_data/ground_truth',
        "train_val_split": (lambda x:x, lambda x:x[:29]),
        "annReadFunc": lambda x: x['image_info'][0][0][0][0][0],
        "mean_std": [96.3414, 66.8793],
        "annReadFuncTest": None
    },
    'shanghaiB': {
        "density_method": "adaptive",
        "density_method_config": {'downsize':32},

        "train_image_path": SHANG_PATH + '/part_B_final/train_data/images/',
        "train_label_path": SHANG_PATH + '/part_B_final/train_data/ground_truth',
        "test_image_path": SHANG_PATH + '/part_B_final/test_data/images/',
        "test_label_path": SHANG_PATH + '/part_B_final/test_data/ground_truth',
        "train_val_split": (lambda x:x, lambda x:x[:29]),
        "annReadFunc": lambda x: x['image_info'][0][0][0][0][0],
        "mean_std": [96.3414, 66.8793],
        "annReadFuncTest": None
    },

    ### pre-crop high resolution images to normal size patches
    'UCF_ECCV_Crop': {
        "density_method": "adaptive",
        "density_method_config": {'downsize':32},

        "train_image_path": UCFEC_PATH + '/Train/crop_images',
        "train_label_path": UCFEC_PATH + '/Train/crop_ground_truth',
        "test_image_path": UCFEC_PATH + '/Test/images/',
        "test_label_path": UCFEC_PATH + '/Test/ground_truth',
        "train_val_split": (lambda x:x, lambda x:x[:1]),
        "annReadFunc": lambda x: x['annPoints'],
        "annReadFuncTest": None
    },

}

def CreateDataLoader(opt, phase=None):
    from RawLoader import ImageDataLoader, basic_config
    from sampler import basic_config as sampler_config
    from sampler import mode_func as sampler_func
    import utils
    import numpy as np
    train_image_path = datasets[opt.dataset]["train_image_path"]
    train_label_path = datasets[opt.dataset]["train_label_path"]
    test_image_path = datasets[opt.dataset]["test_image_path"]
    test_label_path = datasets[opt.dataset]["test_label_path"]

    density_method = datasets[opt.dataset]["density_method"]

    density_method_config = basic_config[density_method]
    for k,v in datasets[opt.dataset]["density_method_config"].items():
        density_method_config[k] = v

    annReadFunc = datasets[opt.dataset]["annReadFunc"]
    annReadFuncTest = datasets[opt.dataset]["annReadFuncTest"] or annReadFunc
    split = (lambda x:x, lambda x:x)
    train_val_split = datasets[opt.dataset]["train_val_split"] or split
    train_split = train_val_split[0]
    val_split = train_val_split[1]

    print("density map config: " + datasets[opt.dataset]["density_method"])
    for k,v in density_method_config.items():
        print("{}:{}".format(k, v))

    if phase is None or phase == 'train':
        crop_type = opt.crop_type
        crop_scale = opt.crop_scale
        crop_size = opt.crop_size

        train_sample_func = sampler_func[crop_type]
        train_sample_config = sampler_config[crop_type]
        if "crop_scale" in train_sample_config.keys():
            train_sample_config['crop_scale'] = crop_scale
        if "crop_size" in train_sample_config.keys():
            train_sample_config['crop_size'] = crop_size

        print("crop config: " + crop_type)
        for k,v in train_sample_config.items():
            print("{}:{}".format(k, v))

    if phase is None or phase == 'test':
        test_crop_type = opt.test_crop_type
        test_sample_func = sampler_func[test_crop_type]
        test_sample_config = sampler_config[test_crop_type]
        if test_crop_type == 'Adap':
            test_sample_config['fixed_size'] = opt.test_fixed_size
            # if opt.test_fixed_size == -1:
            #     assert opt.test_batch_size == 1
        else:
            assert False

        print("test crop config: " + opt.test_crop_type)
        for k,v in test_sample_config.items():
            print("{}:{}".format(k, v))

    if phase is None:

        data_loader_train = ImageDataLoader(train_image_path, train_label_path, density_method, is_preload=opt.is_preload, \
                                            annReadFunc=annReadFunc, split=train_split,
                                            **density_method_config)
        data_loader_val = ImageDataLoader(train_image_path, train_label_path, density_method, is_preload=opt.is_preload, \
                                            annReadFunc=annReadFunc, split=val_split, 
                                            **density_method_config)

        data_loader_test = ImageDataLoader(test_image_path, test_label_path, density_method, is_preload=opt.is_preload, \
                                            annReadFunc=annReadFuncTest, test=True, \
                                            **density_method_config)


        data_loader_train = train_sample_func(data_loader_train, shuffle=True, \
                                        patches_per_sample=opt.patches_per_sample, **train_sample_config)

        data_loader_val = train_sample_func(data_loader_val, shuffle=True, \
                                        patches_per_sample=opt.patches_per_sample, **train_sample_config)
        data_loader_test = test_sample_func(data_loader_test, shuffle=False, **test_sample_config)

        return data_loader_train, data_loader_val, data_loader_test

    elif phase == 'train':

        data_loader_train = ImageDataLoader(train_image_path, train_label_path, density_method, is_preload=opt.is_preload, \
                                            annReadFunc=annReadFunc, split=train_split,
                                            **density_method_config)
        data_loader_train = train_sample_func(data_loader_train, shuffle=True, \
                                        patches_per_sample=opt.patches_per_sample, **train_sample_config)
        return data_loader_train

    elif phase == 'test':
        pure_test = True if not hasattr(opt, 'save_output') else not opt.save_output
        data_loader_test = ImageDataLoader(test_image_path, test_label_path, density_method, is_preload=opt.is_preload, \
                                            annReadFunc=annReadFuncTest, test=pure_test, \
                                            **density_method_config)
        data_loader_test = test_sample_func(data_loader_test, shuffle=False, **test_sample_config)

        return data_loader_test
