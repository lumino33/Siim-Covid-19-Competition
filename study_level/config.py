class Config:
    
    folds = [0,1,2,3,4]
    image_dirs = "/home/hungld11/Documents/SIIM COVID DETECTION/dataset/train"
    image_size = 512
    num_workers = 0
    
    #model
    encoder_name = "efficientnet-b5"
    encoder_weights = "imagenet" # 'imagenet', 'advprop' or None
    decoder_channels = [256, 128, 64, 32, 16]
    num_classes = 4
    in_features = 512 #2048
      
    #train
    lr = 3e-4
    weight_decay = 1e-3
    batch_size = 6
    epochs = 30