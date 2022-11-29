class Config():
    backbone_name = 'resnet101d'
    folds = [3, 4, 0, 1, 2,]
    image_dirs = "/home/hungld11/Documents/SIIM COVID DETECTION/dataset/train"
    image_size = 512
    num_workers = 0
    
    #model
    backbone_name = 'resnet101d'
    trainable_layers = 5 # <=5 and >= 0
    num_classes = 4
    in_features = 2048 
      
    #train
    lr = 3e-4
    weight_decay = 1e-3
    batch_size = 16
    epochs = 30
    
    #test
    test_image_dirs = "/home/hungld11/Documents/SIIM COVID DETECTION/dataset/test"