'''
Store all hyperparameters (learning rate, batch size, epochs)
File paths (data directory, output directory)
Model configuration settings
Makes it easy to run experiments with different settings
'''


import os

class Config:
    # Paths
    DATA_BASE_DIR = '/mnt/c/Users/muham/OneDrive/Desktop/ComputerVision/cv_data_hw2/data'
    ANNOTATION_DIR = 'data/matched_annotations'
    OUTPUT_DIR = 'outputs'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    VIS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    
    # Model
    MODEL_NAME = 'facebook/detr-resnet-50'
    NUM_CLASSES = 6  # Unknown=0, person=1, car=2, other vehicle=3, other object=4, bike=5
    
    # Training
    BATCH_SIZE = 4  # Small batch size for limited memory
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Image dimensions
    IMG_WIDTH = 1920
    IMG_HEIGHT = 1080
    
    # DETR specific
    DETR_INPUT_SIZE = 800  # DETR typically uses 800x800
    
    # Fine-tuning strategy
    # Options: 'all', 'head_only', 'transformer', 'custom_layer'
    #   'all'          - Train all parameters
    #   'head_only'    - Only train classification + bbox heads
    #   'transformer'  - Train transformer + heads, freeze backbone
    #   'custom_layer' - Train our custom layer4 + heads
    FINETUNING_STRATEGY = 'all'