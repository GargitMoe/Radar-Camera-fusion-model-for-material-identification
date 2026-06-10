# config.py

#  "binary" or "four"
TASK_TYPE = "four"   

TRAIN_SCENES = ['scene1','scene2']
VAL_SCENES   = ['scene3']

if TASK_TYPE == "binary":
    # metal vs non-metal
    CLASS_MAP = {
        'metal': 0,
        'glass': 1,
        'plastic': 1,
        'cardboard': 1,
    }
    NUM_CLASSES  = 2
    TARGET_NAMES = ["metal", "non-metal"]

else:  
    CLASS_MAP = {
        'metal': 0,
        'glass': 1,
        'plastic': 2,
        'cardboard': 3,
    }
    NUM_CLASSES  = 4
    TARGET_NAMES = ["metal", "glass", "plastic", "cardboard"]
