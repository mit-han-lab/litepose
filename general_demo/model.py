from core import process
import onnxruntime
from yacs.config import CfgNode as CN


def get_cfg(cfg_path):
    _C = CN(new_allowed=True)

    _C.OUTPUT_DIR = ''
    _C.LOG_DIR = ''
    _C.DATA_DIR = ''
    _C.GPUS = (0,)
    _C.WORKERS = 4
    _C.PRINT_FREQ = 20
    _C.AUTO_RESUME = True
    _C.PIN_MEMORY = True
    _C.RANK = 0
    _C.VERBOSE = True
    _C.DIST_BACKEND = 'nccl'
    _C.MULTIPROCESSING_DISTRIBUTED = True

    # FP16 training params
    _C.FP16 = CN(new_allowed=True)
    _C.FP16.ENABLED = False
    _C.FP16.STATIC_LOSS_SCALE = 1.0
    _C.FP16.DYNAMIC_LOSS_SCALE = False

    # Cudnn related params
    _C.CUDNN = CN(new_allowed=True)
    _C.CUDNN.BENCHMARK = True
    _C.CUDNN.DETERMINISTIC = False
    _C.CUDNN.ENABLED = True

    # common params for NETWORK
    _C.MODEL = CN(new_allowed=True)
    _C.MODEL.NAME = 'pose_multi_resolution_net_v16'
    _C.MODEL.INIT_WEIGHTS = True
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.NUM_JOINTS = 17
    _C.MODEL.TAG_PER_JOINT = True
    _C.MODEL.EXTRA = CN(new_allowed=True)
    _C.MODEL.SYNC_BN = False

    _C.LOSS = CN(new_allowed=True)
    _C.LOSS.NUM_STAGES = 1
    _C.LOSS.WITH_HEATMAPS_LOSS = (True,)
    _C.LOSS.HEATMAPS_LOSS_FACTOR = (1.0,)
    _C.LOSS.WITH_AE_LOSS = (True,)
    _C.LOSS.AE_LOSS_TYPE = 'max'
    _C.LOSS.PUSH_LOSS_FACTOR = (0.001,)
    _C.LOSS.PULL_LOSS_FACTOR = (0.001,)

    # DATASET related params
    _C.DATASET = CN(new_allowed=True)
    _C.DATASET.ROOT = ''
    _C.DATASET.DATASET = 'coco_kpt'
    _C.DATASET.DATASET_TEST = 'coco'
    _C.DATASET.NUM_JOINTS = 17
    _C.DATASET.MAX_NUM_PEOPLE = 30
    _C.DATASET.TRAIN = 'train2017'
    _C.DATASET.TEST = 'val2017'
    _C.DATASET.DATA_FORMAT = 'jpg'

    # training data augmentation
    _C.DATASET.MAX_ROTATION = 30
    _C.DATASET.MIN_SCALE = 0.75
    _C.DATASET.MAX_SCALE = 1.25
    _C.DATASET.SCALE_TYPE = 'short'
    _C.DATASET.MAX_TRANSLATE = 40
    _C.DATASET.INPUT_SIZE = 512
    _C.DATASET.OUTPUT_SIZE = [128, 256, 512]
    _C.DATASET.FLIP = 0.5

    # heatmap generator (default is OUTPUT_SIZE/64)
    _C.DATASET.SIGMA = -1
    _C.DATASET.SCALE_AWARE_SIGMA = False
    _C.DATASET.BASE_SIZE = 256.0
    _C.DATASET.BASE_SIGMA = 2.0
    _C.DATASET.INT_SIGMA = False

    _C.DATASET.WITH_CENTER = False

    # train
    _C.TRAIN = CN(new_allowed=True)

    _C.TRAIN.LR_FACTOR = 0.1
    _C.TRAIN.LR_STEP = [90, 110]
    _C.TRAIN.LR = 0.001

    _C.TRAIN.OPTIMIZER = 'adam'
    _C.TRAIN.MOMENTUM = 0.9
    _C.TRAIN.WD = 0.0001
    _C.TRAIN.NESTEROV = False
    _C.TRAIN.GAMMA1 = 0.99
    _C.TRAIN.GAMMA2 = 0.0

    _C.TRAIN.BEGIN_EPOCH = 0
    _C.TRAIN.END_EPOCH = 140

    _C.TRAIN.RESUME = False
    _C.TRAIN.CHECKPOINT = ''

    _C.TRAIN.IMAGES_PER_GPU = 32
    _C.TRAIN.SHUFFLE = True

    # testing
    _C.TEST = CN(new_allowed=True)

    # size of images for each device
    # _C.TEST.BATCH_SIZE = 32
    _C.TEST.IMAGES_PER_GPU = 32
    # Test Model Epoch
    _C.TEST.FLIP_TEST = False
    _C.TEST.ADJUST = True
    _C.TEST.REFINE = True
    _C.TEST.SCALE_FACTOR = [1]
    # group
    _C.TEST.DETECTION_THRESHOLD = 0.2
    _C.TEST.TAG_THRESHOLD = 1.
    _C.TEST.USE_DETECTION_VAL = True
    _C.TEST.IGNORE_TOO_MUCH = False
    _C.TEST.MODEL_FILE = ''
    _C.TEST.IGNORE_CENTER = True
    _C.TEST.NMS_KERNEL = 3
    _C.TEST.NMS_PADDING = 1
    _C.TEST.PROJECT2IMAGE = False

    _C.TEST.WITH_HEATMAPS = (True,)
    _C.TEST.WITH_AE = (True,)

    _C.TEST.LOG_PROGRESS = False

    # debug
    _C.DEBUG = CN(new_allowed=True)
    _C.DEBUG.DEBUG = True
    _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
    _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
    _C.DEBUG.SAVE_HEATMAPS_GT = True
    _C.DEBUG.SAVE_HEATMAPS_PRED = True
    _C.DEBUG.SAVE_TAGMAPS_PRED = True
    cfg = _C
    cfg.defrost()
    cfg.merge_from_file(cfg_path)

    return cfg


class PEExcutor:
    def __init__(self, model_path, model_type, cfg_path):
        self.cfg = get_cfg(cfg_path)
        if model_type == "onnx":
            self.model = ONNXModel(model_path)
        else:
            raise Exception
        
    def __call__(self, image):
        return process(self.cfg, image, self.model)[:, ::-1]
        
        

class PEModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def __call__(self, image):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError


class ONNXModel(PEModel):
    def load_model(self, model_path):
        return onnxruntime.InferenceSession(model_path)
        
    def __call__(self, image):
        return ortsession.run(None, image)



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms as transforms
    img = Image.open("sample_model/model.jpg")
    resize = transforms.Resize([256,256])
    img = resize(img)
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    img.unsqueeze_(0)

       

    gg = PEExcutor("sample_model/test.onnx", "onnx", "sample_model/mobile.yaml")
    ort_inputs = {gg.model.model.get_inputs()[0].name: to_numpy(img)}
    
    print(gg(ort_inputs))