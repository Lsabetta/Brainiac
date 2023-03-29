class OPT:
    SEED = 0
    THRESHOLD = 18
    MODEL = "openCLIP"
    DEVICE='cuda:0'
    PROCESSING_FRAMES = 1
    TEST_MODE = True
    DISTANCE_TYPE = "inverse_cosine"
    VERBOSE = False
    WEBAPP = False
    WEBAPP_PRETRAIN = False
    DATA_PATH = '/home/leonardolabs/data/'
    PRINT_EVERY = 200
    DATASET = 'CORE50'
    SHUFFLED_SCENARIOS = "ordered"#/ordered
    UPDATE_PROBABILITY = 1