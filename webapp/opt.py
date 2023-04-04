class OPT:
    SEED = 2
    THRESHOLD = 0.4
    MODEL = "openCLIP"
    DEVICE='cuda:0'
    PROCESSING_FRAMES = 1
    TEST_MODE = True
    DISTANCE_TYPE = "inverse_cosine"
    VERBOSE = False
    WEBAPP = True
    WEBAPP_PRETRAIN = True
    DATA_PATH = '/home/leonardolabs/data/'
    PRINT_EVERY = 200
    DATASET = 'CORE50'
    SHUFFLED_SCENARIOS = "shuffled"#shuffled/ordered
    UPDATE_PROBABILITY = 1