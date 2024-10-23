class Options:
    def __init__(self):
        self.exp = ""
        self.max_epoch = 50
        self.lr = 0.001

        self.model_type = 'siamese'
        self.model_params = {'input_dim': 3 * 84 * 84, 'hidden_dim': 128, 'output_dim': 64, 'depth': 2}
        self.init_weights = None

        self.categories = None
        self.dataset = None

        self.way = 5
        self.test_way = 5
        self.shot = 1
        self.query = 10

    def update(self, new_values):
        for key, value in new_values.items():
            if hasattr(self, key):
                setattr(self, key, value)
