from datasets import load_dataset, concatenate_datasets

fleurs_korean = load_dataset('google/fleurs', "ko_kr", split='train')
fleurs_english = load_dataset('google/fleurs', "en_us", split='train')

data = concatenate_datasets([fleurs_korean, fleurs_english])


class LanguageClassifier():
    def __init__(self, data):
        self.data = data
        self.model = None

    def train(self):
        # Train the model
        pass

    def predict(self, text):
        # Predict the language of the text
        pass

    def evaluate(self):
        # Evaluate the model
        pass