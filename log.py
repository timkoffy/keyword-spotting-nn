import pickle


class TrainingStats:
    def __init__(self):
        self.history = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
                "epochs": []
                }

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.history["epochs"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        
    def save(self, path="training_stats.pkl"):
        with open(path, "wb") as f: 
            pickle.dump(self, f)

    @staticmethod
    def load(path="training_stats.pkl"):
        with open(path, "rb") as f: 
            return pickle.load(f)
