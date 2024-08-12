from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier():
    def __init__(self, n_neighbors: int) -> None:
        self.model = None
        self.n_neighbors = n_neighbors

    
    def fit(self, X, y):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(list(X), list(y))


    def predict(self, X):
        predictions = self.model.predict(list(X))

        return predictions
    