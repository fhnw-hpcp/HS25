from mpi4py import MPI
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Generate or load your dataset (a synthetic dataset). Every process holds all data which is
#clearly inefficient, in a productive application, each rank would reads its share from e.g a hdf5 file
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print ("Train", X_train.shape, y_train.shape)
print ("Test", X_test.shape, y_test.shape)

# Split training data across processes (in a module fashion)
X_train_local = X_train[rank::size]
y_train_local = y_train[rank::size]

print ("Local Train", X_train_local.shape, y_train_local.shape)

# Train a local model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train_local, y_train_local)

# Make local predictions (on full test set)
y_pred_local = model.predict(X_test)
print ("Local Test", X_test.shape, y_pred_local.shape)

# Gather predictions from all processes to rank 0
all_preds = comm.gather(y_pred_local, root=0)

if rank == 0:
    print ("Rank 0 gathered a", type(all_preds)) #Above syntax returns a list
    # Stack predictions and compute ensemble prediction
    all_preds = np.vstack(all_preds)
    print (all_preds.dtype, all_preds.shape)
    # For classification: majority vote
    y_pred_ensemble = np.round(np.mean(all_preds, axis=0)).astype(int)
    print (y_pred_ensemble)
    # For regression: average
    #y_pred_ensemble = np.mean(all_preds, axis=0)

    # Evaluate ensemble performance
    accuracy = np.mean(y_pred_ensemble == y_test)
    print(f"Ensemble accuracy: {accuracy:.2f}")

