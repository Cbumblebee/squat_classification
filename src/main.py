import torch
from torch import nn
from learning import train_loop, test_loop
import neural_network
from fetch_data import train_loader, test_loader
from datetime import datetime, timezone
import time


# run it from inside /src, bc of relative paths
loss_fn = nn.CrossEntropyLoss()

# Initialize model and optimizer new because otherwise the models would run with old, already learned, parameters.
# Which we don't want for equal comparison.
def run_experiment(optimizer_type="ADAM", learning_rate=1e-3, epochs=50, batch_size = 32):
    best_accuracy = 0.0 # in the iteration phase, saving the best accuracy to minimize overfitting.
    best_epoch = 0
    best_loss = -1
    model = neural_network.NeuralNetwork()
    model_name = f"model_{optimizer_type.lower()}_lr{learning_rate}.pth"

    if optimizer_type == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    start = time.perf_counter()
    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, batch_size)
        current_acc, current_loss = test_loop(test_loader, model, loss_fn)
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            best_epoch = e+1
            best_loss = current_loss
            # save the model
            torch.save(model.state_dict(), f"../models/{model_name}")
            print(f"--> New accuracy record: {100 * best_accuracy:.1f}% achieved in epoch {e + 1}\n")

    elapsed_time = time.perf_counter() - start
    return model, model_name, best_accuracy, best_epoch, best_loss, elapsed_time


# Later: If I want to load the model, I need to write:
# model.load_state_dict(torch.load(f"name.pth"))
# model.eval()
# The exact names are in the comparison.txt file.
""" have commented that out, because I know that Adam l2=1e-2 is the best
if __name__ == "__main__":
    model_a_lr3, model_a_lr3_name, model_a_lr3_acc, model_a_lr3_epoch, model_a_lr3_loss, model_a_lr3_time = run_experiment()
    model_s_lr3, model_s_lr3_name, model_s_lr3_acc, model_s_lr3_epoch, model_s_lr3_loss, model_s_lr3_time = run_experiment(optimizer_type="SGD")
    model_a_lr2, model_a_lr2_name, model_a_lr2_acc, model_a_lr2_epoch, model_a_lr2_loss, model_a_lr2_time = run_experiment(learning_rate=1e-2)
    model_s_lr2, model_s_lr2_name, model_s_lr2_acc, model_s_lr2_epoch, model_s_lr2_loss, model_s_lr2_time = run_experiment(optimizer_type="SGD", learning_rate=1e-2)

    with open("comparison.txt", "a") as f:
        f.write("\n\n")
        f.write(f"new estimation: {datetime.now(tz=timezone.utc).strftime("%d/%m/%Y %H:%M:%S")}\n")
        f.write("Adam, lr=3:\n")
        f.write(f"name={model_a_lr3_name}; best accuracy={model_a_lr3_acc}; best epoch={model_a_lr3_epoch}; loss={model_a_lr3_loss}; elapsed time={model_a_lr3_time}\n\n")
        f.write("SGD, lr=3:\n")
        f.write(f"name={model_s_lr3_name}; best accuracy={model_s_lr3_acc}; best epoch={model_s_lr3_epoch}; loss={model_s_lr3_loss}; elapsed time={model_s_lr3_time}\n\n")
        f.write("Adam, lr=2:\n")
        f.write(f"name={model_a_lr2_name}; best accuracy={model_a_lr2_acc}; best epoch={model_a_lr2_epoch}; loss={model_a_lr2_loss}; elapsed time={model_a_lr2_time}\n\n")
        f.write("SGD, lr=2:\n")
        f.write(f"name={model_s_lr2_name}; best accuracy={model_s_lr2_acc}; best epoch={model_s_lr2_epoch}; loss={model_s_lr2_loss}; elapsed time={model_s_lr2_time}\n\n")
"""
if __name__ == "__main__":
    torch.manual_seed(42)
    model_a_lr2, model_a_lr2_name, model_a_lr2_acc, model_a_lr2_epoch, model_a_lr2_loss, model_a_lr2_time = run_experiment(learning_rate=1e-2)
    with open("new_run.txt", "a") as f:
        f.write(f"estimation: {datetime.now(tz=timezone.utc).strftime("%d/%m/%Y %H:%M:%S")}\n")
        f.write(f"name={model_a_lr2_name}; best accuracy={model_a_lr2_acc}; best epoch={model_a_lr2_epoch}; loss={model_a_lr2_loss}; elapsed time={model_a_lr2_time}\n\n")

"""
model_adam_lr1e output:
Epoch 50
-------------------------------
loss: 0.215353  [   32/37953]
loss: 0.172428  [ 3232/37953]
loss: 0.057357  [ 6432/37953]
loss: 0.251622  [ 9632/37953]
loss: 0.137782  [12832/37953]
loss: 0.306968  [16032/37953]
loss: 0.184318  [19232/37953]
loss: 0.331413  [22432/37953]
loss: 0.379333  [25632/37953]
loss: 0.195157  [28832/37953]
loss: 0.280697  [32032/37953]
loss: 0.202242  [35232/37953]
Test Error: 
 Accuracy: 91.8%, Avg loss: 0.240197 

--> New accuracy record: 91.8% achieved in epoch 50

Done!

Process finished with exit code 0
"""