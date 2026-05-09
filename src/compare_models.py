"""

# for instance
# 1. Instanzen erstellen
model_adam = SquatClassifier(input_size=12, num_classes=4)
model_sgd = SquatClassifier(input_size=12, num_classes=4)

# 2. Gewichte laden
model_adam.load_state_dict(torch.load("model_adam_lr1e-3.pth"))
model_sgd.load_state_dict(torch.load("model_sgd_lr1e-2.pth"))

# 3. Beide in den Eval-Modus schalten
# eval() geht von training- zu evaluationsmodus (Inferenz).
model_adam.eval() # just paste the test_loop in here, that is the eval-stage
model_sgd.eval()

# Jetzt kannst du in einer Schleife beide Modelle mit demselben Test-Batch füttern
# und die Ergebnisse direkt nebeneinander in einer Tabelle ausgeben.


# my comparison ideas:
model_adam with lr1e-3 => e.g. x Accuracy in x epochs
model_sgd with lr1e-2 => e.g. x Acc. with x epochs

Wichtiger Hinweis: Wenn du verschiedene Algorithmen vergleichst
(z.B. Adam vs. SGD), achte darauf, dass du das Modell vor dem zweiten 
Training neu initialisierst (also das Objekt neu erstellst). 
Sonst trainiert der SGD auf den bereits gelernten Gewichten des 
Adam-Modells weiter, was dein Ergebnis verfälschen würde.

"""