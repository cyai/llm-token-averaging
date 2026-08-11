""" Compute the loss of the Chinchilla model """

A = 406.4
alpha = 0.3392
B = 410.7
beta = 0.2849
E = 1.6934

def chinchilla_loss(N: float, D: float) -> float:
    return A / N**alpha + B / D**beta + E

if __name__ == "__main__":
    N = 125*10**6
    D = 2.5*10**9
    print(f"Chinchilla loss: {chinchilla_loss(N, D)}")