"""
Training FLOPs = (tokens_seen / seq_len) × n_layers × transformer_L × 3 × (23d² + 4 × transformer_L × d) 
"""


def calculate_flops(tokens_seen: int, seq_len: int, n_layers: int, d_model: int, transformer_L: int) -> float:
    return (tokens_seen / seq_len) * n_layers * transformer_L * 3 * (23 * d_model**2 + 4 * transformer_L * d_model) 

if __name__ == "__main__":
    tokens_seen = 5*10**9
    seq_len = 1024
    n_layers = 12
    d_model = 768
    transformer_L = seq_len # k=1
    flops_k1 = calculate_flops(tokens_seen//2, seq_len, n_layers, d_model, transformer_L)
    print(f"FLOPS for k=1: {flops_k1/10**15} TFlops")
    flops_k2 = calculate_flops(tokens_seen, seq_len, n_layers, d_model, transformer_L//2)
    print(f"FLOPS for k=2: {flops_k2/10**15} TFlops")
    tokens_seen_k2 = tokens_seen * flops_k2 / flops_k1

    # total flops saved by k=2
    flops_saved = flops_k1 - flops_k2
    print(f"Total flops saved by k=2: {flops_saved/10**15} TFlops")
    print(f"Percentage of flops saved by k=2: {flops_saved/flops_k1*100}%")

    # total more tokens k=2 can use to be iso-flops with k=1
    flop_per_token = calculate_flops(1, seq_len, n_layers, d_model, transformer_L)
    print(f"Flop per token: {flop_per_token/10**15} TFlops")
    print(f"Total more tokens k=2 can use to be iso-flops with k=1: {flops_saved/flop_per_token/10**9} B tokens")  