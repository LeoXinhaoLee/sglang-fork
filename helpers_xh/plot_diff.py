import pdb
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_same_token_ratio(token_idx_1, token_idx_2):
    assert token_idx_1.shape == token_idx_2.shape
    BS, L = token_idx_1.shape
    same_token_ratio = []
    for t in range(L):
        same_token_ratio_t = (token_idx_1[:, t] == token_idx_2[:, t]).mean()
        same_token_ratio.append(same_token_ratio_t)
    return np.array(same_token_ratio)

def compute_prob_diff(prob_1, prob_2):
    assert prob_1.shape == prob_2.shape
    BS, L = prob_1.shape
    max_prob_diff_list = []
    for t in range(L):
        max_prob_diff_t = np.abs(prob_1[:, t] - prob_2[:, t]).mean()
        max_prob_diff_list.append(max_prob_diff_t)
    return np.array(max_prob_diff_list)

def compute_hidden_states_diff(hidden_states_1, hidden_states_2):
    assert hidden_states_1.shape == hidden_states_2.shape
    BS, L, F = hidden_states_1.shape
    max_h_diff_list = []
    for t in range(L):
        max_h_diff_t = ((hidden_states_1[:, t, :] - hidden_states_2[:, t, :]) ** 2).mean()
        max_h_diff_list.append(max_h_diff_t)
    return np.array(max_h_diff_list)

def plot_metric(x, y_list, title, ylabel, label_list, name):
    plt.figure(figsize=(8, 4))
    for i, y in enumerate(y_list):
        plt.plot(x, y, alpha=0.8, label=label_list[i])
    plt.xlabel("Time step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./{name}')
    plt.close()


def load_npy(folder):
    prefill_token_idx_all_req = np.load(f'./stats_logprob/{folder}/prefill_token_idx_all_req.npy')
    prefill_prob_all_req = np.load(f'./stats_logprob/{folder}/prefill_prob_all_req.npy')
    prefill_hidden_states_all_req = np.load(f'./stats_logprob/{folder}/prefill_hidden_states_all_req.npy')

    decode_token_idx_all_req = np.load(f'./stats_logprob/{folder}/decode_token_idx_all_req.npy')
    decode_prob_all_req = np.load(f'./stats_logprob/{folder}/decode_prob_all_req.npy')
    decode_hidden_states_all_req = np.load(f'./stats_logprob/{folder}/decode_hidden_states_all_req.npy')

    return prefill_token_idx_all_req, prefill_prob_all_req, prefill_hidden_states_all_req, \
           decode_token_idx_all_req, decode_prob_all_req, decode_hidden_states_all_req


folder = '10_28/normal/r128p512d512'
prefill_token_idx_all_req_1, prefill_prob_all_req_1, prefill_hidden_states_all_req_1, \
decode_token_idx_all_req_1, decode_prob_all_req_1, decode_hidden_states_all_req_1 = load_npy(folder)

folder = '10_28/sglang_determ/r128p512d512'
prefill_token_idx_all_req_2, prefill_prob_all_req_2, prefill_hidden_states_all_req_2, \
decode_token_idx_all_req_2, decode_prob_all_req_2, decode_hidden_states_all_req_2 = load_npy(folder)

folder = '10_28/our_triton_stride_determ_debug_plus_10/r128p512d512'
prefill_token_idx_all_req_3, prefill_prob_all_req_3, prefill_hidden_states_all_req_3, \
decode_token_idx_all_req_3, decode_prob_all_req_3, decode_hidden_states_all_req_3 = load_npy(folder)

## Token
prefill_same_token_ratio_12 = compute_same_token_ratio(prefill_token_idx_all_req_1, prefill_token_idx_all_req_2)
decode_same_token_ratio_12 = compute_same_token_ratio(decode_token_idx_all_req_1, decode_token_idx_all_req_2)
prefill_same_token_ratio_13 = compute_same_token_ratio(prefill_token_idx_all_req_1, prefill_token_idx_all_req_3)
decode_same_token_ratio_13 = compute_same_token_ratio(decode_token_idx_all_req_1, decode_token_idx_all_req_3)

## Prob
prefill_prob_diff_12 = compute_prob_diff(prefill_prob_all_req_1, prefill_prob_all_req_2)
decode_prob_diff_12 = compute_prob_diff(decode_prob_all_req_1, decode_prob_all_req_2)
prefill_prob_diff_13 = compute_prob_diff(prefill_prob_all_req_1, prefill_prob_all_req_3)
decode_prob_diff_13 = compute_prob_diff(decode_prob_all_req_1, decode_prob_all_req_3)

## Hidden states
prefill_hidden_states_diff_12 = compute_hidden_states_diff(prefill_hidden_states_all_req_1, prefill_hidden_states_all_req_2)
decode_hidden_states_diff_12 = compute_hidden_states_diff(decode_hidden_states_all_req_1, decode_hidden_states_all_req_2)
prefill_hidden_states_diff_13 = compute_hidden_states_diff(prefill_hidden_states_all_req_1, prefill_hidden_states_all_req_3)
decode_hidden_states_diff_13 = compute_hidden_states_diff(decode_hidden_states_all_req_1, decode_hidden_states_all_req_3)

diff_dict = {
    'prefill_same_token_ratio_12': prefill_same_token_ratio_12,
    'decode_same_token_ratio_12': decode_same_token_ratio_12,
    'prefill_same_token_ratio_13': prefill_same_token_ratio_13,
    'decode_same_token_ratio_13': decode_same_token_ratio_13,

    'prefill_prob_diff_12': prefill_prob_diff_12,
    'decode_prob_diff_12': decode_prob_diff_12,
    'prefill_prob_diff_13': prefill_prob_diff_13,
    'decode_prob_diff_13': decode_prob_diff_13,

    'prefill_hidden_states_diff_12': prefill_hidden_states_diff_12,
    'decode_hidden_states_diff_12': decode_hidden_states_diff_12,
    'prefill_hidden_states_diff_13': prefill_hidden_states_diff_13,
    'decode_hidden_states_diff_13': decode_hidden_states_diff_13,
}

# folder_name = "10_28/diff_trace/r128p512d512"
# os.makedirs(f'{folder_name}', mode=0o777, exist_ok=True)
# np.savez_compressed(f"{folder_name}/diff_dict.npz", **diff_dict)
# exit(0)

######## Plot ########
## Token
# plot_metric(x, prefill_same_token_ratio, "Prefill Same Token Ratio", "Normal vs Ours")
# plot_metric(x, decode_same_token_ratio, "Decode Same Token Ratio", "Normal vs Ours")

## Hidden states
x = np.arange(prefill_hidden_states_diff_12.shape[0])
plot_metric(
    x, 
    [prefill_hidden_states_diff_12, prefill_hidden_states_diff_13], 
    label_list=["SGLang", "Ours"],
    ylabel='MSE', title="Distance from non-determ prefill hidden states", 
    # name="prefill_stride_to_bf16_mse.png",
    # name="prefill_stride_mse.png",
    # name="prefill_bias_mse.png",
    # name="prefill_r1p16_mse.png",
    name="prefill_stride_debug_plust_10_mse.png",
)
x = np.arange(decode_hidden_states_diff_12.shape[0])
plot_metric(
    x, 
    [decode_hidden_states_diff_12, decode_hidden_states_diff_13],
    label_list=["SGLang", "Ours"],
    ylabel='MSE', title="Distance from non-determ decode hidden states", 
    # name="decode_stride_to_bf16_mse.png",
    # name="decode_stride_mse.png",
    # name="decode_bias_mse.png",
    # name="decode_r1p16_mse.png",
    name="decode_stride_debug_plust_10_mse.png",
)
