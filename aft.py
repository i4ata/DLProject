import torch
import torch.nn as nn

class AFTSimple(nn.Module):
    def __init__(self, e_dim: int, sequence_len: int):
        super(AFTSimple, self).__init__()
        self.e_dim = e_dim
        self.sequence_len = sequence_len

        # a separate variable for the dimensions of the query, key, and value can be used later,
        # but I decided to keep it simple for now
        self.W_q = nn.Linear(e_dim, e_dim)
        self.W_k = nn.Linear(e_dim, e_dim)
        self.W_v = nn.Linear(e_dim, e_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None):
        if None in [k, v]:  # we have an encoder/decoder-only architecture - singular input
            k = q
            v = q
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        weights = torch.softmax(K, dim=1) # gets the "weights" for the weighted sum
        weighted_context = torch.sum(weights * V, dim=1).sum(dim=1, keepdim=True) # weighted sum
        Q = self.sigmoid(Q)
        Y = Q * weighted_context

        return Y

class AFTLocal(nn.Module):
    def __init__(self, e_dim: int, sequence_len: int, s: int):
        super(AFTLocal, self).__init__()
        self.e_dim = e_dim
        self.sequence_len = sequence_len # T
        self.s = s # window size

        self.W_q = nn.Linear(e_dim, e_dim)
        self.W_k = nn.Linear(e_dim, e_dim)
        self.W_v = nn.Linear(e_dim, e_dim)

        self.position_biases = nn.Parameter(torch.randn(self.s * 2 + 1, self.e_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None):
        if None in [k, v]: # we have an encoder/decoder-only architecture - singular input
            k = q
            v = q
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        batch_size, seq_len, e_dim = q.size()
        context = torch.zeros_like(Q)

        full_mask = self.generate_full_mask(seq_len, self.s)

        for i in range(self.sequence_len): # for each position in the sequence
            start = max(0, i - self.s)
            end = min(seq_len, i + self.s + 1)

            relative_positions = torch.arange(start, end) - i
            bias_indices = relative_positions + self.s

            position_biases = self.position_biases[bias_indices]  # Shape: [window_size, e_dim]
            position_biases = position_biases.unsqueeze(0).expand(batch_size, end - start, e_dim)

            masked_K = K[:, start:end] + position_biases
            masked_V = V[:, start:end]

            window_mask = full_mask[i, start:end].unsqueeze(0).unsqueeze(-1)
            window_mask = window_mask.expand(batch_size, end-start, e_dim)  # [batch_size, window_size, e_dim]

            masked_K = torch.where(window_mask, masked_K, torch.zeros_like(masked_K))

            q_i = Q[:, i, :].unsqueeze(1)  # makes it [batch_size, 1, e_dim]
            weights = torch.softmax(q_i * masked_K, dim=1)

            Q_sig = torch.sigmoid(Q)
            q_i_sig = Q_sig[:, i, :].unsqueeze(1)

            context[:, i, :] = torch.sum(weights * masked_V, dim=1) * q_i_sig.squeeze(1)

        y = self.sigmoid(Q) * context

        return y

    @staticmethod
    def generate_full_mask(seq_len, window_size):
        indices = torch.arange(seq_len).unsqueeze(0)
        abs_diff = torch.abs(indices - indices.T)
        mask = abs_diff < window_size
        return mask


if __name__ == '__main__':
    e_dim = 64
    sequence_len = 32
    s = 5
    batch_size = 32

    aft_simple = AFTSimple(e_dim=e_dim, sequence_len=sequence_len)
    aft_local = AFTLocal(e_dim=e_dim, sequence_len=sequence_len, s=s)
    x = torch.rand(batch_size, sequence_len, e_dim)

    try:
        # y = aft_simple(x)
        y = aft_local(x)
        print(f"Output / shape: {y} \n {y.shape}\n"
              f"Output produced successfully.")
    except Exception as e:
        print(f"Error during model forward pass: {e}.")
