import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.omega_c = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.num_heads = num_heads

        self.mh_sa_concat = None

        self.activations = []

    def forward(self, x, mask=None):
        mh_sa_array = []
        d_queries = self.d_model / self.num_heads

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Omega_c = self.omega_c(x)

        for h in range(self.num_heads):
            dot_product = Q[:, h, :] @ K[:, h, :].transpose()
            dot_product_div = dot_product / torch.sqrt(torch.tensor(d_queries))

            if mask is not None:
                dot_product_div = torch.where(mask == 0, torch.tensor(-1e9), torch.tensor(dot_product_div))

            activations = torch.softmax(dot_product_div, dim=-1)
            self.activations.append(activations)  # adding the activations to a list, so it can be accessed in back pass

            sa = activations @ V[:, h, :]

            mh_sa_array.append(sa)

        mh_sa_concat = torch.cat(mh_sa_array, dim=1)
        mh_sa = Omega_c(mh_sa_concat) @ mh_sa_concat

        self.mh_sa_concat = mh_sa_concat

        return mh_sa

    def backward(self, grad_output, mask=None):
        grad_mh_sa_concat = np.array_split(grad_output, self.num_heads, axis=1)
        grad_omega_c = grad_output @ self.mh_sa_concat.T

        grad_q_list = []
        grad_k_list = []
        grad_v_list = []

        for h in range(self.num_heads):
            grad_sa = grad_mh_sa_concat[h]
            grad_activations = grad_sa @ self.v_proj[:, h, :].T
            grad_v_slice = self.activations[h].T @ grad_sa

            # the softmax
            grad_dot_product_div = self.activations * (grad_activations - np.sum(grad_activations * self.activations[:, h, :], axis = 1, keepdims=True))

            if mask is not None:
                grad_dot_product_div = torch.where(mask == 0, 0, grad_dot_product_div)

            head_dim = self.q_proj.shape(2)
            grad_dot_product = grad_dot_product_div / np.sqrt(head_dim)

            grad_q_slice = grad_dot_product @ self.k_proj[:, h, :].T
            grad_k_slice = self.q_proj[:, h, :].T @ grad_dot_product

            grad_q_list.append(grad_q_slice)
            grad_k_list.append(grad_k_slice)
            grad_v_list.append(grad_v_slice)

        grad_q = np.concatenate(grad_q_list, axis=1)
        grad_k = np.concatenate(grad_k_list, axis=1)
        grad_v = np.concatenate(grad_v_list, axis=1)

        self.q_proj.grad_weight = grad_q
        self.k_proj.grad_weight = grad_k
        self.v_proj.grad_weight = grad_v

        grad_x = grad_q + grad_k + grad_v

        return grad_x
