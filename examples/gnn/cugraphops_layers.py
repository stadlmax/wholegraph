import torch
import torch.nn as nn

from typing import Union

from pylibcugraphops import make_mfg_csr, make_mfg_csr_hg, make_fg_csr, make_fg_csr_hg
import pylibcugraphops.torch.autograd as cugraphops


def create_mfg_csr(csr_row_ptr, csr_col_idx, sample_size):
    n_out_nodes = len(csr_row_ptr) - 1
    n_in_nodes = csr_col_idx.max().item() + 1
    out_nodes = torch.arange(n_out_nodes, device="cuda", dtype=csr_row_ptr.dtype)
    in_nodes = torch.arange(n_in_nodes, device="cuda", dtype=csr_col_idx.dtype)
    return make_mfg_csr(out_nodes, in_nodes, csr_row_ptr, csr_col_idx, sample_size)


def create_mfg_csr_hg(csr_row_ptr, csr_col_idx, edge_types, num_relations, sample_size):
    n_in_nodes = len(csr_row_ptr) - 1
    in_nodes = torch.arange(n_in_nodes, device="cuda", dtype=csr_row_ptr.dtype)
    n_out_nodes = csr_col_idx.max()
    out_nodes = torch.arange(n_out_nodes, device="cuda", dtype=csr_col_idx.dtype)
    return make_mfg_csr_hg(
        out_nodes,
        in_nodes,
        csr_row_ptr,
        csr_col_idx,
        sample_size,
        n_node_types=0,
        n_edge_types=num_relations,
        out_node_types=None,
        in_node_types=None,
        edge_types=edge_types,
    )


class SAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super(SAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = torch.nn.Linear(in_channels + in_channels, out_channels)

        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.lin.weight, gain=gain)

    def forward(
        self,
        x_feat,
        csr_row_ptr,
        csr_col_ind,
        _,
        sample_count,
    ):
        torch.cuda.nvtx.range_push("create_mfg_csr")
        mfg = create_mfg_csr(csr_row_ptr, csr_col_ind, sample_count)
        torch.cuda.nvtx.range_pop()

        x_agg = cugraphops.agg_concat_n2n(x_feat, mfg, "mean")

        y = self.lin(x_agg)
        if self.bias is not None:
            y = y + self.bias
        return y


class GATConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        num_heads: int = 1,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        mean_output: bool = False,
    ):
        super(GATConv, self).__init__()
        self.in_channels = in_channels
        self.head_channels = head_channels
        self.out_channels = head_channels * num_heads
        self.num_heads = num_heads
        self.add_self_loops = add_self_loops
        self.mean_output = mean_output
        self.negative_slope = negative_slope
        self.lin = torch.nn.Linear(in_channels, self.out_channels, bias=False)
        self.attn_weights = torch.nn.Parameter(
            torch.FloatTensor(size=(1, self.out_channels * 2))
        )
        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.lin.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.attn_weights[:, : self.out_channels])
        torch.nn.init.xavier_uniform_(self.attn_weights[:, self.out_channels :])

    def forward(self, x_feat, csr_row_ptr, csr_col_ind, sample_dup_count, sample_count):

        if self.add_self_loops:
            (
                csr_row_ptr_looped,
                csr_col_ind_looped,
                _,
            ) = torch.ops.wholegraph.csr_add_self_loop(
                csr_row_ptr, csr_col_ind, sample_dup_count
            )
            sample_count += 1

        torch.cuda.nvtx.range_push("create_mfg_csr")
        mfg = create_mfg_csr(csr_row_ptr_looped, csr_col_ind_looped, sample_count)
        torch.cuda.nvtx.range_pop()
        x_lin = self.lin(x_feat)

        y = cugraphops.mha_gat_n2n(
            x_lin,
            self.attn_weights.squeeze(),
            mfg,
            num_heads=self.num_heads,
            activation="LeakyReLU",
            negative_slope=self.negative_slope,
            concat_heads=not self.mean_output,
        )

        return y


class RGCNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        root_weight: bool = True,
        bias: bool = True,
        aggr: str = "mean",
    ):
        super(RGCNConv, self).__init__()
        assert aggr in ("mean", "sum")
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.root_weight = root_weight
        dim_root_weight = 1 if root_weight else 0
        self.lin_weight = nn.Parameter(
            torch.Tensor(num_relations + dim_root_weight, in_channels, out_channels)
        )
        if bias is True:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        end = -1 if self.root_weight else None
        torch.nn.init.xavier_uniform_(self.weight[:end])
        if self.root_weight:
            torch.nn.init.xavier_uniform_(self.weight[-1])
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, subgraph, x_feat):
        csr_row_ptr = subgraph["csr_row_ptr"]
        csr_col_ind = subgraph["csr_col_ind"]
        dup_count = subgraph["dup_count"]
        edge_type = subgraph["edge_type"]

        torch.cuda.nvtx.range_push("create_mfg_csr_hg")
        mfg = create_mfg_csr_hg(
            csr_row_ptr, csr_col_ind, edge_type, self.num_relations, dup_count
        )
        torch.cuda.nvtx.range_pop()

        x_agg = cugraphops.agg_hg_basis_n2n_post(
            x_feat, None, mfg, not self.root_weight, bool(self.aggr == "mean")
        )

        y = x_agg @ self.weight.view(-1, self.out_channels)
        if self.bias is not None:
            y = y + self.bias

        return y
