import torch

def find_union(x, budget):
    """
    Find the union of the topk indices."""
    assert x.dim() == 2, "x should be a 2D tensor"
    sorted_x, sorted_indices = torch.sort(x, dim=-1)
    flag = torch.cat([torch.zeros((x.shape[0], 1), device=x.device).bool(), sorted_x[:, 1:] == sorted_x[:, :-1]], dim=1)
    sorted_indices.masked_fill_(mask = flag, value=sorted_indices.max() + 1)
    _, sorted_indices = torch.sort(sorted_indices, dim=-1)
    return sorted_x[torch.arange(sorted_x.size(0)).unsqueeze(1), sorted_indices[:, :budget]]

def _get_tidal_mask(attn_mask, attn_weights, window_budget, sink_budget, tidal_budget, merge_type="union"):
    """
    This function computes the tidal mask for the attention weights based on the given budgets.
    
    Args:
        attn_mask (torch.Tensor): The attention mask tensor of shape [batch_size, 1, qo_len, kv_len].
        attn_weights (torch.Tensor): The attention weights tensor of shape [batch_size, num_heads, qo_len, kv_len].
        window_budget (int): The budget for the window.
        sink_budget (int): The budget for the sink.
        tidal_budget (int): The budget for the tidal.
        
    Returns:
        torch.Tensor: The computed tidal mask.
    """
    # Compute the tidal mask based on the given budgets
    # This is a placeholder implementation. Replace with actual logic.
    
    assert merge_type in ["union", "average"], "merge_type should be either 'union' or 'average'"
    
    tidal_mask = None
    
    device = attn_mask.device
    dtype = attn_mask.dtype
    min_dtype = torch.finfo(dtype).min
    bsz, num_heads, qo_len, kv_len = attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2], attn_weights.shape[3]
    original_mask = (attn_mask != min_dtype)
    
    """
    Window mask example:
    kv_len = 10, qo_len = 4, window_budget = 2
    
    window_mask:
    tensor([[0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]])
            
    Sink mask example:
    kv_len = 10, qo_len = 4, sink_budget = 2, left_padding_len[request] = 3
    
    sink_mask for request:
                
    tensor([[0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.]])
    """
    
    
    left_padding_len = (original_mask.shape[-1] - original_mask.sum(dim=-1)[:, :, -1]).view(bsz, 1, 1, 1) # shape [bsz, 1, 1, 1]
    window_ub = torch.arange(kv_len-qo_len, kv_len, device=device).view(1, 1, qo_len, 1).repeat(bsz, 1, 1, 1) # shape [bsz, 1, qo_len, 1]
    window_sink_mask = torch.arange(kv_len, device=device).repeat(bsz, 1, qo_len, 1)
    window_sink_mask = ((window_sink_mask <= window_ub) & (window_sink_mask > (window_ub - window_budget))) | ((window_sink_mask >= left_padding_len) & (window_sink_mask < (left_padding_len+sink_budget)))
    
    tidal_mask_shape = (bsz, 1, qo_len, kv_len)
    
    if merge_type == "average":
        
        attn_weights = attn_weights.mean(dim=1, keepdim=True)
        attn_weights[window_sink_mask] = 1.
        _, topk_indices = torch.topk(attn_weights, tidal_budget+sink_budget+window_budget, dim=-1) # shape [bsz, 1, qo_len, token_budget]
        
    elif merge_type == "union":
        
        attn_weights[window_sink_mask.repeat(1, num_heads, 1, 1)] = 1.
        _, topk_indices = torch.topk(attn_weights, tidal_budget+sink_budget+window_budget, dim=-1) # shape [bsz, num_heads, qo_len, token_budget]
        topk_indices = topk_indices.permute(0, 2, 3, 1).reshape(bsz * qo_len, -1) # shape [bsz * qo_len, token_budget * num_heads]
        topk_indices = find_union(topk_indices, budget=tidal_budget+sink_budget+window_budget).reshape(bsz, 1, qo_len, tidal_budget+sink_budget+window_budget)

    tidal_mask = torch.zeros(tidal_mask_shape, dtype=torch.bool, device=device)
    tidal_mask.scatter_(-1, topk_indices, True)
    tidal_mask = tidal_mask & original_mask
    
    return tidal_mask