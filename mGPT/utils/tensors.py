import torch


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    lenbatch = [len(b[0][0][0]) for b in batch]

    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    # x - [bs, njoints, nfeats, lengths]
    #   - nfeats, the representation of a joint
    # y - [bs]
    # mask - [bs, lengths]
    # lengths - [bs]
    batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, 'lengths': lenbatchTensor}
    return batch


# slow version with padding
def collate_data3d_slow(batch):
    batchTensor = {}
    for key in batch[0].keys():
        databatch = [b[key] for b in batch]
        batchTensor[key] = collate_tensors(databatch)
    batch = batchTensor
    # theta - [bs, lengths, 85], theta shape (85,)
    #       - (np.array([1., 0., 0.]), pose(72), shape(10)), axis=0)
    # kp_2d - [bs, lengths, njoints, nfeats], nfeats (x,y,weight)
    # kp_3d - [bs, lengths, njoints, nfeats], nfeats (x,y,z)
    # w_smpl - [bs, lengths] zeros
    # w_3d - [bs, lengths] zeros
    return batch

def collate_data3d(batch):
    batchTensor = {}
    for key in batch[0].keys():
        databatch = [b[key] for b in batch]
        if key == "paths":
            batchTensor[key] = databatch
        else:    
            batchTensor[key] = torch.stack(databatch,axis=0)
    batch = batchTensor
    # theta - [bs, lengths, 85], theta shape (85,)
    #       - (np.array([1., 0., 0.]), pose(72), shape(10)), axis=0)
    # kp_2d - [bs, lengths, njoints, nfeats], nfeats (x,y,weight)
    # kp_3d - [bs, lengths, njoints, nfeats], nfeats (x,y,z)
    # w_smpl - [bs, lengths] zeros
    # w_3d - [bs, lengths] zeros
    return batch
