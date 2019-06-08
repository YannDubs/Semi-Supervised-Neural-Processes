

def random_masker(batch_size, mask_shape,
                  min_nnz=.1,
                  max_nnz=.5,
                  is_batch_repeat=False):
    """
    Return a random context mask. The number of non zero values will be in
    [min_nnz, max_nnz]. If min_perc, max_perc are smaller than 1 these represent
    a percentage of points. If `is_batch_repeat` use the same mask for all elements
    in the batch.
    """
    n_possible_points = prod(mask_shape)
    min_nnz = ratio_to_int(min_nnz, n_possible_points)
    max_nnz = ratio_to_int(max_nnz, n_possible_points)
    n_nnz = random.randint(min_nnz, max_nnz)

    if is_batch_repeat:
        mask = torch.zeros(n_possible_points).byte()
        mask[torch.randperm(n_possible_points)[:n_nnz]] = 1
        mask = mask.view(*mask_shape).contiguous()
        mask = mask.unsqueeze(0).expand(batch_size, *mask_shape)
    else:
        mask = np.zeros((batch_size, n_possible_points), dtype=np.uint8)
        mask[:, :n_nnz] = 1
        indep_shuffle(mask, -1)
        mask = torch.from_numpy(mask)
        mask = mask.view(batch_size, *mask_shape).contiguous()

    return mask


def masker_composes(maskers):
    """Return a masker composed of a list of maskers."""
    def masker_coposed(batch_size, mask_shape):
        mask = no_masker(batch_size, mask_shape)
        for masker in maskers:
            mask *= masker(batch_size, mask_shape)
    return mask


def half_masker(batch_size, mask_shape, dim=0):
    """Return a mask which masks the top half features of `dim`."""
    mask = torch.zeros((batch_size, *mask_shape)).byte()
    slcs = [slice(None)] * (len(mask_shape))
    slcs[dim] = slice(0, mask_shape[dim] // 2)
    mask[[slice(None)] + slcs] = 1
    return mask


def no_masker(batch_size, mask_shape):
    """Return a mask of all 1."""
    mask = torch.ones((batch_size, *mask_shape)).byte()
    return mask
