import torch


class LaplacianPyramid(torch.nn.Module):

    def __init__(self, n_levels):
        super().__init__()
        self.reduce_blur_conv = torch.nn.Conv2d(3, 3, 5, stride=2, padding=2, padding_mode='reflect', groups=3,
                                                bias=False)
        self.expand_unpool_conv = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, groups=3, bias=False)
        self.expand_blur_conv = torch.nn.Conv2d(3, 3, 5, padding=2, padding_mode='reflect', groups=3, bias=False)
        blur_kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float)
        self.reduce_blur_conv.weight.data[::, 0] = blur_kernel / 256.0
        self.expand_unpool_conv.weight.data[::, 0] = torch.tensor([[1, 0], [0, 0]], dtype=torch.float)
        self.expand_blur_conv.weight.data[::, 0] = (4 * blur_kernel) / 256.0

        self.n_levels = n_levels

    def forward(self, x):
        _img = x
        _prev_img = None
        pyramid = []
        for _ in range(self.n_levels):
            _img = self.reduce_blur_conv(_img)
            if _prev_img is not None:
                _img_expanded = self.expand_unpool_conv(_img)
                _img_expanded = self.expand_blur_conv(_img_expanded)
                p = torch.relu(torch.subtract(_prev_img, _img_expanded))
                pyramid.append(p)
            _prev_img = _img
        pyramid.append(_img)

        return pyramid
