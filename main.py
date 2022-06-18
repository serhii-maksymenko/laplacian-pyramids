import cv2
import time
import numpy as np
import torch
from laplacian_pyramid import LaplacianPyramid


A = cv2.imread("data/apple.jpg")
B = cv2.imread("data/orange.jpg")


def pyramid_method1(img, n_levels):
    _img = img
    _prev_img = None
    pyramid = []
    for _ in range(n_levels):
        _img = cv2.pyrDown(_img)
        if _prev_img is not None:
            pyramid.append(cv2.subtract(_prev_img, cv2.pyrUp(_img)))
        _prev_img = _img
    pyramid.append(_img)

    return pyramid


def pyramid_method2(img, n_levels):
    lp = LaplacianPyramid(n_levels=n_levels).to(dtype=torch.float)

    _img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0)
    pyramid = lp(_img)
    pyramid = [
        p.detach().cpu().numpy()[0].astype(np.uint8).transpose(1, 2, 0)
        for p in pyramid
    ]

    return pyramid


def blend(pyramid_A, pyramid_B):
    pyramid_A.reverse()
    pyramid_B.reverse()

    _blended_result = None
    for pA, pB in zip(pyramid_A, pyramid_B):
        rows, cols, dpt = pA.shape
        _stacked_image = np.hstack((pA[:, 0:cols//2], pB[:, cols//2:]))
        if _blended_result is None:
            _blended_result = _stacked_image
        else:
            _blended_result = cv2.pyrUp(_blended_result)
            _blended_result = cv2.add(_blended_result, _stacked_image)

    return _blended_result


p1 = pyramid_method1(A, n_levels=2)
p2 = pyramid_method2(A, n_levels=2)
p3 = cv2.vconcat([p1[0], p2[0]])
the_same = cv2.absdiff(p1[0], p2[0]).max() == 0


pyramid_A = pyramid_method1(A, n_levels=6)
pyramid_B = pyramid_method1(B, n_levels=6)
_blended_result1 = blend(pyramid_A, pyramid_B)

pyramid_A = pyramid_method2(A, n_levels=6)
pyramid_B = pyramid_method2(B, n_levels=6)
_blended_result2 = blend(pyramid_A, pyramid_B)

cv2.putText(_blended_result1, 'OpenCV', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(_blended_result2, 'PyTorch', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

result_image = cv2.hconcat([
    _blended_result1,
    _blended_result2
])

cv2.imwrite('data/result.jpg', result_image)
cv2.imshow('image', result_image)
_ = cv2.waitKey(0)
