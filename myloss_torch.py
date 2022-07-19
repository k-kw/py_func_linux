import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class SSIMLoss(Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5, gray: bool = False) -> None:

        """Computes the structural similarity (SSIM) index map between two images.

        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """

        super().__init__()
        if(gray):
            self.channels=1
        else:
            self.channels=3
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

        

    def forward(self, x: Tensor, y: Tensor, as_loss: bool = True) -> Tensor:

        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:

        # Compute means
        #kernel_sizeが奇数であれば、xと同じサイズのuxが帰ってくる、各ウィンドウの平均が返ってくる
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)


        # Compute variances
        #２乗の平均を求める
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    #実用上SSIM計算前にガウシアンフィルタで平滑化
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:

        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        
        #定数を除いて１次元のガウス分布
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        
        #総和が１になるようにスケーリング
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        #列、行どちらからも見てもガウス分布になるように２次元化,kernel_2dの総和も１
        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)

        #channel分複製
        kernel_2d = kernel_2d.expand(self.channels, 1, kernel_size, kernel_size).contiguous()

        #kernel_2dは、２次元ガウス分布に従った重みの2次元行列
        return kernel_2d

