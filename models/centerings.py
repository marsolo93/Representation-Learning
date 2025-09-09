import torch
import torch.nn as nn

# TODO: maybe it is better to implement a abstract class and to inherit from it


class EMAPatchCentering(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.center_momentum = center_momentum
        self.center = nn.Parameter(torch.zeros([1, 1, hidden_dim]))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == self.center.ndim, (
            "Sorry, please ensure that the number of dimensions between the "
            "input tensor and the output centering tensor is the same."
        )
        return x - self.center

    @torch.no_grad()
    def update_center(self, x: torch.Tensor) -> None:
        self.center = self.center * self.center_momentum + x * (
            1 - self.center_momentum
        )


class EMAClassCentering(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.center_momentum = center_momentum
        self.center = nn.Parameter(torch.zeros([1, hidden_dim]))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == self.center.ndim, (
            "Sorry, please ensure that the number of dimensions between the "
            "input tensor and the output centering tensor is the same."
        )
        return x - self.center

    @torch.no_grad()
    def update_center(self, x: torch.Tensor) -> None:
        self.center = self.center * self.center_momentum + x * (
            1 - self.center_momentum
        )


class SinkhornKnopp(nn.Module):
    """
    This module forms from the class and patch token
    doubly stochastic representatives. This means, it scales the rows and the
    columns to 1.

    sum_i(x_ij) = sum_j(x_ij) = 1

    This is achieved in an iterative procedure.

    References
    ----------
    [1] - https://en.wikipedia.org/wiki/Doubly_stochastic_matrix#
    [2] - https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem
    """

    def __init__(self, n_iterations: int = 3) -> None:
        """
        Constructs the module, which can be employed for the class and patch
        tokens.

        Parameters
        ----------
        n_iterations: int
            Number of iterations to achieve the conditions stated above.
        """
        super().__init__()
        self.n_iterations = n_iterations

    def forward(
        self,
        teacher_token: torch.Tensor,
        teacher_temp: float,
        mask: torch.Tensor | None = None,
        reshape_original_shape: bool = True,
    ):
        """
        Computes the doubly stochastic matrices from the class or patch tokens.
        It expects the parallel computing shape:

        For class tokens:
            [NUM_CROPS * BATCH_SIZE, HIDDEN_DIM]
        For patch tokens:
            [NUM_CROPS * BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_DIM]

        Parameters
        ----------
        teacher_token: torch.Tensor
            The teacher tokens, which should be transformed to doubly
            stochastic matrices.

        teacher_temp: float
            The float number for the sharpness of the distributions.

        mask: torch.Tensor
            The mask for the case for processing of the patch tokens case.

        reshape_original_shape: bool
            An option for the patch tokens processing. It determines, whether
            the original shape should be regenerated.

        Returns
        -------
        teacher_token: torch.Tensor
            The doubly stochastic representations of the teacher tokens.
        """
        teacher_token = teacher_token.float()
        original_ndim = teacher_token.ndim

        if original_ndim == 3:
            assert mask is not None, (
                "You need to specify a mask pattern if you want to use "
                "Sinkhorn Knopp on patches."
            )
            original_batchsize, sequence_length = (
                teacher_token.shape[0],
                teacher_token.shape[1],
            )
            teacher_token = teacher_token.flatten(0, 1)
            hidden_dim = teacher_token.shape[-1]
            batch_num = torch.sum(mask)
            teacher_token = torch.exp(
                teacher_token / teacher_temp
            ) * mask.float().unsqueeze(-1)

        elif original_ndim == 2:
            hidden_dim = teacher_token.shape[-1]
            batch_num = teacher_token.shape[0]
            teacher_token = torch.exp(teacher_token / teacher_temp)
            original_batchsize = None
            sequence_length = None
        else:
            raise NotImplementedError(
                f"The dimensionality of {teacher_token.ndim}"
                f"is not supported for the Sinkhorn Knopp "
                f"scaling."
            )

        teacher_token = teacher_token.t()
        teacher_token = teacher_token / torch.sum(teacher_token)
        for i in range(self.n_iterations):
            batch_sum = torch.sum(teacher_token, dim=1, keepdim=True).clamp(
                1e-16
            )

            teacher_token = teacher_token / batch_sum

            teacher_token = teacher_token / hidden_dim

            dim_sum = torch.sum(teacher_token, dim=0, keepdim=True).clamp(
                1e-16
            )

            teacher_token = teacher_token / dim_sum

            teacher_token = teacher_token / batch_num

        teacher_token = teacher_token * batch_num
        teacher_token = teacher_token.t()
        if original_ndim == 3 and reshape_original_shape:
            teacher_token = teacher_token.unflatten(
                0, [original_batchsize, sequence_length]
            )

        return teacher_token


if __name__ == "__main__":
    BATCH_SIZE = 2
    SEQ_LENGTH = 3
    HIDDEN_DIM = 4

    teacher_out_patches = torch.randn([BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM])

    mask = (torch.randn([BATCH_SIZE * SEQ_LENGTH]) < 0.5).bool()

    sink_horn = SinkhornKnopp(n_iterations=5)
    result = sink_horn(
        teacher_token=teacher_out_patches,
        teacher_temp=0.5,
        mask=mask,
        reshape_original_shape=False,
    )

    print(f"REUSLTS WITH PATCH TOKENS: {result}")
    print(f"PATCH TOKENS ROW SUM: {torch.sum(result, dim=0)}")
    print(f"PATCH TOKENS COL SUM: {torch.sum(result, dim=1)}")

    teacher_out_patches = torch.randn([BATCH_SIZE, HIDDEN_DIM])

    result = sink_horn(teacher_token=teacher_out_patches, teacher_temp=0.3)

    print(f"REUSLTS WITH CLS TOKENS: {result}")
    print(f"CLS TOKENS ROW SUM: {torch.sum(result, dim=0)}")
    print(f"CLS TOKENS COL SUM: {torch.sum(result, dim=1)}")
