import torch
import torch.nn as nn
from torch.nn import functional as F


class IBOTLoss(nn.Module):

    def __init__(self, student_temperature: float = 0.1) -> None:
        super().__init__()
        self.student_temperature = student_temperature

    def forward(
        self,
        student_global_patches: torch.Tensor,
        teacher_global_patches: torch.Tensor,
        student_mask_patches: torch.Tensor,
    ) -> torch.Tensor:
        NUM_STUDENT_CROPS, B, T, DIM = student_global_patches.shape
        NUM_TEACHER_CROPS, B, T, DIM = teacher_global_patches.shape
        assert NUM_TEACHER_CROPS == NUM_STUDENT_CROPS, (
            f"Sorry, the number of globals between student "
            f"({NUM_STUDENT_CROPS}) and teacher ({NUM_TEACHER_CROPS}) is not "
            f"equal!"
        )

        teacher_global_patches = torch.reshape(
            teacher_global_patches, shape=[NUM_TEACHER_CROPS * B, T, DIM]
        )
        student_global_patches = torch.reshape(
            student_global_patches, shape=[NUM_STUDENT_CROPS * B, T, DIM]
        )
        BATCH_MASK, T = student_mask_patches.shape
        assert BATCH_MASK == NUM_TEACHER_CROPS * B, (
            "Sorry, the size of the "
            "mask batch does not"
            "correspond with the "
            "NUM_TEACHER_CROPS * B"
        )
        assert BATCH_MASK == NUM_STUDENT_CROPS * B, (
            "Sorry, the size of the "
            "mask batch does not"
            "correspond with the "
            "NUM_STUDENT_CROPS * B"
        )

        student_global_patches = (
            student_global_patches * student_mask_patches.unsqueeze(-1).float()
        )
        teacher_global_patches = (
            teacher_global_patches * student_mask_patches.unsqueeze(-1).float()
        )

        loss = -teacher_global_patches * F.log_softmax(
            student_global_patches / self.student_temperature, dim=-1
        )

        loss = torch.sum(
            loss * student_mask_patches.float().unsqueeze(-1), dim=[-2, -1]
        ) / student_mask_patches.sum(dim=-1).clamp(min=1.0)
        return loss.mean()


if __name__ == "__main__":
    NUM_GLOBALS = 2

    T = 5
    BATCH_SIZE = 3
    HIDDEN_DIM = 4

    ibot_loss = IBOTLoss()

    student_mask = torch.randn([BATCH_SIZE * NUM_GLOBALS, T]) < 0.5
    student_global_repr = torch.randn([NUM_GLOBALS, BATCH_SIZE, T, HIDDEN_DIM])
    teacher_global_repr = torch.randn([NUM_GLOBALS, BATCH_SIZE, T, HIDDEN_DIM])

    teacher_global_repr = F.softmax(teacher_global_repr, dim=-1)

    loss_globals = ibot_loss(
        student_global_patches=student_global_repr,
        teacher_global_patches=teacher_global_repr,
        student_mask_patches=student_mask,
    )

    print(loss_globals)
