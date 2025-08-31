import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    This method computes the loss of the DINO method. It gets the teacher
    probabilities, which were generated in the centering and softmax
    procedure before. And it also gets the student representations, which
    are softmaxed and transfered to logs to form log probabilities as
    required from the cross entropy:

    loss = -p(teacher) * log(p(student))

    Please note that the pairing teacher_globals, student_locals and
    teacher_globals and student_globals are processed separately (not in
    the same call):

    dino_loss = DINOLoss()

    globals = global_augmentation(image)
    local = local_augmentation(image)

    globals_teacher_repr = teacher(globals)
    globals_student_repr = student(globals)
    locals_student_repr = student(locals)

    globals_teacher_probs = softmax_centering(globals_teacher_repr)

    local_loss = dino_loss(locals_student_repr, globals_teacher_probs)
    global_loss = dino_loss(locals_student_repr, globals_teacher_probs,
                            between_globals=True)
    """
    def __init__(
            self,
            student_temperature: float = 0.1
    ):
        super().__init__()
        self.student_temperature = student_temperature

    def forward(
            self,
            student_out: torch.Tensor,
            teacher_out: torch.Tensor,
            between_globals: bool = False
    ) -> None:
        """
        Computes the loss between the student and teacher representation.

        Parameters
        ----------
        student_out: torch.Tensor
            The student representation as logits.

        teacher_out: torch.Tensor
            The teacher representation as probabilities.

        between_globals: bool
            Excludes the computation between the same globals.

        Returns
        -------
        loss: torch.Tensor
            The loss value computed.
        """
        num_student, b_student, student_dim = student_out.shape
        num_teacher, b_teacher, teacher_dim = teacher_out.shape

        assert b_teacher == b_student, (
            "Please ensure, that the batch sizes of the teacher and student "
            f"tensors are equal. Batch size teacher: {b_teacher}, Batch size "
            f"student: {b_student}"
        )
        assert student_dim == teacher_dim, (
            "Please ensure, that the hidden dim of the teacher and student "
            f"tensors are equal. Hidden dim teacher: {teacher_dim}, Hidden "
            f"dim student: {student_dim}"
        )
        #  create the log of the softmax
        student_logits = F.log_softmax(student_out / self.student_temperature,
                                       dim=-1)
        if between_globals:
            loss = -torch.einsum('t b n, s b n -> s t', student_logits,
                                 teacher_out)
            min_dim = min(num_student, num_teacher)
            loss = torch.diagonal_scatter(loss, src=torch.zeros(min_dim))
            loss = loss.sum() / (
                b_teacher * num_teacher * num_student - b_teacher * min_dim
            )
        else:
            loss = -torch.einsum('t b n, s b n -> ', student_logits,
                                 teacher_out)
            loss = loss / (b_teacher * num_teacher * num_student)
        return loss


if __name__ == "__main__":
    NUM_LOCALS = 8
    NUM_GLOBALS = 2

    BATCH_SIZE = 3
    HIDDEN_DIM = 4

    dino_loss = DINOLoss()

    student_local_repr = torch.randn([NUM_LOCALS, BATCH_SIZE, HIDDEN_DIM])
    student_global_repr = torch.randn([NUM_GLOBALS, BATCH_SIZE, HIDDEN_DIM])
    teacher_global_repr = torch.randn([NUM_GLOBALS, BATCH_SIZE, HIDDEN_DIM])

    teacher_global_repr = F.softmax(teacher_global_repr, dim=-1)

    loss_locals = dino_loss(student_out=student_local_repr,
                            teacher_out=teacher_global_repr)

    loss_globals = dino_loss(student_out=student_global_repr,
                             teacher_out=teacher_global_repr,
                             between_globals=True)
