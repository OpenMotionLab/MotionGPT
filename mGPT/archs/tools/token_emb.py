
from torch import Tensor, nn

class NewTokenEmb(nn.Module):
    """
    For adding new tokens to a pretrained model
    """

    def __init__(self,
                 old_embeddings: nn.Embedding,
                 new_num_tokens: int = None) -> None:

        super().__init__()

        self.num_tokens = old_embeddings.num_embeddings + new_num_tokens
        self.old_num_tokens = old_embeddings.num_embeddings
        self.new_num_tokens = new_num_tokens
        self.embedding_dim = old_embeddings.embedding_dim

        # For text embeddings
        self.text_embeddings = nn.Embedding(
            self.num_tokens,
            self.embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype)
        with torch.no_grad():
            self.text_embeddings.weight.data[:old_embeddings.
                                             num_embeddings] = old_embeddings.weight.data
            self.text_embeddings.weight.data[
                self.old_num_tokens:] = torch.zeros(
                    self.new_num_tokens,
                    self.embedding_dim,
                    dtype=old_embeddings.weight.dtype,
                    device=old_embeddings.weight.device)
        self.text_embeddings.weight.requires_grad_(False)

        # For motion embeddings
        self.motion_embeddings = nn.Embedding(
            new_num_tokens,
            self.embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype)
        with torch.no_grad():
            self.motion_embeddings.weight.data[:self.
                                               old_num_tokens] = torch.zeros(
                                                   new_num_tokens,
                                                   self.embedding_dim,
                                                   dtype=old_embeddings.weight.
                                                   dtype,
                                                   device=old_embeddings.
                                                   weight.device)
        self.word2motionProj = nn.Linear(self.old_num_tokens, new_num_tokens)

    def forward(self, input: Tensor) -> Tensor:

        with torch.no_grad():
            self.motion_embeddings.weight.data[:self.
                                               old_num_tokens] = torch.zeros(
                                                   self.new_num_tokens,
                                                   self.embedding_dim,
                                                   dtype=self.motion_embeddings
                                                   .weight.dtype,
                                                   device=self.
                                                   motion_embeddings.weight.
                                                   device)

        self.motion_embeddings.weight.data[
            self.old_num_tokens:] = self.word2motionProj(
                self.text_embeddings.weight.data[:self.old_num_tokens].permute(
                    1, 0)).permute(1, 0)

        return self.text_embeddings(input) + self.motion_embeddings(input)

