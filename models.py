import math
from typing import Type, Optional

import torch


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dims: int=32) -> None:
        """
        Transformer positional embedding.
        """
        super().__init__()
        self._dims = dims
        self.register_buffer('_k', torch.arange(0, self._dims//2, 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        k = self._k.expand(t.shape[0], -1)
        w = torch.exp(k*-math.log(10000)/self._dims)
        p = torch.stack([torch.sin(t*w), torch.cos(t*w)], dim=-1).flatten(start_dim=1)
        return p


class ChannelAttention(torch.nn.Module):
    def __init__(self, channels: int, reduction_ratio: int=16) -> None:
        super().__init__()
        self._channels = channels
        self._reduction_ratio = reduction_ratio
        self.mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self._channels, self._channels // self._reduction_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(self._channels // self._reduction_ratio, self._channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.nn.functional.avg_pool2d(
            input=x,
            kernel_size=(x.size(2), x.size(3)),
            stride=(x.size(2), x.size(3))
        )
        avg_pool = self.mlp(avg_pool)

        max_pool = torch.nn.functional.max_pool2d(
            input=x,
            kernel_size=(x.size(2), x.size(3)),
            stride=(x.size(2), x.size(3))
        )
        max_pool = self.mlp(max_pool)

        channel_attn = avg_pool + max_pool

        scale = torch.sigmoid(channel_attn).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale * x


class NonBottleneck2D(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            bias: bool = True,
            norm_layer: bool = True,
            res_h: float = 1.0,
            use_cattn: bool = False,
            act_fn: Type[torch.nn.Module] = torch.nn.ReLU
        ) -> None:
        """
        NonBottleneck2D: x + h*conv(relu(conv(x)))
        """
        super().__init__()
        self._res_h = res_h
        self._use_cattn = use_cattn

        self.conv3_1 = torch.nn.Conv2d(
            in_channels, in_channels, (3, 3), stride=1, padding=1, bias=bias
        )
        self.norm1 = (
            torch.nn.GroupNorm(8, in_channels) if norm_layer else torch.nn.Identity()
        )
        if not bias:
            self.norm1.register_parameter("bias", None)  # remove bias from norm layer

        self.conv3_2 = torch.nn.Conv2d(
            in_channels, in_channels, (3, 3), stride=1, padding=1, bias=bias
        )

        self.norm2 = (
            torch.nn.GroupNorm(8, in_channels) if norm_layer else torch.nn.Identity()
        )
        if not bias:
            self.norm2.register_parameter("bias", None)  # remove bias norm layer

        self._act_fn = act_fn(inplace=True)
        self.cattn = ChannelAttention(in_channels) if self._use_cattn else torch.nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv3_1(input)
        output = self.norm1(output)
        output = self._act_fn(output)

        output = self.conv3_2(output)
        output = self.norm2(output)

        output = self.cattn(output)

        return input + self._res_h * output


class DownBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_dims: int,
            bias: bool = True,
            use_norm_layer: bool = True,
            res_h: float = 1.0,
            use_cattn: bool = False,
            use_dropout: bool = False,
            dropout_p: float = 0.1,
            act_fn: Type[torch.nn.Module] = torch.nn.ReLU
    ) -> None:
        super().__init__()
        self._hidden_dims = hidden_dims
        self._bias = bias
        self._use_norm_layer = use_norm_layer
        self._res_h = res_h
        self._use_cattn = use_cattn
        self._use_dropout = use_dropout
        self._dropout_p = dropout_p
        self._act_fn = act_fn

        self._positional_embedding = PositionalEmbedding(dims=self._hidden_dims)
        self._dropout = torch.nn.Dropout(p=dropout_p) if self._use_dropout else torch.nn.Identity()
        self._resblock_1 = NonBottleneck2D(
            in_channels=self._hidden_dims,
            bias=self._bias,
            norm_layer=self._use_norm_layer,
            res_h=self._res_h,
            use_cattn=self._use_cattn,
            act_fn=self._act_fn,
        )
        self._resblock_2 = NonBottleneck2D(
            in_channels=self._hidden_dims,
            bias=self._bias,
            norm_layer=self._use_norm_layer,
            res_h=self._res_h,
            use_cattn=self._use_cattn,
            act_fn=self._act_fn,
        )
        self._sconv = torch.nn.Conv2d(
            1 * self._hidden_dims,
            2 * self._hidden_dims,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=self._bias,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_embed = self._positional_embedding(t).unsqueeze(-1).unsqueeze(-1).to(x.dtype)
        x = self._resblock_1(x + t_embed)
        x = self._dropout(x)
        x = self._resblock_2(x + t_embed)
        x2 = self._sconv(x)
        return x2, x


class UpBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_dims: int,
            bias: bool = True,
            norm_layer: bool = True,
            res_h: float = 1.0,
            use_cattn: bool = False,
            use_dropout: bool = False,
            dropout_p: float = 0.1,
            act_fn: Type[torch.nn.Module] = torch.nn.ReLU
    ) -> None:
        super().__init__()
        self._hidden_dims = hidden_dims
        self._bias = bias
        self._norm_layer = norm_layer
        self._res_h = res_h
        self._use_cattn = use_cattn
        self._use_dropout = use_dropout
        self._dropout_p = dropout_p
        self._act_fn = act_fn

        self._upsample = torch.nn.UpsamplingNearest2d(scale_factor=2.0)
        self._upsample_conv = torch.nn.Conv2d(
            in_channels=self._hidden_dims,
            out_channels=self._hidden_dims//2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            bias=self._bias
        )
        self._conv = torch.nn.Conv2d(
            in_channels=self._hidden_dims,
            out_channels=self._hidden_dims//2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self._bias
        )
        self._positional_embedding = PositionalEmbedding(dims=self._hidden_dims//2)
        self._resblock_1 = NonBottleneck2D(
            in_channels=self._hidden_dims//2,
            bias=self._bias,
            norm_layer=self._norm_layer,
            res_h=self._res_h,
            use_cattn=self._use_cattn,
            act_fn=self._act_fn,
        )
        self._resblock_2 = NonBottleneck2D(
            in_channels=self._hidden_dims//2,
            bias=self._bias,
            norm_layer=self._norm_layer,
            res_h=self._res_h,
            use_cattn=self._use_cattn,
            act_fn=self._act_fn,
        )
        self._dropout = torch.nn.Dropout(p=dropout_p) if self._use_dropout else torch.nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self._upsample(x)
        x = self._upsample_conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self._conv(x)
        t_embed = self._positional_embedding(t).unsqueeze(-1).unsqueeze(-1).to(x.dtype)
        x = self._resblock_1(x + t_embed)
        x = self._dropout(x)
        x = self._resblock_2(x + t_embed)
        return x


class Bottleneck(torch.nn.Module):
    def __init__(
            self,
            hidden_dims: int,
            bias: bool = True,
            use_norm_layer: bool = True,
            res_h: float = 1.0,
            use_cattn: bool = False,
            use_dropout: bool = False,
            dropout_p: float = 0.1,
            act_fn: Type[torch.nn.Module] = torch.nn.ReLU,
    ) -> None:
        super().__init__()
        self._hidden_dims = hidden_dims
        self._bias = bias
        self._use_norm_layer = use_norm_layer
        self._res_h = res_h
        self._use_cattn = use_cattn
        self._use_dropout = use_dropout
        self._dropout_p = dropout_p
        self._act_fn = act_fn

        self._positional_embedding = PositionalEmbedding(dims=self._hidden_dims)
        self._resblock_1 = NonBottleneck2D(
            in_channels=self._hidden_dims,
            bias=self._bias,
            norm_layer=self._use_norm_layer,
            res_h=self._res_h,
            use_cattn=self._use_cattn,
            act_fn=self._act_fn,
        )
        self._resblock_2 = NonBottleneck2D(
            in_channels=self._hidden_dims,
            bias=self._bias,
            norm_layer=self._use_norm_layer,
            res_h=self._res_h,
            use_cattn=self._use_cattn,
            act_fn=self._act_fn,
        )
        self._dropout = torch.nn.Dropout(p=dropout_p) if self._use_dropout else torch.nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self._positional_embedding(t).unsqueeze(-1).unsqueeze(-1).to(x.dtype)
        x = self._resblock_1(x + t_embed)
        x = self._dropout(x)
        x = self._resblock_2(x + t_embed)
        return x


class ResUNet(torch.nn.Module):
    """
    Implements a U-net inspired denoising autoencoder using non-bottleneck robust residual blocks
    with step factor `h < 1` as proposed by Zhang et al. (IJCAI2019).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = 32,
        stages: int = 2,
        residual_step_h: float = 0.5,
        bias: bool = True,
        use_norm_layer: bool = True,
        use_attention: bool = False,
        use_dropout: bool = False,
        dropout_p: float = 0.1,
        act_fn: Type[torch.nn.Module] = torch.nn.ReLU
    ) -> None:
        """
        Initializes ResUNet.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param hidden_channels: Number of hidden channels (width of the network).
        :param residual_step_h: Step factor in the residual blocks, see Zhang et al. (IJCAI2019).
        :param bias: Use bias in all convolutions.
        :param use_norm_layer: Use normalization layer in the residual blocks.
        :param use_attention: Use channel attention in the residual blocks.
        :param use_dropout: Use dropout in the residual blocks. Also see `dropout_p`
        :param dropout_p: Dropout probability used if `use_dropout=True`.
        :param act_fn: Activation function for the residual blocks.
        """
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels if out_channels else in_channels
        self._hidden_dims = hidden_channels
        self._stages = stages
        self._res_h = residual_step_h
        self._bias = bias
        self._use_norm_layer = use_norm_layer
        self._use_cattn = use_attention
        self._use_dropout = use_dropout
        self._dropout_p = dropout_p
        self._act_fn = act_fn

        self._init_conv = torch.nn.Conv2d(
            self._in_channels,
            self._hidden_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self._bias,
        )
        self._downs = torch.nn.ModuleList([
            DownBlock(
                2**i * self._hidden_dims,
                self._bias,
                self._use_norm_layer,
                self._res_h,
                self._use_cattn,
                self._use_dropout,
                self._dropout_p,
                self._act_fn
            ) for i in range(self._stages)
        ])
        self._bottleneck = Bottleneck(
            2**self._stages * self._hidden_dims,
            self._bias,
            self._use_norm_layer,
            self._res_h,
            self._use_cattn,
            self._use_dropout,
            self._dropout_p
        )
        self._ups = torch.nn.ModuleList([
            UpBlock(
                2*2**i * self._hidden_dims,
                self._bias,
                self._use_norm_layer,
                self._res_h,
                self._use_cattn,
                self._use_dropout,
                self._dropout_p,
                self._act_fn
            ) for i in reversed(range(self._stages))
        ])
        self._final_conv = torch.nn.Conv2d(
            self._hidden_dims,
            self._out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self._bias,
        )

    def forward(self, x: torch.Tensor, t: torch.IntTensor) -> torch.Tensor:
        x = self._init_conv(x)
        t = t.to(x.dtype)

        skips = []
        for down in self._downs:
            x, skip = down(x, t)
            skips.append(skip)
        skips = skips[::-1]

        # bottleneck
        x = self._bottleneck(x, t)

        for i, up in enumerate(self._ups):
            x = up(x, skips[i], t)

        x = self._final_conv(x)

        return x


def compute_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
        t = torch.arange(timesteps)
        f_0 = torch.cos((torch.FloatTensor([0])+s)/(1+s)*math.pi/2)**2
        f_t = torch.cos((t/timesteps+s)/(1+s)*math.pi/2)**2
        a = f_t / f_0
        return a


class DDPM(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        timesteps: int = 1000,
        beta_schedule: str = "ho",
        loss_fn: torch.nn.Module = torch.nn.MSELoss()
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.loss_fn = loss_fn

        if self.beta_schedule == "ho":
            beta = torch.linspace(1e-4, 0.02, self.timesteps, dtype=torch.float32)
            beta = beta[:, None, None, None]
            alpha = 1 - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
        elif self.beta_schedule == "nichol":
            alpha_bar = compute_alpha_bar(self.timesteps+1)[:, None, None, None]
            beta = 1-(alpha_bar[1:]/alpha_bar[:-1])
            beta = beta.clamp_max(0.999)
            alpha_bar = alpha_bar[1:]
            alpha = 1 - beta
        else:
            raise ValueError("`beta_schedule` not in ['ho', 'nichol']")

        sigma = torch.sqrt(beta)
        sigma[0] = 0

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("inv_sqrt_alpha", 1/torch.sqrt(alpha))
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_over_sqrt_alphabar", (1-alpha)/torch.sqrt(1-alpha_bar))
        self.register_buffer("sigma", sigma)

    def training_step(self, x_0: torch.Tensor):
        """
        This method is used in training. `t` will be randomly sampled.
        """
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), dtype=torch.int32, device=self.device)
        eps = torch.randn_like(x_0)
        x_t = torch.sqrt(self.alpha_bar[t])*x_0 + torch.sqrt(1-self.alpha_bar[t])*eps
        eps_pred = self.model(x_t, t.unsqueeze(-1))
        return self.loss_fn(eps_pred, eps)

    @torch.inference_mode()
    def validate(self, x_0: torch.Tensor) -> float:
        loss_sum = 0.0
        for t in reversed(range(0, self.timesteps)):
            t = torch.IntTensor([t]).expand(x_0.shape[0])
            eps = torch.randn_like(x_0)

            x_t = torch.sqrt(self.alpha_bar[t])*x_0 + torch.sqrt(1-self.alpha_bar[t])*eps
            eps_pred = self.model(x_t, t.unsqueeze(-1).to(self.device))

            x_t_1 = self.inv_sqrt_alpha[t]*(x_t-self.alpha_over_sqrt_alphabar[t]*eps)
            x_t_1_pred = self.inv_sqrt_alpha[t]*(x_t-self.alpha_over_sqrt_alphabar[t]*eps_pred)

            loss_sum += self.loss_fn(x_t_1_pred, x_t_1).item()
        return loss_sum

    @torch.inference_mode()
    def sample(
        self,
        size: torch.Size,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        This method should be used during test time to sample new images.
        """
        x_t = torch.randn(size, dtype=dtype).to(self.device)
        for t in reversed(range(0, self.timesteps)):
            x_t = self.forward(x_t, torch.IntTensor([t]))

        return x_t

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.IntTensor,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if z is None:
            z = torch.randn_like(x_t)
        eps = self.model(x_t, t.expand(x_t.shape[0]).unsqueeze(-1).to(self.device))
        x_t = self.inv_sqrt_alpha[t]*(x_t-self.alpha_over_sqrt_alphabar[t]*eps)
        x_t = x_t + self.sigma[t]*z
        return x_t
