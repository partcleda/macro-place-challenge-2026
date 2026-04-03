import torch
from torch.nn import Module
from enum import Enum

class InterpolantType(Enum):
    LINEAR = 0
    # POW3 interpolant from: 
    # Don't Start from Scratch: Behavioral Refinement via Interpolant-based Policy Diffusion
    # https://arxiv.org/abs/2402.16075
    POW3 = 1 

class StochasticInterpolant:
    # Stochastic Interpolant framework from: 
    # Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
    # https://arxiv.org/abs/2303.08797

    def __init__(self, 
                 T: int, 
                 interpolant_type: InterpolantType, 
                 v_model: Module,
                 s_model: Module,
                 b_model: Module | None,
                 device: torch.device,
                 train_b: bool = False,
                 c: int = 3,
                 d: float = .3):
        
        self.T = T
        self.interpolant_type = interpolant_type
        self.d = d
        self.c = c
        self.device = device

        if train_b and b_model is None:
            raise RuntimeError("we require a b_model for training")
        self.train_b = train_b

        self.s_model = s_model.to(self.device)
        self.v_model = v_model.to(self.device)
        if self.train_b: 
            self.b_model = b_model.to(self.device)

    # loss

    def loss(self, x0: torch.Tensor, x1: torch.Tensor, obv: torch.Tensor):
        # x0 = prior, xt = target, obv = state/observation
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        obv = obv.to(self.device)
        t = self.uniform_t(x0)
        xt = self.x_t(x0, x1, t)
        z = torch.rand_like(xt)
        xt *= z
        dI_dt = self.dI_dt(x0, x1, t)
        v_loss = self.loss_v(xt, t, obv, dI_dt) 
        s_loss = self.loss_s(xt, t, obv, z) 

        if not self.train_b:
            return v_loss + s_loss
        else:
            b_loss = self.loss_b(xt, t, obv, dI_dt, z)
            return v_loss + s_loss + b_loss

    def loss_v(self, xt: torch.Tensor, t: torch.Tensor, obv: torch.Tensor, dI_dt: torch.Tensor) -> torch.Tensor:
        v = dI_dt + self.gamma(t)
        return ((self.v_model(xt, t, obv) - v) ** 2).mean()

    def loss_s(self, xt: torch.Tensor, t: torch.Tensor, obv: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return ((self.s_model(xt, t, obv) - z) ** 2).mean() 

    def loss_b(self, xt: torch.Tensor, t: torch.Tensor, obv: torch.Tensor, dI_dt: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return ((self.b_model(xt, t, obv) - dI_dt * z) ** 2).mean()
    
    # interpolant

    def x_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.interpolant(x0, x1, t) + self.gamma(t)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        return self.d * torch.sqrt(2 * t * (1 - t))
    
    def interpolant(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.interpolant_type == InterpolantType.LINEAR:
            return (1 - t) * x0 + t * x1
        elif self.interpolant_type == InterpolantType.POW3:
            return (1 - t) ** 3 * x0 + (1 - (1 - t) ** 3) * x1
        else:
            raise NotImplementedError
            
    def dI_dt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.interpolant_type == InterpolantType.LINEAR:
            return x1 - x0
        elif self.interpolant_type == InterpolantType.POW3:
            return 3 * (1 - t) ** 2 * (x1 - x0)
        else:
            raise NotImplementedError
    
    # sampling

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, obv: torch.Tensor, steps: int = 100) -> list[torch.Tensor]:
        x0 = x0.to(self.device)
        xt = x0.clone()
        obv = obv.to(self.device)
        dt: float = 1.0 / steps

        trajectories = []
        for i in range(steps):
            t = torch.full((x0.shape[0], 1), i / steps, device=self.device)
            t = t.view(-1, *([1] * (xt.dim() - 1))).clamp(min = 1e-6, max = 1 - 1e-6)
            v = self.v_model(xt, t, obv)
            s = self.s_model(xt, t, obv)
            z = torch.rand_like(xt)
            eps = self.epsilon(t)
            if self.train_b:
                b = self.b_model(xt, t, obv)
            else:
                b = v - self.gamma_dot(t) * self.gamma(t) * s
            bF = b + eps * s
            xt = xt + bF * dt + torch.sqrt(2 * eps) * z
            trajectories.append(xt)
        return trajectories
    
    def epsilon(self, t: torch.Tensor) -> torch.Tensor:
        return self.c * (1 - t)
    
    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return self.d * (1 - 2 * t) / (torch.sqrt(2 * t * (1 - t)) + 1e-9)
    
    # utils

    def uniform_t(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(x).clamp(min = 1e-6, max = 1 - 1e-6)
        
    def save(self, fp: str) -> None:
        checkpoint = {
            "T": self.T,
            "interpolant_type": self.interpolant_type.value,
            "d": self.d,
            "v_model": self.v_model.state_dict(),
            "s_model": self.s_model.state_dict(),
        }
        if self.train_b:
            checkpoint["b_model"] = self.b_model.state_dict()
        torch.save(checkpoint, fp)

    def load(self, fp: str, map_location=None) -> None:
        checkpoint = torch.load(fp, map_location=map_location)
        self.T = checkpoint["T"]
        self.interpolant_type = InterpolantType(checkpoint["interpolant_type"])
        self.d = checkpoint["d"]
        self.v_model.load_state_dict(checkpoint["v_model"])
        self.s_model.load_state_dict(checkpoint["s_model"])

        self.v_model.to(self.device)
        self.s_model.to(self.device)

        if self.train_b:
            self.b_model.load_state_dict(checkpoint["b_model"])
            self.b_model.to(self.device)
        