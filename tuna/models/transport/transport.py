# pyre-unsafe
import enum
import math
from typing import Any, Callable

import numpy as np
import torch as th

from tuna.models.transport import path
from tuna.models.transport.integrators import ode, sde
from tuna.models.transport.utils import mean_flat


class ModelType(enum.Enum):
    """Which type of output the model predicts."""

    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    """Which type of path to use."""

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """Which type of weighting to use."""

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        snr_type,
        do_shift,
        seq_len,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        self.snr_type = snr_type
        self.do_shift = do_shift
        self.seq_len = seq_len

    def prior_logp(self, z):
        """Standard multivariate normal prior. Assumes z is batched."""
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0
        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):
            t0 = (
                eps
                if (diffusion_form == "SBDM" and sde)
                or self.model_type != ModelType.VELOCITY
                else 0
            )
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1, max_t0=None):
        """Sample x0 & t based on the shape of x1."""
        if isinstance(x1, (list, tuple)):
            x0 = [th.randn_like(img_start) for img_start in x1]
        else:
            x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if max_t0 is not None:
            t0 = max_t0

        if self.snr_type.startswith("uniform"):
            assert t0 == 0.0 and t1 == 1.0, "not implemented."
            if "_" in self.snr_type:
                _, t0, t1 = self.snr_type.split("_")
                t0, t1 = float(t0), float(t1)
            t = th.rand((len(x1),)) * (t1 - t0) + t0
        elif self.snr_type == "lognorm":
            u = th.normal(mean=0.0, std=1.0, size=(len(x1),))
            t = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0
        else:
            raise NotImplementedError("Not implemented snr_type %s" % self.snr_type)

        if self.do_shift:
            base_shift: float = 0.5
            max_shift: float = 1.15
            mu = self.get_lin_function(y1=base_shift, y2=max_shift)(self.seq_len)
            t = self.time_shift(mu, 1.0, t)
        t = t.to(x1[0])
        return t, x0, x1

    def time_shift(self, mu: float, sigma: float, t: th.Tensor):
        t = 1 - t
        t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
        t = 1 - t
        return t

    def get_lin_function(
        self, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def training_losses(self, model, x1, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        if "cond" in model_kwargs:
            conds = model_kwargs.pop("cond")
            xt = [
                th.cat([x, cond], dim=0) if cond is not None else x
                for x, cond in zip(xt, conds)
            ]
        model_output = model(xt, t, **model_kwargs)
        B = len(x0)

        terms = {}
        if self.model_type == ModelType.VELOCITY:
            if isinstance(x1, (list, tuple)):
                assert len(model_output) == len(ut) == len(x1)
                for i in range(B):
                    assert model_output[i].shape == ut[i].shape == x1[i].shape, (
                        f"{model_output[i].shape} {ut[i].shape} {x1[i].shape}"
                    )
                terms["task_loss"] = th.stack(
                    [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
                    dim=0,
                )
            else:
                terms["task_loss"] = mean_flat(((model_output - ut) ** 2))
        else:
            raise NotImplementedError

        terms["loss"] = terms["task_loss"]
        terms["task_loss"] = terms["task_loss"].clone().detach()
        terms["t"] = t
        return terms

    def training_losses_v2(self, ut, t, model_output):
        terms = {}
        if self.model_type == ModelType.VELOCITY:
            terms["task_loss"] = mean_flat(((model_output - ut) ** 2))
        else:
            raise NotImplementedError

        terms["loss"] = terms["task_loss"]
        terms["task_loss"] = terms["task_loss"].clone().detach()
        terms["t"] = t
        return terms

    def get_drift(self):
        """Member function for obtaining the drift of the probability flow ODE."""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, (
                "Output shape from ODE solver must match input shape"
            )
            return model_output

        return body_fn

    def get_score(self):
        if self.model_type == ModelType.NOISE:
            score_fn = (
                lambda x, t, model, **kwargs: model(x, t, **kwargs)
                / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = (
                lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
                    model(x, t, **kwargs), x, t
                )
            )
        else:
            raise NotImplementedError()
        return score_fn


class Sampler:
    """Sampler class for the transport model."""

    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )

        sde_drift = lambda x, t, model, **kwargs: self.drift(
            x, t, model, **kwargs
        ) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)

        return sde_drift, diffusion_fn

    def __get_last_step(self, sde_drift, *, last_step, last_step_size):
        if last_step is None:
            return lambda x, t, model, **model_kwargs: x
        if last_step == "Mean":
            return (
                lambda x, t, model, **model_kwargs: x
                + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        if last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t
            return lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (
                sigma(t)[0][0] ** 2
            ) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        if last_step == "Euler":
            return (
                lambda x, t, model, **model_kwargs: x
                + self.drift(x, t, model, **model_kwargs) * last_step_size
            )
        raise NotImplementedError()

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size
        )

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            assert len(xs) == num_steps, "Samples does not match the number of steps"
            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        do_shift=False,
        time_shifting_factor=None,
        noise_level=None,
    ):
        drift = lambda x, t, model, **kwargs: self.drift(x, t, model, **kwargs)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )
        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            do_shift=do_shift,
            time_shifting_factor=time_shifting_factor,
            noise_level=noise_level,
        )

        return _ode.sample, _ode.t_start

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(
                    th.sum(self.drift(x, t, model, **model_kwargs) * eps), x
                )[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
