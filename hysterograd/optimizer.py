import torch
from torch.optim import Optimizer

class HIOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, stiffening_factor=0.01, beta=0.99, epsilon=1e-7, 
                 cooling_rate=0.99, initial_temp=1.0, metric_scale=1.0, adaptive_threshold=True,
                 grad_clip=None, hysteresis_scale=1.0):
        defaults = dict(lr=lr, stiffening_factor=stiffening_factor, beta=beta, 
                        epsilon=epsilon, cooling_rate=cooling_rate, initial_temp=initial_temp,
                        metric_scale=metric_scale, adaptive_threshold=adaptive_threshold,
                        grad_clip=grad_clip, hysteresis_scale=hysteresis_scale)
        super(HIOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        total_liquid = 0
        sum_h_width = 0.0
        sum_grad_norm = 0.0
        num_groups = len(self.param_groups)
        
        for group in self.param_groups:
            lr, beta, eps = group['lr'], group['beta'], group['epsilon']
            p_ref = group['params'][0]
            state_group = self.state[p_ref]
            
            if 'group_tau' not in state_group:
                state_group['group_tau'] = torch.tensor(0.0, device=p_ref.device)
                state_group['group_temp'] = torch.tensor(group['initial_temp'], device=p_ref.device)
                state_group['avg_energy'] = torch.tensor(0.0, device=p_ref.device)

            group_sq_norm = torch.tensor(0.0, device=p_ref.device)
            valid_params = []

            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if 'G' not in state: state['G'] = torch.zeros_like(p)

                # Metric Update
                state['G'].mul_(beta).addcmul_(p.grad, p.grad, value=(1 - beta) * group['metric_scale'])
                denom = state['G'].sqrt().add_(eps)
                step_comp = p.grad / denom
                
                group_sq_norm += step_comp.pow(2).sum()
                valid_params.append((p, step_comp))

            if not valid_params: continue

            # Norm & Dynamics
            total_step_norm = group_sq_norm.sqrt()
            scale_f = min(1.0, group['grad_clip'] / (total_step_norm + 1e-6)) if group['grad_clip'] else 1.0
            eff_norm = total_step_norm * scale_f

            state_group['group_tau'] += lr * eff_norm
            state_group['group_temp'].mul_(group['cooling_rate'])
            state_group['avg_energy'] = beta * state_group['avg_energy'] + (1 - beta) * eff_norm

            # Hysteresis Threshold
            cool_f = 1.0 - state_group['group_temp'].clamp(0.0, 1.0)
            h_width = group['stiffening_factor'] * state_group['group_tau'].sqrt() * cool_f * group['hysteresis_scale']
            if group['adaptive_threshold']: h_width *= (state_group['avg_energy'] + 1e-8)

            # Metrics for return
            sum_h_width += h_width.item()
            sum_grad_norm += eff_norm.item()

            if eff_norm > h_width:
                total_liquid += 1
                for p, step_comp in valid_params:
                    p.add_(step_comp, alpha=-lr * scale_f)
            
        avg_h = sum_h_width / num_groups
        avg_gnorm = sum_grad_norm / num_groups
        status = "Liquid" if total_liquid > 0 else "Frozen"
        
        return status, avg_gnorm, avg_h

    def shock(self, factor=0.5):
        for group in self.param_groups:
            p_ref = group['params'][0]
            if p_ref in self.state and 'group_tau' in self.state[p_ref]:
                self.state[p_ref]['group_tau'].mul_(factor)
                self.state[p_ref]['group_temp'].clamp_(min=0.2)
