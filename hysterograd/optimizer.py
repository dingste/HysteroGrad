import torch
from torch.optim import Optimizer
import numpy as np

class HIOptimizer(Optimizer):
    """
    Hysteretic Information-Geometric Optimizer (HIO). 
    
    Implements:
    1. Natural Gradient Descent (Diagonal approx)
    2. Dynamic Stiffening (Hysteresis) based on information path length (tau)
    3. Geometric Cooling (Annealing) to delay freezing
    4. Feedback mechanism (Shock) to break local minima
    5. Metric Scaling (Amplifies Fisher Information for high-dim data)
    6. Adaptive Thresholds (Hysteresis scales with energy/grad_norm)
    7. Gradient Clipping & Hysteresis Scaling (Forcing stabilization)
    """
    def __init__(self, params, lr=1e-3, stiffening_factor=0.01, beta=0.99, epsilon=1e-7, 
                 cooling_rate=0.99, initial_temp=1.0, metric_scale=1.0, adaptive_threshold=True,
                 grad_clip=None, hysteresis_scale=1.0):
        defaults = dict(lr=lr, stiffening_factor=stiffening_factor, beta=beta, 
                        epsilon=epsilon, cooling_rate=cooling_rate, initial_temp=initial_temp,
                        metric_scale=metric_scale, adaptive_threshold=adaptive_threshold,
                        grad_clip=grad_clip, hysteresis_scale=hysteresis_scale)
        super(HIOptimizer, self).__init__(params, defaults)
        
        # Global stats for logging
        self.last_status = "Liquid"
        self.last_avg_h_width = 0.0
        self.last_avg_grad_norm = 0.0

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        total_liquid = 0
        total_groups = 0
        sum_h_width = 0.0
        sum_grad_norm = 0.0
        
        for group in self.param_groups:
            total_groups += 1
            lr = group['lr']
            stiff = group['stiffening_factor']
            beta = group['beta']
            eps = group['epsilon']
            cooling = group['cooling_rate']
            metric_scale = group.get('metric_scale', 1.0)
            adaptive = group.get('adaptive_threshold', True)
            grad_clip = group.get('grad_clip', None)
            hyst_scale = group.get('hysteresis_scale', 1.0)
            
            # 1. Gather Grads & Update Metric
            group_grads = [] # List of (param, step_component_tensor) 
            
            # Aggregate metrics for the group step norm
            group_d_tau_sq = 0.0
            group_step_norm_sq = 0.0
            
            # Track group stats
            first_param = group['params'][0]
            if first_param not in self.state:
                 self.state[first_param] = {} # Ensure dict exists

            if 'group_tau' not in self.state[first_param]:
                self.state[first_param]['group_tau'] = 0.0
                self.state[first_param]['group_temp'] = group['initial_temp']
                self.state[first_param]['avg_energy'] = 0.0
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['G'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                
                # Update Metric G (Fisher Diagonal)
                # MASSIVE AMPLIFICATION: Multiply the update term by metric_scale
                state['G'].mul_(beta).addcmul_(grad, grad, value=(1 - beta) * metric_scale)
                
                # Calculate local contribution to d_tau and step_norm
                denom = state['G'].sqrt().add_(eps)
                step_component = grad / denom
                
                # We sum squares for the group norm calculation
                # Note: We don't add to group_d_tau_sq here yet because we might clip
                
                # Store for update
                group_grads.append((p, step_component))

            if not group_grads:
                continue

            # 2. Gradient Clipping (Geometric Constraint)
            # Calculate total norm BEFORE clipping
            raw_norm_sq = sum([g[1].pow(2).sum().item() for g in group_grads])
            total_step_norm = np.sqrt(raw_norm_sq)
            
            scale_factor = 1.0
            if grad_clip is not None and total_step_norm > grad_clip:
                scale_factor = grad_clip / (total_step_norm + 1e-6)
                total_step_norm = grad_clip # Effectively clipped
                
            # Apply scaling to step components (virtually) for tau calculation
            # and prepare them for actual update
            final_grads = []
            group_d_tau_sq = 0.0
            
            for p, step_comp in group_grads:
                if scale_factor != 1.0:
                    step_comp.mul_(scale_factor)
                
                final_grads.append((p, step_comp))
                group_d_tau_sq += step_comp.pow(2).sum().item()

            # 3. Group Dynamics
            d_tau = lr * np.sqrt(group_d_tau_sq)
            
            # Update Group Tau
            self.state[first_param]['group_tau'] += d_tau
            current_tau = self.state[first_param]['group_tau']
            
            # Update Group Temp (Cooling)
            current_temp = self.state[first_param]['group_temp']
            self.state[first_param]['group_temp'] = current_temp * cooling
            
            # Update Avg Energy (Moving Average of Gradient Norm)
            # We track the CLIPPED energy now, as that's the effective system energy
            avg_energy = self.state[first_param]['avg_energy']
            avg_energy = beta * avg_energy + (1 - beta) * total_step_norm
            self.state[first_param]['avg_energy'] = avg_energy
            
            # 4. Dynamic Hysteresis Barrier
            # Base: stiff * sqrt(tau)
            # Cooling: * (1 - temp)
            # Adaptive: * avg_energy
            # Scaling: * hyst_scale
            
            eff_temp = max(0.0, min(1.0, current_temp))
            cooling_factor = (1.0 - eff_temp)
            
            h_width = stiff * np.sqrt(current_tau) * cooling_factor * hyst_scale
            
            if adaptive:
                h_width *= (avg_energy + 1e-8)
            
            # 5. Check State
            sum_h_width += h_width
            sum_grad_norm += total_step_norm
            
            if total_step_norm > h_width:
                # LIQUID: Perform Update
                total_liquid += 1
                for p, step_comp in final_grads:
                    p.data.add_(step_comp, alpha=-lr)
            else:
                # FROZEN: No Update
                pass
        
        # Aggregate Status
        status = "Liquid" if total_liquid > 0 else "Frozen"
        avg_h_width = sum_h_width / max(1, total_groups)
        avg_grad_norm = sum_grad_norm / max(1, total_groups)
        
        self.last_status = status
        self.last_avg_h_width = avg_h_width
        self.last_avg_grad_norm = avg_grad_norm
        
        return status, avg_grad_norm, avg_h_width

    def shock(self, factor=0.5):
        """
        Applies a 'Geometric Shock' by reducing the accumulated internal time (tau)
        for all groups.
        """
        print(f"\n⚡ GEOMETRIC SHOCK APPLIED! (Factor: {factor}) ⚡")
        for group in self.param_groups:
            if not group['params']: continue
            first_param = group['params'][0]
            
            if first_param in self.state and 'group_tau' in self.state[first_param]:
                self.state[first_param]['group_tau'] *= factor
                if 'group_temp' in self.state[first_param]:
                    self.state[first_param]['group_temp'] = max(self.state[first_param]['group_temp'], 0.2)