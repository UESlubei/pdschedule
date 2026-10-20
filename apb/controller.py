"""Pure bounded controller. Observations are completed, unique step windows."""
from dataclasses import dataclass
import math

MODES = {'off','observe','fixed','load','aimd','joint','aging','equalized'}

@dataclass
class Config:
    mode: str = 'off'
    cap: int = 1024
    minimum: int = 256
    maximum: int = 4096
    initial: int = 2048
    quantum: int = 256
    alpha: float = .2
    beta: float = .5
    additive: int = 256
    deadband: float = .1
    interval: int = 8
    target_ms: float = 20.
    margin: float = .2
    stale_ms: float = 100.
    aging_ms: float = 200.
    aging_floor: int = 1024
    rescue_interval: int = 8
    reserve: bool = True
    fixed_idle_relax: bool = False
    telemetry: bool = True
    output: str = ''
    target_table: str = ''
    context_weight: float = 0.00012347502462458387
    chunk_alignment: int = 16

    def __post_init__(self):
        if self.mode not in MODES: raise ValueError('unknown mode')
        if not 0 < self.alpha <= 1 or not 0 < self.beta < 1: raise ValueError('invalid alpha/beta')
        if not 0 < self.minimum <= self.initial <= self.maximum: raise ValueError('invalid budget bounds')
        if min(self.quantum,self.additive,self.interval,self.rescue_interval,self.cap) < 1: raise ValueError('nonpositive parameter')
        if self.deadband < 0 or self.target_ms <= 0 or self.aging_ms < 0: raise ValueError('invalid latency parameter')
        if self.additive % self.quantum: raise ValueError('additive must be a quantum multiple')
        if not math.isfinite(self.context_weight) or self.context_weight < 0 or self.chunk_alignment < 1:
            raise ValueError('invalid equalized chunk parameter')

def equalized_chunk(start, remaining, base, weight, alignment=16):
    """Equalize linear-token + causal-attention work over ceil(remaining/base) chunks.

    Uses the actual computed frontier (including cache hits/replay). Rounding
    upward avoids an additional tiny tail. No per-request persistent plan is
    needed, so cancellation and preemption cannot leave a stale plan behind.
    """
    if remaining <= 0: return 0
    chunks = (remaining + base - 1) // base
    if chunks == 1: return remaining
    target = remaining * (1 + weight * (start + (remaining + 1) / 2)) / chunks
    linear = 1 + weight * (start + .5)
    # Stable positive root of weight*n*n/2 + linear*n = target.
    n = 2 * target / (linear + math.sqrt(linear * linear + 2 * weight * target))
    return min(remaining, math.ceil(n / alignment) * alignment)

class Controller:
    def __init__(self, cfg):
        self.cfg=cfg; self.budget=cfg.initial; self.ema=None
        self.latest_ms=None; self.last_observed_step=-1; self.completed=0; self.last_rescue=-cfg.rescue_interval
        self.window={}; self.windows=0; self.reason='initial'; self.target=cfg.target_ms
        self.last_decode=0; self.stall_reacted=False; self.window_prefill=False

    def observe(self, step, now_ms, intervals, target=None, had_prefill=True):
        if step <= self.last_observed_step: return False
        self.last_observed_step=step; self.completed+=1
        if target is not None: self.target=target
        if not intervals: return False
        self.latest_ms=now_ms; self.window.update(intervals);self.window_prefill |= had_prefill
        if not hasattr(self,'valid_steps'): self.valid_steps=0
        self.valid_steps+=1
        if self.valid_steps < self.cfg.interval: return False
        values=sorted(self.window.values()); pos=.9*(len(values)-1); i=int(pos)
        d=values[i]+(values[min(i+1,len(values)-1)]-values[i])*(pos-i)
        self.ema=d if self.ema is None else self.cfg.alpha*d+(1-self.cfg.alpha)*self.ema
        self.window.clear(); self.valid_steps=0; self.windows+=1
        error=self.ema/self.target-1
        had_work=self.window_prefill;self.window_prefill=False
        if error>self.cfg.deadband and not had_work:
            self.reason="decode_only_overload"
        elif error>self.cfg.deadband:
            self.budget=max(self.cfg.minimum, int(self.cfg.beta*self.budget)//self.cfg.quantum*self.cfg.quantum)
            self.reason='multiplicative_decrease'
        elif error < -self.cfg.deadband:
            self.budget=min(self.cfg.maximum,self.budget+self.cfg.additive); self.reason='additive_increase'
        else: self.reason='deadband'
        return True

    def decide(self, decode, now_ms, stall_ms=0., prefill_wait_ms=0.):
        c=self.cfg; load=c.maximum if decode==0 else (2048 if decode<=2 else 1024 if decode<=6 else 512)
        load=min(load,c.maximum)
        if c.mode in {'off','observe'}: return c.maximum,'native'
        if c.mode=='equalized': return c.maximum,'equalized_pending' if decode else 'no_decode'
        if c.mode=='fixed': return (c.maximum,'fixed_idle_relax') if c.fixed_idle_relax and decode==0 else (min(c.cap,c.maximum),'fixed')
        if decode==0:
            self.budget=c.maximum; self.ema=None; self.window.clear(); self.valid_steps=0
            self.latest_ms=None; self.last_decode=0; self.stall_reacted=False;self.window_prefill=False
            return c.maximum,'no_decode'
        # Reset across major load transitions; use current load until fresh feedback.
        if self.last_decode and ((decode<=2)!=(self.last_decode<=2)):
            self.ema=None; self.window.clear(); self.valid_steps=0; self.latest_ms=None;self.window_prefill=False
        self.last_decode=decode
        fresh=self.latest_ms is not None and now_ms-self.latest_ms<=max(c.stale_ms,4*self.target)
        if c.mode=='load' or not fresh: budget=load; reason='load' if c.mode=='load' else 'stale_fallback'
        else:
            budget=self.budget if c.mode=='aimd' else min(load,self.budget); reason=self.reason
        if stall_ms > self.target*(1+c.deadband) and not self.stall_reacted and c.mode in {'aimd','joint','aging'}:
            self.budget=max(c.minimum,int(self.budget*c.beta)//c.quantum*c.quantum)
            budget=min(budget,self.budget); reason='decode_stall'; self.stall_reacted=True
        elif stall_ms <= self.target: self.stall_reacted=False
        if c.mode=='aging' and prefill_wait_ms>c.aging_ms and self.completed-self.last_rescue>=c.rescue_interval:
            budget=max(budget,c.aging_floor);self.last_rescue=self.completed;reason='aging_rescue'
        return max(c.minimum,min(c.maximum,budget)),reason
