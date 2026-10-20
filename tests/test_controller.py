import unittest
from types import SimpleNamespace as NS
from apb.controller import Config,Controller
from apb.runtime import partition

class ControllerTests(unittest.TestCase):
    def test_invalid(self):
        for kw in [dict(alpha=0),dict(beta=1),dict(interval=0),dict(minimum=3000),dict(additive=1)]:
            with self.assertRaises(ValueError):Config(**kw)
    def test_load_and_idle(self):
        c=Controller(Config(mode='load'))
        self.assertEqual([c.decide(n,0)[0] for n in [0,1,3,8]],[4096,2048,1024,512])
    def test_fixed_stays_fixed_without_decode(self):
        self.assertEqual(Controller(Config(mode='fixed',cap=512)).decide(0,0)[0],512)
    def test_fixed_idle_relaxation_is_explicit(self):
        c=Controller(Config(mode="fixed",cap=512,fixed_idle_relax=True))
        self.assertEqual(c.decide(0,0)[0],4096);self.assertEqual(c.decide(8,0)[0],512)
    def test_additive_not_stuck_at_powers_of_two(self):
        c=Controller(Config(mode='aimd',interval=1,initial=1024,target_ms=20))
        for i in range(3):c.observe(i,i,{'r':5})
        self.assertEqual(c.budget,1792)
    def test_decrease_and_bound(self):
        c=Controller(Config(mode='aimd',interval=1,target_ms=10))
        for i in range(8):c.observe(i,i,{'r':30})
        self.assertEqual(c.budget,256)
    def test_duplicate_and_empty_not_counted(self):
        c=Controller(Config(mode='aimd',interval=2));c.observe(1,1,{})
        c.observe(2,2,{'a':30});c.observe(2,3,{'a':30})
        self.assertEqual(c.windows,0)
        c.observe(3,4,{'a':30});self.assertEqual(c.windows,1)
    def test_stale_fallback(self):
        c=Controller(Config(mode='joint',interval=1));c.observe(1,0,{'r':5})
        self.assertEqual(c.decide(8,1000)[1],'stale_fallback')
    def test_rescue_has_cooldown(self):
        c=Controller(Config(mode='aging'));self.assertEqual(c.decide(8,0,prefill_wait_ms=500),(1024,'aging_rescue'))
        self.assertEqual(c.decide(8,1,prefill_wait_ms=500)[0],512)
    def test_stall_responds_once(self):
        c=Controller(Config(mode='joint',target_ms=10))
        c.decide(8,0,stall_ms=50);a=c.budget;c.decide(8,1,stall_ms=100)
        self.assertEqual(c.budget,a)
    def test_decode_only_overload_does_not_keep_reducing(self):
        c=Controller(Config(mode="joint",interval=1,target_ms=10))
        for i in range(20):c.observe(i,i,{"r":30},had_prefill=False)
        self.assertEqual(c.budget,2048);self.assertEqual(c.reason,"decode_only_overload")
    def test_prompt_replay_decode_partition(self):
        r=NS(num_prompt_tokens=100,num_tokens=140)
        self.assertEqual(partition(r,80,60),(20,39,1))
        self.assertEqual(partition(r,100,17),(0,17,0))
        self.assertEqual(partition(r,139,1),(0,0,1))
        self.assertEqual(partition(NS(num_prompt_tokens=17,num_tokens=17),0,17),(17,0,0))

if __name__=='__main__':unittest.main()
