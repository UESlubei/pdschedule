import unittest
from apb.controller import Config, equalized_chunk


class EqualizedTests(unittest.TestCase):
    def test_equal_cost_three_chunks_without_extra_tail(self):
        weight=Config().context_weight
        start=0;chunks=[];costs=[]
        while start<6144:
            n=equalized_chunk(start,6144-start,2048,weight)
            chunks.append(n);costs.append(n*(1+weight*(start+(n+1)/2)));start+=n
        self.assertEqual(len(chunks),3)
        self.assertGreater(chunks[0],2048)
        self.assertLess(chunks[-1],2048)
        self.assertLess(max(costs)/min(costs),1.03)

    def test_progress_cache_frontiers_and_unaligned_tails(self):
        for end in [17,513,2049,6143,6144,8191]:
            for start in [0,end//2,end-1]:
                initial=start;steps=0
                while start<end:
                    n=equalized_chunk(start,end-start,2048,Config().context_weight)
                    self.assertGreater(n,0);self.assertLessEqual(n,end-start)
                    start+=n;steps+=1
                self.assertLessEqual(steps,(end-initial+2047)//2048)

    def test_zero_weight_and_validation(self):
        self.assertEqual(equalized_chunk(0,6144,2048,0),2048)
        self.assertEqual(equalized_chunk(6144,0,2048,0),0)
        for kw in [dict(context_weight=-1),dict(context_weight=float('nan')),dict(chunk_alignment=0)]:
            with self.assertRaises(ValueError):Config(**kw)

    def test_selected_alignment_reduces_peak_work_without_an_extra_step(self):
        w=Config().context_weight;start=0;costs=[];chunks=[]
        while start<6144:
            n=equalized_chunk(start,6144-start,2048,w,256)
            chunks.append(n);costs.append(n*(1+w*(start+(n+1)/2)));start+=n
        static=[2048*(1+w*(s+2049/2)) for s in [0,2048,4096]]
        self.assertEqual(len(chunks),3)
        self.assertTrue(all(n%256==0 for n in chunks))
        self.assertLess(max(costs),max(static))

if __name__=='__main__':unittest.main()
