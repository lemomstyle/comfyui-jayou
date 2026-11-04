import unittest
from unittest.mock import patch
import numpy as np
from test_native import fixture, options
from jayou_native import texture_finish as tf, pipeline, color


class TextureTests(unittest.TestCase):
    def pair(self):
        y, x = np.mgrid[:128, :192]
        base = .16+.64*x/191+.025*np.sin(y*.4)+.02*np.cos(x*.5)
        b = np.stack([base, base*.92, base*.83], -1).astype(np.float32)
        a = b*.72+.14
        return a, b

    def test_improves_known_haze(self):
        a, b = self.pair(); recipe, report = tf.fit(a, b, a, np.ones(a.shape[:2]))
        self.assertTrue(report['accepted'])
        out = tf.apply(a, recipe, 1)
        self.assertLess(np.mean((out-b)**2), np.mean((a-b)**2)*.9)
        self.assertGreater(np.std(out @ color.LUMA), np.std(a @ color.LUMA))

    def test_identical_is_rejected(self):
        a = fixture(128,192)
        recipe, report = tf.fit(a, a, a, np.ones(a.shape[:2]))
        self.assertIsNone(recipe); self.assertFalse(report['accepted'])

    def test_no_overlap_is_unchanged(self):
        a, b = self.pair(); before = a.copy()
        tf.finish(a,b,a,np.zeros(a.shape[:2]),1)
        np.testing.assert_array_equal(a,before)

    def test_zero_strength_exact(self):
        a,b=self.pair();before=a.copy()
        tf.finish(a,b,a,np.ones(a.shape[:2]),0)
        np.testing.assert_array_equal(a,before)

    def test_invalid_border_exact(self):
        a,b=self.pair();before=a.copy();valid=np.ones(a.shape[:2],np.float32)
        valid[:,:13]=0
        tf.finish(a,b,a.copy(),valid,1)
        np.testing.assert_array_equal(a[:,:13],before[:,:13])

    def test_no_new_clipping(self):
        a,b=self.pair();recipe,_=tf.fit(a,b,a,np.ones(a.shape[:2]))
        a[:8]=0;a[-8:]=1
        out=tf.apply(a,recipe,1)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreaterEqual(out.min(),-1e-7);self.assertLessEqual(out.max(),1+1e-7)

    def test_gray_stays_gray(self):
        a,b=self.pair(); a=np.repeat((a@color.LUMA)[...,None],3,-1);b=np.repeat((b@color.LUMA)[...,None],3,-1)
        recipe,_=tf.fit(a,b,a,np.ones(a.shape[:2]));out=tf.apply(a,recipe,1)
        np.testing.assert_array_equal(out[...,0],out[...,1]);np.testing.assert_array_equal(out[...,1],out[...,2])

    def test_disabled_pipeline_exact_and_mask_unchanged(self):
        a,b=self.pair();m=np.zeros(a.shape[:2],np.float32);m[20:40,20:40]=1
        with patch.object(pipeline,'fit_parameters',return_value=color.defaults()):
            first, fm,_=pipeline.process(a,b,m,options(大区域参考补偿=True))
            with patch.object(pipeline,'finish',side_effect=AssertionError('must not run')):
                second,sm,_=pipeline.process(a,b,m,options(大区域参考补偿=True,质感精修=True,质感强度=0))
            third,tm,_=pipeline.process(a,b,m,options(大区域参考补偿=True,质感精修=True,质感强度=.5))
        np.testing.assert_array_equal(first,second);np.testing.assert_array_equal(fm,sm);np.testing.assert_array_equal(fm,tm)

    def test_strip_render_no_seams(self):
        a,b=self.pair();recipe,_=tf.fit(a,b,a,np.ones(a.shape[:2]))
        whole=tf.apply(a,recipe,.7)
        result=np.empty_like(a);r=int(np.ceil(3*max(.8,max(a.shape[:2])/180)))
        for top,bottom in [(0,31),(31,67),(67,128)]:
            lo,hi=max(0,top-r),min(128,bottom+r)
            strip=tf.apply(a[lo:hi],recipe,.7,lo,a.shape[:2]);result[top:bottom]=strip[top-lo:bottom-lo]
        np.testing.assert_allclose(result,whole,atol=1e-7)

if __name__=='__main__':unittest.main()
