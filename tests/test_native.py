import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('jayou_native', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
from jayou_native import color, fitting, fields, masks, registration, pipeline, nodes


def options(**updates):
    opts = {}
    for k, (kind, *extra) in nodes.JayouSDRAWPython.INPUT_TYPES()['required'].items():
        if k in ('image_a', 'image_b'): continue
        opts[k] = (extra[0] if extra else {}).get('default', kind[0] if isinstance(kind, list) else None)
    opts['对齐'] = '关闭'; opts.update(updates)
    return opts


def fixture(h=64, w=96):
    rng = np.random.default_rng(13)
    a = rng.uniform(.08, .85, (h, w, 3)).astype(np.float32)
    return cv2.GaussianBlur(a, (3, 3), .5)


class NativeTests(unittest.TestCase):
    def test_identity_render(self):
        a = fixture(); np.testing.assert_allclose(color.render(a, color.defaults()), a, atol=1e-7)

    def test_hsl_roundtrip(self):
        a = fixture(); np.testing.assert_allclose(color.hsl_rgb(*color.rgb_hsl(a)), a, atol=3e-7)

    def test_monotonic_curve_no_overshoot(self):
        p = np.array([0, .1, .55, .7, 1]); v = color.curve(np.linspace(0, 1, 1000), p)
        self.assertTrue(np.all(np.diff(v) >= 0)); self.assertGreaterEqual(v.min(), 0); self.assertLessEqual(v.max(), 1)

    def test_known_parameter_fit(self):
        a = fixture(); p = color.defaults(); p['basic'][:4] = [.4, 10, -3, 12]
        b = np.clip(color.render(a, p), 0, 1); fit = fitting.fit_parameters(a, b, np.ones(a.shape[:2]))
        before = np.mean((a-b)**2); after = np.mean((color.render(a, fit)-b)**2)
        self.assertLess(after, before*.15)

    def test_affine_recovers_transform(self):
        a = fixture(); b = a*np.array([.9, 1.06, .94])+.02
        g = fields.fit_affine(a, b, np.ones(a.shape[:2]))
        self.assertLess(np.mean((fields.apply_affine(g, a)-b)**2), np.mean((a-b)**2)*.05)

    def test_field_strip_coordinates(self):
        a = fixture(72, 96); b = a*(.7+np.linspace(0, .5, 96)[None, :, None])
        g = fields.fit_affine(a, b, np.ones(a.shape[:2]))
        whole = fields.apply_affine(g, a)
        split = np.concatenate([fields.apply_affine(g, a[:31], 0, a.shape[:2]), fields.apply_affine(g, a[31:], 31, a.shape[:2])])
        np.testing.assert_array_equal(whole, split)

    def test_light_field_improves_and_zero_controls(self):
        a = fixture(); b = a*np.linspace(.65, 1.3, a.shape[1])[None, :, None]
        g = (fields.fit_gain(a, b, np.ones(a.shape[:2]), 8), fields.fit_gain(a, b, np.ones(a.shape[:2]), 24))
        out = fields.apply_light(g, a, 1, 1, 0)
        self.assertLess(np.mean((out-b)**2), np.mean((a-b)**2)*.3)
        np.testing.assert_allclose(fields.apply_light(g, a, 0, 0, 0), a, atol=1e-7)

    def test_adaptive_preserves_interior(self):
        a = fixture(); m = np.zeros(a.shape[:2], np.float32); m[10:50, 10:80] = 1
        refined = masks.process_edges(m, a, '自适应RGB贴合', 0, 0)
        np.testing.assert_array_equal(refined[15:45, 15:75], 1)
        self.assertEqual(refined[0, 0], 0)

    def test_holes_keep_open_background(self):
        m = np.ones((40, 40), np.float32); m[10:12, 10:12] = 0; m[:8, 20:22] = 0
        out = masks.fill_small_holes(m, 64)
        self.assertTrue(out[10:12, 10:12].all()); self.assertFalse(out[:8, 20:22].any())

    def test_edge_expand_and_feather(self):
        a = fixture(); m = np.zeros(a.shape[:2], np.float32); m[20:40, 20:40] = 1
        expanded = masks.process_edges(m, a, '保持分割结果', 2, 0)
        shrunk = masks.process_edges(m, a, '保持分割结果', -2, 0)
        self.assertGreater(expanded.sum(), m.sum()); self.assertLess(shrunk.sum(), m.sum())
        softened = masks.process_edges(m, a, '保持分割结果', 0, 2)
        self.assertGreater(np.count_nonzero((softened > 0)&(softened < 1)), 0)

    def test_black_text_excluded_from_highlight(self):
        a = np.ones((80, 80, 3), np.float32); a[30:50, 30:50] = .03
        b = np.full_like(a, .65)
        out, alpha = masks.recovery(a, a, b, np.ones((80, 80)), np.zeros((80, 80)), options(高光细节补偿=True))
        np.testing.assert_array_equal(out[32:48, 32:48], a[32:48, 32:48]); self.assertEqual(alpha[40, 40], 0)
        self.assertGreater(alpha[10, 10], .9)

    def test_dark_object_not_highlight(self):
        a = fixture(); base = np.ones_like(a); b = a*.8
        _, alpha = masks.recovery(base, a, b, np.ones(a.shape[:2]), np.zeros(a.shape[:2]), options(高光细节补偿=True))
        self.assertFalse(alpha.any())

    def test_mask_replaces_only_selected_pixels(self):
        a = fixture(); b = a*.7; m = np.zeros(a.shape[:2], np.float32); m[10:25, 10:25] = 1
        out, used = masks.recovery(a, a, b, np.ones(m.shape), m, options(大区域参考补偿=True))
        np.testing.assert_array_equal(out[m == 0], a[m == 0]); np.testing.assert_allclose(out[m == 1], b[m == 1], atol=1e-7)
        np.testing.assert_array_equal(used, m)

    def test_invalid_overlap_cannot_compensate(self):
        a = fixture(); z = np.zeros(a.shape[:2]); m = np.ones(z.shape)
        out, used = masks.recovery(a, a, a*.5, z, m, options(大区域参考补偿=True))
        np.testing.assert_array_equal(out, a); self.assertFalse(used.any())

    def test_identity_pipeline_batch_and_empty_mask(self):
        a = torch.from_numpy(fixture()).unsqueeze(0).repeat(2, 1, 1, 1)
        out, mask = nodes.JayouSDRAWPython().process(a, a[:1], torch.zeros(a.shape[1:3]), 对齐='关闭', 大区域参考补偿=True)
        np.testing.assert_allclose(out, a, atol=1e-7); self.assertFalse(mask.any()); self.assertEqual(out.shape, a.shape)

    def test_reject_inputs(self):
        n = nodes.JayouSDRAWPython(); a = torch.from_numpy(fixture()).unsqueeze(0)
        for kw in [dict(整体强度=2), dict(高光细节补偿='true'), dict(大区域参考补偿=True), dict(unknown=1)]:
            with self.assertRaises(ValueError): n.process(a, a, 对齐='关闭', **kw)
        with self.assertRaises(ValueError): n.process(a*float('nan'), a, 对齐='关闭')
        with self.assertRaises(ValueError): n.process(a, a, torch.ones(3, 10, 10), 对齐='关闭')

    def test_identity_warp_and_invalid_fill(self):
        a = fixture(); matrix = np.eye(3)
        b, m = registration.warp(a, a, matrix)
        np.testing.assert_array_equal(a, b); self.assertTrue(m.all())
        matrix[0, 2] = .2
        b, m = registration.warp(a, a, matrix)
        np.testing.assert_array_equal(b[m == 0], a[m == 0])

    def test_translation(self):
        a = fixture(192, 256)
        b = cv2.warpAffine(a, np.array([[1, 0, 7], [0, 1, -4]], np.float32), (256, 192), borderMode=cv2.BORDER_REFLECT)
        matrix = registration.estimate(a, b, '平移')
        self.assertAlmostEqual(matrix[0, 2]*256, 7, delta=.3)
        self.assertAlmostEqual(matrix[1, 2]*192, -4, delta=.3)

    def test_sift_alignment(self):
        a = fixture(240, 320)
        b = cv2.warpAffine(a, np.array([[1, 0, 6], [0, 1, 3]], np.float32), (320, 240), borderMode=cv2.BORDER_REFLECT)
        matrix = registration.estimate(a, b, 'SIFT + MAGSAC++')
        self.assertAlmostEqual(matrix[0, 2]*320, 6, delta=.3)
        self.assertAlmostEqual(matrix[1, 2]*240, 3, delta=.3)

    def test_chunked_recovery_no_seams(self):
        a = fixture(80, 120); b = a.copy(); b[10:60, 10:90] = .75
        m = np.zeros(a.shape[:2], np.float32); m[10:60, 10:90] = 1
        o = options(大区域参考补偿=True, 边缘类型='自适应RGB贴合', 羽化像素=2, 颜色柔和过渡=True)
        # Fitting is already covered; isolate strip topology with fixed parameters.
        with patch.object(pipeline, 'fit_parameters', return_value=color.defaults()):
            whole, ma, _ = pipeline.process(a, b, m, o)
            with patch.object(pipeline, 'chunks', return_value=iter([(0, 17), (17, 41), (41, 80)])):
                split, mb, _ = pipeline.process(a, b, m, o)
        np.testing.assert_allclose(whole, split, atol=2e-6); np.testing.assert_allclose(ma, mb, atol=2e-6)

    def test_no_js_runtime_files(self):
        self.assertFalse(list(ROOT.rglob('*.js'))+list(ROOT.rglob('*.mjs')))
        for p in ROOT.glob('*.py'):
            self.assertNotIn('subprocess', p.read_text()); self.assertNotIn('JAYOU_NODE_BIN', p.read_text())

if __name__ == '__main__': unittest.main()
