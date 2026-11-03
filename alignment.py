"""Non-destructive reference registration: RootSIFT + LightGlue + MAGSAC++."""
import json, sys, time
import numpy as np
import cv2


def estimate_geometry(a, b, source_size, target_size):
    if len(a) < 16:
        raise ValueError('可靠匹配点不足，请使用同一张照片的参考版本')
    H, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 2.5, maxIters=10000, confidence=0.999)
    if H is None or mask is None or not np.isfinite(H).all():
        raise ValueError('无法估计稳定的透视关系')
    keep = mask.ravel().astype(bool)
    if keep.sum() < 16 or keep.mean() < .25:
        raise ValueError('对应关系不可靠，已保留原图与参考图')
    wa, ha = source_size; wb, hb = target_size
    coverage = cv2.contourArea(cv2.convexHull(a[keep].astype(np.float32))) / (wa*ha)
    if coverage < .08:
        raise ValueError('匹配点集中在局部，无法可靠对齐整张照片')
    # Coordinates are normalized pixel centers, independent of preview resolution.
    N = np.array([[1/wb,0,.5/wb],[0,1/hb,.5/hb],[0,0,1]]) @ H @ np.array([[wa,0,-.5],[0,ha,-.5],[0,0,1]])
    N /= N[2, 2]
    grid = np.array([[x,y] for y in np.linspace(0,1,31) for x in np.linspace(0,1,31)],np.float64)
    denom = grid @ N[2,:2] + N[2,2]
    if np.min(denom)*np.max(denom) <= 0 or np.min(np.abs(denom)) < .05:
        raise ValueError('透视变换不稳定，已拒绝应用')
    mapped = cv2.perspectiveTransform(grid[None], N)[0]
    overlap = np.mean(np.all((mapped >= 0) & (mapped <= 1), axis=1))
    corners=cv2.perspectiveTransform(np.array([[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]]),N)[0]
    area=cv2.contourArea(corners.astype(np.float32),oriented=True)
    if overlap < .3 or not .08 < area < 12 or not cv2.isContourConvex(corners.astype(np.float32)):
        raise ValueError('有效重叠区域不足或变形过大，已拒绝应用')
    error=np.linalg.norm(cv2.perspectiveTransform(a[keep,None,:],H)[:,0]-b[keep],axis=1)
    return {'kind':'homography','matrix':N.ravel().tolist(),'score':float(keep.mean()),'matches':len(a),'inliers':int(keep.sum()),'coverage':float(coverage),'overlap':float(overlap),'error':float(np.median(error)),'method':'RootSIFT + LightGlue + MAGSAC++'}


def align(source, target):
    import torch
    from lightglue import SIFT, LightGlue
    from lightglue.utils import load_image, rbd
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    # Bounded point count and CPU inference avoid taking memory from the editor GPU.
    extractor=SIFT(max_num_keypoints=2048).eval()
    matcher=LightGlue(features='sift',flash=False).eval()
    with torch.inference_mode():
        images=[load_image(p) for p in (source,target)]
        feats=[extractor.extract(im,resize=1280) for im in images]
        if min(f['keypoints'].shape[1] for f in feats)<16:
            raise ValueError('图片纹理不足，无法可靠对齐')
        result=rbd(matcher({'image0':feats[0],'image1':feats[1]}))
        feats=[rbd(f) for f in feats]; matches=result['matches']
        pts=[f['keypoints'][matches[:,i]].cpu().numpy() for i,f in enumerate(feats)]
    sizes=[(im.shape[2],im.shape[1]) for im in images]
    return estimate_geometry(*pts,*sizes)

if __name__=='__main__':
    try:
        started=time.monotonic();result=align(sys.argv[1],sys.argv[2]);result['seconds']=time.monotonic()-started
        with open(sys.argv[3],'w') as f:json.dump(result,f)
    except Exception as e:
        print('图像对齐失败：'+str(e),file=sys.stderr);sys.exit(1)
