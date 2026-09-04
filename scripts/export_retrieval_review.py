"""Build the STRAP-retrieval review page: for sampled query segments, show the
imagined query frames (top row) aligned via the DTW warping path with the
retrieved real-demo frames (bottom row), plus per-segment action traces
(original imagined actions vs retrieved labels) and pool-level stats.

Self-contained HTML (JPEG data URIs), same console styling as PairScope.
Reads retr400_matches.npz + the imagined/real zarrs.

Usage:
  python scripts/export_retrieval_review.py --out outputs/h1h2/retrieval.html
"""
import argparse
import base64
import json
from pathlib import Path
import sys

import cv2
import numpy as np

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

IWS = Path(__file__).resolve().parent.parent


def jpeg_uri(img_rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def strip(frames, size=104):
    return np.concatenate(
        [cv2.resize(f, (size, size), interpolation=cv2.INTER_NEAREST) for f in frames],
        axis=1)


def svg_traces(a_orig, a_retr, w=420, h=90):
    """Tiny inline SVG: original (amber) vs retrieved (cyan) actions, 4 dims."""
    n = len(a_orig)
    lo = min(a_orig.min(), a_retr.min())
    hi = max(a_orig.max(), a_retr.max())
    span = max(hi - lo, 1e-6)
    X = lambda i: i / max(n - 1, 1) * (w - 8) + 4
    Y = lambda v: h - 4 - (v - lo) / span * (h - 8)
    parts = [f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px;background:#10171C;border-radius:4px">']
    for d in range(4):
        for arr, col, op in ((a_orig, "#E8A33D", .9), (a_retr, "#4FB8C9", .9)):
            pts = " ".join(f"{X(i):.1f},{Y(arr[i, d]):.1f}" for i in range(n))
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                         f'stroke-width="1" opacity="{op * (0.55 + 0.15 * d)}"/>')
    parts.append("</svg>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="datasets/retr400_matches.npz")
    ap.add_argument("--manifest", default="datasets/retr400_manifest.json")
    ap.add_argument("--shards", default="D,E")
    ap.add_argument("--n_examples", type=int, default=36)
    ap.add_argument("--frames_per_strip", type=int, default=9)
    ap.add_argument("--out", default="outputs/h1h2/retrieval.html")
    args = ap.parse_args()

    m = np.load(IWS / args.matches)
    man = json.loads((IWS / args.manifest).read_text())
    seg = m["seg"]  # (S, 7): imag_ep, qs, qe, real_ep, rs, re, cost/frame
    lens = m["episode_lens"]
    off = np.r_[0, np.cumsum(lens[:-1])]
    match_fr = m["match_fr"]

    real = ReplayBuffer.copy_from_path(str(IWS / "datasets/rotate_t_rand800_dp.zarr"))
    retr = ReplayBuffer.copy_from_path(str(IWS / "datasets/rotate_t_r20_retr400_dp.zarr"),
                                       keys=["action"])
    n_real_mix = man["n_real_mix"]
    wms, base = {}, 0
    for sh in args.shards.split(","):
        z = ReplayBuffer.copy_from_path(
            str(IWS / f"datasets/rotate_t_imagined_pair{sh}_dp.zarr"),
            keys=["img", "action"])
        wms[sh] = (z, base)
        base += z.n_episodes

    # sample segments stratified by cost quantile
    order = np.argsort(seg[:, 6])
    qidx = [order[int(q * (len(order) - 1))] for q in np.linspace(0.02, 0.98, args.n_examples)]

    cards = []
    for si in qidx:
        iep, qs, qe, rep, rs, re_, cost = seg[si]
        iep, qs, qe, rep = int(iep), int(qs), int(qe), int(rep)
        for sh, (z, b) in wms.items():
            if b <= iep < b + z.n_episodes:
                wm_ep = z.get_episode(iep - b)
                break
        real_ep = real.get_episode(rep)
        ks = np.linspace(qs, qe - 1, min(args.frames_per_strip, qe - qs)).astype(int)
        mf = match_fr[off[iep]:off[iep] + lens[iep]]
        top = strip(wm_ep["img"][ks])
        bot = strip(real_ep["img"][np.clip(mf[ks], 0, len(real_ep["img"]) - 1)])
        a_orig = wm_ep["action"][qs:qe]
        a_retr = retr.get_episode(n_real_mix + iep)["action"][qs:qe]
        mean_d = float(np.linalg.norm(a_retr - a_orig, axis=1).mean() * 100)
        cards.append(f"""
<div class="card">
 <div class="meta"><b>imagined ep {iep}</b> frames {qs}–{qe} &nbsp;→&nbsp;
   <b>real ep {rep}</b> &nbsp;|&nbsp; DTW cost/frame <b>{cost:.1f}</b>
   &nbsp;|&nbsp; label Δ <b>{mean_d:.1f} cm</b></div>
 <img src="{jpeg_uri(top)}" alt="query"><div class="rowlab wm">imagined query W_t</div>
 <img src="{jpeg_uri(bot)}" alt="retrieved"><div class="rowlab sim">DTW-matched real frames</div>
 {svg_traces(a_orig, a_retr)}
 <div class="rowlab">actions over the segment — <span class="wm">original π(a|w)</span> vs <span class="sim">retrieved expert</span> (4 EE-xy dims)</div>
</div>""")

    stats = (f"segments <b>{man['n_segments']}</b> · seg len median "
             f"<b>{man['seg_len']['median']}</b>f · DTW cost/frame p10/mean/p90 "
             f"<b>{man['cost_per_frame']['p10']}/{man['cost_per_frame']['mean']}/"
             f"{man['cost_per_frame']['p90']}</b> · label Δ by drift (cm): " +
             " ".join(f"{k}:{round(v * 100, 1)}"
                      for k, v in man["mean_abs_diff_by_plan_drift"].items()))

    html = f"""<title>Retrieval Review</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700&family=IBM+Plex+Mono&display=swap">
<style>
 body {{ background:#0C1216; color:#E8EEF2; font-family:Archivo,system-ui,sans-serif; margin:0; padding:18px 22px; }}
 h1 {{ font-size:19px; margin:0 0 4px; }} h1 .sim {{ color:#4FB8C9; }}
 .sub {{ font-family:"IBM Plex Mono",monospace; font-size:12px; color:#8FA0AC; margin-bottom:16px; }}
 .sub b {{ color:#E8EEF2; font-weight:500; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(460px,1fr)); gap:14px; }}
 .card {{ background:#151C22; border:1px solid #26313A; border-radius:6px; padding:10px; }}
 .card img {{ width:100%; image-rendering:pixelated; border-radius:3px; display:block; }}
 .meta {{ font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:#8FA0AC; margin-bottom:6px; }}
 .meta b {{ color:#E8EEF2; font-weight:500; }}
 .rowlab {{ font-family:"IBM Plex Mono",monospace; font-size:10.5px; color:#5C6B76; margin:2px 0 6px; }}
 .wm {{ color:#E8A33D; }} .sim {{ color:#4FB8C9; }}
</style>
<h1>Retrieval <span class="sim">Review</span> — STRAP sub-trajectory retrieval quality</h1>
<div class="sub">{stats}<br>Cards sampled across the DTW-cost range (best → worst). Each shows the
imagined query segment, the real frames the warping path aligned it to, and the action-label swap.</div>
<div class="grid">{"".join(cards)}</div>"""
    out = IWS / args.out
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(cards)} cards)")


if __name__ == "__main__":
    main()
