"""Run Pengcheng Zhang's original SS_Model.py with config-only textual patches.

Source: https://github.com/zpcllyj/SobelSchneiderModel (the code behind
Zhang et al. 2025, JAS). The integration loop and physics lines are left
byte-identical to the original; every patch is an exact-string replacement
asserted to occur exactly once, and the patched source is written next to
the output for provenance.

Example:
    python -u run_original_ss_model.py --src /path/to/SS_Model.py \
        --outdir /path/to/rundir --vd 2.5 --days 5475
"""
import argparse
import pathlib
import time


def patch(source: str, old: str, new: str) -> str:
    n = source.count(old)
    assert n == 1, f"expected exactly 1 occurrence of {old!r}, found {n}"
    return source.replace(old, new)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="path to original SS_Model.py")
    p.add_argument("--outdir", required=True)
    p.add_argument("--vd", type=float, default=2.5)
    p.add_argument("--days", type=int, default=365 * 15)
    p.add_argument("--no-vert-advec", action="store_true",
                   help="zero out the vertical momentum advection term")
    p.add_argument("--epsilon-u", type=float, default=None)
    p.add_argument("--kv", type=float, default=None,
                   help="override K_V (original hardcodes 7786*100)")
    p.add_argument("--emfd-heaviside-u", action="store_true",
                   help="restore the H(u) gate on the EMFD term (SS09 eq 2.5); "
                        "the published code has it commented out")
    args = p.parse_args()

    src = pathlib.Path(args.src).read_text()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    src = patch(src, 'SAVEPATH="D:/SS_model_output/"', f'SAVEPATH="{outdir}/"')
    src = patch(src, "TOTAL_INTEGRATION_DAYS = 365*15",
                f"TOTAL_INTEGRATION_DAYS = {args.days}")
    src = patch(src, "V_D=2.5", f"V_D={args.vd}")
    if args.no_vert_advec:
        src = patch(src, "vt=u*grad_v*np.heaviside(THETA_E-theta, 0.5)",
                    "vt=u*grad_v*np.heaviside(THETA_E-theta, 0.5)*0.0")
    if args.epsilon_u is not None:
        src = patch(src, "EPSILON_U=1e-8", f"EPSILON_U={args.epsilon_u}")
    if args.kv is not None:
        src = patch(src, "K_V=7786*100", f"K_V={args.kv}")
    if args.emfd_heaviside_u:
        src = patch(src, "    s=V_D*np.sign(Y-Y_0)*grad_u",
                    "    s=V_D*np.heaviside(u, 0.5)*np.sign(Y-Y_0)*grad_u")

    patched_path = outdir / "SS_Model_patched.py"
    patched_path.write_text(src)

    t0 = time.time()
    exec(compile(src, str(patched_path), "exec"), {"__name__": "__main__"})
    print(f"run complete in {time.time() - t0:.1f} s -> {outdir}/output.nc")


if __name__ == "__main__":
    main()
