"""
    The precision in which the PyTorch backend of bpnet-shap computes
    (bpnet.pytorch.attribution._FullPrecision): while the scores are
    computed, float32 is float32, whatever the caller had set; and
    afterwards every switch of PyTorch is what it was, including
    which settings follow the ones above them.

    The switches are global to a process and some of them cannot be
    set back to the state PyTorch starts in, so all of this runs in a
    process of its own. It needs PyTorch only (not tangermeme), and
    covers both generations of switches: the settings per backend and
    kind of operation of PyTorch 2.9 and later, and the older flags
    that are all there is before. It runs twice: as PyTorch starts,
    and as it starts with the environment variable
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE, which switches TF32 on for
    matrix products (and, before PyTorch 2.0, keeps it on).
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")


PRECISION = r'''
import os

import torch
from bpnet.pytorch import attribution

b = torch.backends
OVERRIDE = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1"
read = getattr(torch._C, "_get_fp32_precision_getter", None)
write = getattr(torch._C, "_set_fp32_precision_setter", None)
NEWER = read is not None
NODES = [("generic", "all"), ("cuda", "all"), ("mkldnn", "all"),
         ("cuda", "conv"), ("cuda", "rnn"), ("cuda", "matmul"),
         ("mkldnn", "conv"), ("mkldnn", "rnn"), ("mkldnn", "matmul")]
CUDNN = [("cuda", "conv"), ("cuda", "rnn")]


def older():
    """The older switches, as far as PyTorch reports them"""
    out = {}
    for name, reader in (("flag of cuDNN", lambda: b.cudnn.allow_tf32),
                         ("flag of matmul", lambda: b.cuda.matmul.allow_tf32),
                         ("matmul precision",
                          torch.get_float32_matmul_precision)):
        try:
            out[name] = reader()
        except RuntimeError:
            out[name] = "not reported"
    return out


def resolved():
    """Every setting as a computation finds it"""
    return {node: read(*node) for node in NODES}


def values_of_their_own():
    """
        What every setting holds itself: "none" for one that follows
        the setting above it, "follows (tf32 at the top)" for the two
        of cuDNN as PyTorch 2.13 and later start them. PyTorch
        reports resolved values only, so the settings above are taken
        away for the reading: 'generic' follows nothing, a backend
        reads its own value while 'generic' is "none", an operation
        its own while backend and 'generic' are. The settings above
        are put back to the values just read.
    """
    generic = read("generic", "all")
    write("generic", "all", "none")
    backends = {name: read(name, "all") for name in ("cuda", "mkldnn")}
    for name in backends:
        write(name, "all", "none")
    out = {("generic", "all"): generic}
    out.update({(name, "all"): value for name, value in backends.items()})
    for node in NODES[3:]:
        value = read(*node)
        if value != "none":
            # a value of its own, or does it follow one above it?
            write(node[0], "all", "tf32" if value == "ieee" else "ieee")
            if read(*node) != value:
                value = "follows ({} at the top)".format(value)
            write(node[0], "all", "none")
        out[node] = value
    for name, value in backends.items():
        write(name, "all", value)
    write("generic", "all", generic)
    return out


def under(generic):
    """Every setting as it resolves when 'generic' is given a value"""
    was = read("generic", "all")
    write("generic", "all", generic)
    out = resolved()
    write("generic", "all", was)
    return out


def state():
    if not NEWER:
        return (b.cudnn.allow_tf32, b.cuda.matmul.allow_tf32,
                torch.get_float32_matmul_precision())
    return {"older": older(), "resolved": resolved(),
            "own": values_of_their_own(), "generic ieee": under("ieee"),
            "generic tf32": under("tf32")}


def start_over(cudnn=False):
    """
        Every switch as PyTorch starts, except the two settings of
        cuDNN (which cannot be given their starting state again):
        those are left alone, or set by the older flag
    """
    torch.set_float32_matmul_precision("highest")
    if NEWER:
        for node in NODES:
            if node not in CUDNN:
                write(node[0], node[1], "none")
    if cudnn or not NEWER:
        b.cudnn.allow_tf32 = True


def setting(backend, kind, value):
    write(backend, kind, value)


def flag(of, value):
    setattr(of, "allow_tf32", value)


def matmul(value):
    torch.set_float32_matmul_precision(value)


def check(name, setup, cudnn=False):
    """
        `setup`: what the caller did to a PyTorch that had just
        started (None: nothing at all, the process as it is)
    """
    if setup is not None:
        start_over(cudnn)
        try:
            setup()
        except RuntimeError as error:
            # (a precision this build of PyTorch does not have)
            print("not possible here:", name, "-",
                  str(error).split("\n")[0])
            return
    before = state()
    with attribution._FullPrecision():
        inside = resolved() if NEWER else state()
    after = state()
    assert after == before, (name, before, after)
    if NEWER:
        assert set(inside.values()) == {"ieee"}, (name, inside)
    else:
        assert inside == (False, STUCK, "highest"), (name, inside)
    # as it was after a failure inside, too
    try:
        with attribution._FullPrecision():
            raise KeyError("no scores today")
    except KeyError:
        pass
    assert state() == before, (name, before, state())
    print("as before:", name)


print("switches:", "newer" if NEWER else "older")
at_start = state()
# before PyTorch 2.0 the environment variable keeps TF32 on for the
# matrix products of a GPU, whatever is set
STUCK = False
if not NEWER:
    torch.set_float32_matmul_precision("highest")
    STUCK = b.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision(at_start[2])
    assert state() == at_start
assert OVERRIDE or not STUCK
if OVERRIDE:
    # TF32 is on for matrix products from the start: as the precision
    # "high", or (before PyTorch 2.0) in the flag alone, and stuck
    assert (at_start[1:] if not NEWER else (
        at_start["older"]["flag of matmul"],
        at_start["older"]["matmul precision"])) == (
            True, "highest" if STUCK else "high"), at_start
check("as PyTorch starts", None)
assert state() == at_start
# the state the checks below start from: as PyTorch starts without
# the environment variable
start_over()
usual = state()
assert OVERRIDE or usual == at_start
check("nothing set", lambda: None)
check("matmul precision high", lambda: matmul("high"))
check("matmul precision medium", lambda: matmul("medium"))
check("flag of matmul on", lambda: flag(b.cuda.matmul, True))
if NEWER:
    # the two settings of cuDNN are not written by any of these: with
    # PyTorch 2.13 and later they stay as they start, following what
    # is above them
    for value in ("ieee", "tf32", "bf16"):
        check("generic " + value,
              lambda value=value: setting("generic", "all", value))
        check("mkldnn " + value,
              lambda value=value: setting("mkldnn", "all", value))
    for value in ("ieee", "tf32"):
        check("cuda " + value,
              lambda value=value: setting("cuda", "all", value))
    check("matmul of cuda alone tf32",
          lambda: setting("cuda", "matmul", "tf32"))
    check("matmul of cuda bfx9", lambda: setting("cuda", "matmul", "bfx9"))
    check("matmul of mkldnn alone bf16",
          lambda: setting("mkldnn", "matmul", "bf16"))
    check("convolutions of mkldnn bf16",
          lambda: setting("mkldnn", "conv", "bf16"))
    check("generic bf16, convolutions of mkldnn ieee", lambda: (
        setting("generic", "all", "bf16"), setting("mkldnn", "conv", "ieee")))
    check("generic tf32, cuda ieee, matmul of cuda tf32", lambda: (
        setting("generic", "all", "tf32"), setting("cuda", "all", "ieee"),
        setting("cuda", "matmul", "tf32")))
    check("generic ieee, mkldnn bf16", lambda: (
        setting("generic", "all", "ieee"), setting("mkldnn", "all", "bf16")))
    # (PyTorch refuses to report the matmul precision in this one)
    check("matmul precision medium, then matmul of mkldnn tf32", lambda: (
        matmul("medium"), setting("mkldnn", "matmul", "tf32")))
    assert older()["matmul precision"] == "not reported"
    check("matmul precision high, then generic ieee", lambda: (
        matmul("high"), setting("generic", "all", "ieee")))
    start_over()
    assert state() == usual, "the settings of cuDNN were written"

    # the review's cases, spelled out. A reduced precision above: what
    # follows it before follows it after
    setting("generic", "all", "bf16")
    with attribution._FullPrecision():
        pass
    setting("generic", "all", "ieee")
    assert read("mkldnn", "matmul") == read("mkldnn", "conv") == "ieee"
    assert read("cuda", "conv") == under("ieee")[("cuda", "conv")]
    start_over()
    # and the settings of cuDNN follow as they did when PyTorch started
    with attribution._FullPrecision():
        pass
    assert under("ieee") == usual["generic ieee"]
    assert state() == usual

    # a PyTorch that does not take a setting is not computed with
    setting("mkldnn", "conv", "bf16")
    before = state()

    def deaf(backend, kind, value):
        if (backend, kind) != ("mkldnn", "conv"):
            write(backend, kind, value)

    torch._C._set_fp32_precision_setter = deaf
    try:
        with attribution._FullPrecision():
            raise AssertionError("computed with less")
    except RuntimeError as error:
        assert "does not take the precision settings" in str(error) and \
            "mkldnn conv: bf16" in str(error), error
    finally:
        torch._C._set_fp32_precision_setter = write
    assert state() == before
    print("as before:", "a setting that is not taken")

# from here on the settings of cuDNN are written (by the older flag,
# or one by one)
check("flag of cuDNN on", lambda: None, cudnn=True)
check("flag of cuDNN off", lambda: flag(b.cudnn, False), cudnn=True)
check("both flags on", lambda: flag(b.cuda.matmul, True), cudnn=True)
check("matmul precision medium, then the flag of matmul off", lambda: (
    matmul("medium"), flag(b.cuda.matmul, False)), cudnn=True)
if NEWER:
    check("convolutions ieee, rnn tf32",
          lambda: setting("cuda", "conv", "ieee"), cudnn=True)
    check("convolutions tf32 after the flag was off", lambda: (
        flag(b.cudnn, False), setting("cuda", "conv", "tf32")), cudnn=True)
    check("flag of cuDNN off, generic tf32", lambda: (
        flag(b.cudnn, False), setting("generic", "all", "tf32")), cudnn=True)
    check("flag of cuDNN off, cuda tf32, rnn ieee", lambda: (
        flag(b.cudnn, False), setting("cuda", "all", "tf32"),
        setting("cuda", "rnn", "ieee")), cudnn=True)

# computing on a GPU. With TF32 that cannot be switched off nothing
# is computed there, and what was set on the way in is put back; on
# the CPU it is computed (TF32 is a matter of GPUs)
for precision in ("highest", "high", "medium"):
    for cudnn in (True, False):
        start_over(cudnn=True)
        matmul(precision)
        flag(b.cudnn, cudnn)
        before = state()
        try:
            with attribution._FullPrecision(torch.device("cuda")):
                inside = resolved() if NEWER else state()
        except RuntimeError as error:
            inside = None
            assert STUCK and "keeps TF32 on for matrix products on a " \
                "GPU" in str(error) and "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" \
                in str(error), error
        assert (inside is None) == bool(STUCK), inside
        if NEWER:
            assert set(inside.values()) == {"ieee"}, inside
        elif not STUCK:
            assert inside == (False, False, "highest"), inside
        assert state() == before, (precision, cudnn, before, state())
print("as before:", "a GPU, TF32 stuck" if STUCK else "a GPU")

if not NEWER:
    # a cuDNN whose TF32 flag does not go off (made up: the flag reads
    # True whatever is set): nothing is computed on a GPU either
    start_over()
    kind = type(b.cudnn)
    real = kind.__dict__["allow_tf32"]
    kind.allow_tf32 = property(lambda self: True, lambda self, value: None)
    try:
        try:
            with attribution._FullPrecision(torch.device("cuda")):
                raise AssertionError("computed with TF32")
        except RuntimeError as error:
            assert "keeps TF32 on for convolutions" in str(error), error
        with attribution._FullPrecision():
            assert torch.get_float32_matmul_precision() == "highest"
    finally:
        kind.allow_tf32 = real
    assert state() == (True, STUCK, "highest"), state()
    print("as before:", "a cuDNN flag that stays on")

# the autocast of the caller: off inside, on again afterwards
with torch.autocast("cpu", dtype=torch.bfloat16):
    with attribution._FullPrecision():
        assert torch.ones(2, 2).mm(torch.ones(2, 2)).dtype == torch.float32
    assert torch.ones(2, 2).mm(torch.ones(2, 2)).dtype == torch.bfloat16
    try:
        with attribution._FullPrecision(torch.device("cpu")):
            assert torch.ones(2, 2).mm(torch.ones(2, 2)).dtype == \
                torch.float32
            raise KeyError("no scores today")
    except KeyError:
        pass
    assert torch.ones(2, 2).mm(torch.ones(2, 2)).dtype == torch.bfloat16
print("autocast: off inside, on again afterwards")
print("done")
'''


@pytest.mark.parametrize("override", [False, True], ids=[
    "as PyTorch starts", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"])
def test_precision_is_full_inside_and_as_it_was_afterwards(override):
    """
        Inside the context in which the scores are computed every
        setting that decides the precision of a float32 computation
        says full precision; afterwards each switch is what it was,
        however the caller had set them: the matmul precision, the
        older TF32 flags, the newer settings per backend and
        operation with and without values of their own, and mixtures
        that PyTorch itself refuses to report. A setting that
        followed the one above it follows it again. With the
        environment variable that switches TF32 on for matrix
        products: the same; where PyTorch (before 2.0) then keeps it
        on whatever is set, nothing is computed on a GPU.
    """

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=repo, CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    if override:
        env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", PRECISION], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=900)
    assert done.returncode == 0, done.stdout[-4000:]
    lines = done.stdout.splitlines()
    assert lines[-1] == "done", done.stdout[-2000:]
    assert "autocast: off inside, on again afterwards" in lines
    checked = [line for line in lines if line.startswith("as before:")]
    assert "as before: as PyTorch starts" in checked
    if "switches: newer" in lines:
        assert len(checked) >= 31, checked
        assert "as before: a setting that is not taken" in checked
        assert "as before: matmul precision medium, then matmul of " \
            "mkldnn tf32" in checked
        assert "as before: a GPU" in checked
    else:
        assert "switches: older" in lines
        assert len(checked) == 11, checked
        assert "as before: a cuDNN flag that stays on" in checked
        # (PyTorch 2.0 to 2.8 switch TF32 off also with the variable)
        import torch
        stuck = override and int(torch.__version__.split(".")[0]) < 2
        assert ("as before: a GPU, TF32 stuck" in checked) == stuck
        assert ("as before: a GPU" in checked) == (not stuck)
