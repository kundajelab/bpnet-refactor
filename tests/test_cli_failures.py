"""
    The command line tools must report failure through their exit
    status: pipelines (WDL, shell scripts with `set -e`) rely on it
"""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time

import pytest

pytest.importorskip("tensorflow")

from tests.test_cli_end_to_end import REPO, run_cli

pytestmark = pytest.mark.slow


def train_args(dataset, model_dir, **overrides):
    values = {
        "--input-data": dataset["input_json"],
        "--output-dir": model_dir,
        "--reference-genome": dataset["fasta"],
        "--chrom-sizes": dataset["chrom_sizes"],
        "--splits": dataset["splits_json"],
        "--model-arch-name": "BPNet",
        "--model-arch-params-json": dataset["params_json"],
        "--sequence-generator-name": "BPNet",
        "--model-output-filename": "model",
        "--input-seq-len": dataset["input_len"],
        "--output-len": dataset["output_len"],
        "--threads": 2, "--epochs": 1, "--batch-size": 16,
    }
    values.update(overrides)
    args = ["--chroms"] + dataset["chroms"]
    for key, value in values.items():
        args.extend([key, value])
    return args


def test_training_process_failure_gives_an_error_status(synth_dataset,
                                                        tmp_path):
    """
        The model parameters are read inside the training process; when
        that process fails, bpnet-train must fail too
    """

    broken = tmp_path / "broken_params.json"
    broken.write_text("{ this is not json")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    code, output = run_cli("bpnet.cli.bpnettrainer", train_args(
        synth_dataset, model_dir,
        **{"--model-arch-params-json": broken}), cwd=str(tmp_path))
    assert code != 0, output[-2000:]
    assert "Unable to load json file" in output
    assert "Training of split 0 failed" in output
    # no model was written
    assert not any(name.startswith("model") for name in os.listdir(
        str(model_dir)))


def test_training_stops_at_the_first_failed_split(synth_dataset, tmp_path):
    # two splits; the broken model parameter file makes the first one
    # fail, the second must not be started
    with open(synth_dataset["splits_json"]) as fh:
        split = json.load(fh)["0"]
    splits = tmp_path / "two_splits.json"
    splits.write_text(json.dumps({"0": split, "1": split}))
    broken = tmp_path / "broken_params.json"
    broken.write_text("{ this is not json")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    code, output = run_cli("bpnet.cli.bpnettrainer", train_args(
        synth_dataset, model_dir,
        **{"--model-arch-params-json": broken, "--splits": splits}),
        cwd=str(tmp_path))
    assert code != 0
    assert "Training of split 0 failed" in output
    # the second split was not started
    assert "Split #0" in output
    assert "Split #1" not in output
    # the message points at the directory that holds the logs
    assert "trainer logs in {}".format(model_dir) in output


def test_missing_splits_file_message(synth_dataset, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    missing = tmp_path / "no_such_splits.json"
    code, output = run_cli("bpnet.cli.bpnettrainer", train_args(
        synth_dataset, model_dir, **{"--splits": missing}),
        cwd=str(tmp_path))
    assert code != 0
    assert "File not found: {}".format(missing) in output


def test_failure_in_a_batch_worker_ends_the_training(tmp_path):
    """
        A signal file that does not cover the genome: the worker
        processes cannot read some of the loci. bpnet-train must stop
        with an error status and say why (it used to wait forever for
        the batches of the worker that failed).
    """

    import time

    from tests import synth

    dataset = synth.make_dataset(
        str(tmp_path / "data"), chrom_len=20000, n_peaks=60, n_background=60)
    path = dataset["signal"][0]
    short = {chrom: 6000 for chrom in dataset["chroms"]}
    synth.write_bigwig(path, short, {
        chrom: values[:6000]
        for chrom, values in dataset["tracks"][path].items()})

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    start = time.time()
    code, output = run_cli("bpnet.cli.bpnettrainer",
                           train_args(dataset, model_dir), cwd=str(tmp_path))
    assert code != 0, output[-2000:]
    assert "Batch generation failed in worker process" in output
    assert "Training of split 0 failed" in output
    # far below the time after which run_cli gives up on a hung tool
    assert time.time() - start < 300
    assert not any(name.startswith("model") for name in os.listdir(
        str(model_dir)))


# --------------------------------------------------------------------
# a tool that fails, or is interrupted, while its batch workers have
# work left
# --------------------------------------------------------------------

#: seconds a tool gets to reach the point of the test, and to end after
#: it
START_LIMIT = 300
END_LIMIT = 60

#: seconds within which a tool has to end. (Not at once: TensorFlow 2.7
#: or later can hold a failure back inside fit() until the batch that
#: is being fetched has arrived, or the one after it.)
PROMPT = 20

#: seconds a batch worker of these tests takes for every batch after
#: its first. With one locus per batch the workers of an epoch then
#: have minutes of work left when the tool fails.
SLOW = 2.0

#: for both drivers: batch workers (of the mode WORKERS_OF) that are
#: slow from their second batch on, and say so
WORKERS = """
import os, time
from bpnet.generators import generators

construct = generators.MBPNetSequenceGenerator._generate_batch


def slow(self, coords):
    self._made = getattr(self, '_made', 0) + 1
    if self._mode == os.environ['WORKERS_OF'] and self._made > 1:
        open('worker.{}'.format(os.getpid()), 'w').close()
        time.sleep(float(os.environ['SLOW']))
    return construct(self, coords)


generators.MBPNetSequenceGenerator._generate_batch = slow
"""

#: bpnet-train whose second training or validation step fails
TRAIN_AND_FAIL = WORKERS + """
import tensorflow as tf
from bpnet.cli import bpnettrainer


class Fails(tf.keras.callbacks.Callback):
    def fail(self, batch, what):
        if os.environ['FAIL_IN'] == 'loses an interrupt in ' + what and \
                batch == 1:
            # simulates what happens where TensorFlow discards the
            # KeyboardInterrupt (Keras' own batch hooks pass it on):
            # the training goes on
            try:
                open('waiting', 'w').close()
                time.sleep(120)
            except KeyboardInterrupt:
                open('lost', 'w').close()
        if os.environ['FAIL_IN'] == what and batch == 1:
            open('failed', 'w').close()
            raise RuntimeError('the {} step failed'.format(what))

    def on_train_batch_end(self, batch, logs=None):
        self.fail(batch, 'train')

    def on_test_batch_end(self, batch, logs=None):
        self.fail(batch, 'val')


def with_the_callback(call):
    def run(self, *args, **kwargs):
        # (after those of the tool)
        kwargs['callbacks'] = list(kwargs.get('callbacks') or []) + [Fails()]
        return call(self, *args, **kwargs)
    return run


tf.keras.Model.fit = with_the_callback(tf.keras.Model.fit)
tf.keras.Model.evaluate = with_the_callback(tf.keras.Model.evaluate)
bpnettrainer.main()
"""

#: bpnet-predict whose second call of the model fails
PREDICT_AND_FAIL = WORKERS + """
import tensorflow as tf
from bpnet.cli import predict

model_predict = tf.keras.Model.predict
calls = []


def fails_the_second_time(self, *args, **kwargs):
    calls.append(1)
    if len(calls) == 2 and os.environ.get('FAIL_IN') != 'never':
        open('failed', 'w').close()
        raise RuntimeError('the prediction failed')
    return model_predict(self, *args, **kwargs)


tf.keras.Model.predict = fails_the_second_time
predict.predict_main()
"""


#: bpnet-predict in which an interrupt gets lost during the second call
#: of the model: KeyboardInterrupt is raised and discarded, as Python
#: does when an interrupt arrives inside a destructor or a callback
PREDICT_AND_LOSE_AN_INTERRUPT = WORKERS + """
import tensorflow as tf
from bpnet.cli import predict

model_predict = tf.keras.Model.predict
calls = []


def loses_an_interrupt(self, *args, **kwargs):
    calls.append(1)
    open('predicted', 'w').close()
    if len(calls) == 2 and os.environ.get('LOSE_IN') != 'loading':
        try:
            open('waiting', 'w').close()
            time.sleep(120)
        except KeyboardInterrupt:
            open('lost', 'w').close()
            if os.environ['FAIL_IN'] == 'another error':
                # what TensorFlow made of an interrupt that arrived
                # inside its C code
                raise TypeError()
    return model_predict(self, *args, **kwargs)


load_model = predict.loading.load_model


def loses_an_interrupt_while_loading(path):
    try:
        open('waiting', 'w').close()
        time.sleep(120)
    except KeyboardInterrupt:
        open('lost', 'w').close()
    return load_model(path)


predict_all = predict.predict


def loses_an_interrupt_at_the_end(*args, **kwargs):
    predict_all(*args, **kwargs)
    try:
        open('waiting', 'w').close()
        time.sleep(120)
    except KeyboardInterrupt:
        open('lost', 'w').close()


tf.keras.Model.predict = loses_an_interrupt
if os.environ.get('LOSE_IN') == 'loading':
    predict.loading.load_model = loses_an_interrupt_while_loading
if os.environ.get('LOSE_IN') == 'after the batches':
    predict.predict = loses_an_interrupt_at_the_end
predict.predict_main()
"""


def processes_of(session):
    """
        The processes of a session that are still running
    """

    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/{}/stat".format(name)) as fh:
                state, _, _, sid = fh.read().rsplit(")", 1)[1].split()[:4]
        except (IOError, OSError, ValueError):
            continue
        if int(sid) == session and state != "Z":
            found.append(int(name))
    return found


def run_until(code, args, cwd, env, event, then=None):
    """
        Run `python -c <code> <args>` in a session of its own until
        `event(cwd)` is true, call `then(process)`, and wait for the
        tool to end

        Returns:
            tuple: (return code, output, seconds from the event to the
                end of the tool, processes of the session that were
                still running a moment later)

        The test fails if the event does not come or the tool does not
        end in time; everything the tool started is killed either way.
    """

    env = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES="",
               SLOW=str(SLOW), **env)
    log = os.path.join(cwd, "output.txt")
    with open(log, "w") as out:
        process = subprocess.Popen(
            [sys.executable, "-c", code] + [str(a) for a in args], cwd=cwd,
            env=env, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True)
    try:
        deadline = time.time() + START_LIMIT
        while not event(cwd) and process.poll() is None and \
                time.time() < deadline:
            time.sleep(0.1)
        with open(log) as fh:
            assert event(cwd), "the tool did not get there:\n" + \
                fh.read()[-3000:]
        reached = time.time()
        if then is not None:
            then(process)
        try:
            code = process.wait(timeout=END_LIMIT)
        except subprocess.TimeoutExpired:
            with open(log) as fh:
                pytest.fail("the tool was still running {} s later\n{}".format(
                    END_LIMIT, fh.read()[-3000:]))
        seconds = time.time() - reached
        # what the tool started has a moment to go away
        deadline = time.time() + 5
        while processes_of(process.pid) and time.time() < deadline:
            time.sleep(0.1)
        left = processes_of(process.pid)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        process.wait()
    with open(log) as fh:
        return code, fh.read(), seconds, left


def failed(cwd):
    return os.path.exists(os.path.join(cwd, "failed"))


def two_workers_have_work_left(cwd):
    return len([name for name in os.listdir(cwd)
                if name.startswith("worker.")]) == 2


needs_proc = pytest.mark.skipif(
    not os.path.isdir("/proc/self"),
    reason="needs /proc to look at other processes")


@needs_proc
@pytest.mark.parametrize("step", ["train", "val"])
def test_failed_step_ends_bpnet_train_and_its_workers(synth_dataset,
                                                      tmp_path, step):
    """
        A training or validation step fails while the batch workers of
        the epoch have minutes of work left. bpnet-train ends with the
        error of the step, without waiting for them, and leaves no
        process behind. (It used to end when the workers had made the
        rest of the epoch.)
    """

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    code, output, seconds, left = run_until(
        TRAIN_AND_FAIL, train_args(synth_dataset, model_dir,
                                   **{"--batch-size": 1}),
        str(tmp_path), {"WORKERS_OF": step, "FAIL_IN": step}, failed)
    assert code != 0, output[-3000:]
    assert "the {} step failed".format(step) in output
    assert "Training of split 0 failed" in output
    # the failure of the step is what is reported, not a batch worker
    assert "Batch generation failed" not in output
    print("ended {:.1f} s after the {} step failed".format(seconds, step))
    assert seconds < PROMPT
    assert left == []
    assert not any(name.startswith("model") for name in os.listdir(
        str(model_dir)))


@needs_proc
@pytest.mark.parametrize("to", ["bpnet-train alone", "its process group"])
def test_interrupt_of_bpnet_train_ends_its_processes(synth_dataset, tmp_path,
                                                     to):
    """
        An interrupt that reaches only the process that waits for the
        training process is passed on: the training process stops its
        batch workers and ends. (bpnet-train used to wait until the
        training was over.) An interrupt from a keyboard reaches all
        processes of the tool; the one that is passed on then comes
        on top, and does not disturb the cleanup.
    """

    def interrupt(process):
        if to == "bpnet-train alone":
            os.kill(process.pid, signal.SIGINT)
        else:
            os.killpg(process.pid, signal.SIGINT)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    code, output, seconds, left = run_until(
        TRAIN_AND_FAIL, train_args(synth_dataset, model_dir,
                                   **{"--batch-size": 1}),
        str(tmp_path), {"WORKERS_OF": "train", "FAIL_IN": "never"},
        two_workers_have_work_left, then=interrupt)
    assert code != 0, output[-3000:]
    # the interrupt of bpnet-train and the one it passed on to the
    # training process
    assert output.count("KeyboardInterrupt") >= 2, output[-3000:]
    print("ended {:.1f} s after the interrupt".format(seconds))
    assert seconds < PROMPT
    assert left == []


@pytest.fixture(scope="module")
def model_for_predictions(synth_dataset, tmp_path_factory):
    directory = tmp_path_factory.mktemp("model_for_predictions")
    code, output = run_cli("bpnet.cli.bpnettrainer", train_args(
        synth_dataset, directory), cwd=str(directory))
    assert code == 0, output[-3000:]
    return str(directory / "model_split000")


def predict_args(d, model, out_dir):
    return ["--model", model, "--chrom-sizes", d["chrom_sizes"],
            "--chroms"] + d["splits"]["0"]["test"] + [
            "--test-indices-file", "None", "--reference-genome", d["fasta"],
            "--output-dir", out_dir, "--input-data", d["input_json"],
            "--sequence-generator-name", "BPNet",
            "--input-seq-len", d["input_len"],
            "--output-len", d["output_len"],
            "--output-window-size", d["output_len"], "--batch-size", 1,
            "--threads", 2]


@needs_proc
def test_failure_in_bpnet_predict_ends_its_workers(
        synth_dataset, model_for_predictions, tmp_path):
    code, output, seconds, left = run_until(
        PREDICT_AND_FAIL,
        predict_args(synth_dataset, model_for_predictions, tmp_path),
        str(tmp_path), {"WORKERS_OF": "test"}, failed)
    assert code != 0, output[-3000:]
    assert "the prediction failed" in output
    assert "Batch generation failed" not in output
    print("ended {:.1f} s after the prediction failed".format(seconds))
    assert seconds < PROMPT
    assert left == []


@needs_proc
def test_interrupt_of_bpnet_predict_ends_its_workers(
        synth_dataset, model_for_predictions, tmp_path):
    code, output, seconds, left = run_until(
        PREDICT_AND_FAIL,
        predict_args(synth_dataset, model_for_predictions, tmp_path),
        str(tmp_path), {"WORKERS_OF": "test", "FAIL_IN": "never"},
        two_workers_have_work_left,
        then=lambda process: os.kill(process.pid, signal.SIGINT))
    assert code != 0, output[-3000:]
    assert "KeyboardInterrupt" in output
    print("ended {:.1f} s after the interrupt".format(seconds))
    assert seconds < PROMPT
    assert left == []


def waits_for_the_interrupt(cwd):
    return os.path.exists(os.path.join(cwd, "waiting"))


@needs_proc
def test_interrupt_that_python_loses_ends_bpnet_predict(
        synth_dataset, model_for_predictions, tmp_path):
    """
        Python raises KeyboardInterrupt wherever the main thread is
        when an interrupt arrives; inside a destructor or a callback
        (TensorFlow runs many) the exception is discarded and the
        tool used to go on. It notes the interrupt now, and ends
        before the next batch.
    """

    code, output, seconds, left = run_until(
        PREDICT_AND_LOSE_AN_INTERRUPT,
        predict_args(synth_dataset, model_for_predictions, tmp_path),
        str(tmp_path), {"WORKERS_OF": "test", "FAIL_IN": "never"},
        waits_for_the_interrupt,
        then=lambda process: os.kill(process.pid, signal.SIGINT))
    assert os.path.exists(str(tmp_path / "lost"))
    assert code != 0, output[-3000:]
    assert "KeyboardInterrupt" in output
    print("ended {:.1f} s after the interrupt".format(seconds))
    assert seconds < PROMPT
    assert left == []


@needs_proc
def test_interrupt_lost_before_the_batches_ends_bpnet_predict(
        synth_dataset, model_for_predictions, tmp_path):
    """
        An interrupt that is lost while the model is loaded ends the
        tool before its first batch: the loop over the batches looks
        at the notes the tool took from its start
    """

    code, output, seconds, left = run_until(
        PREDICT_AND_LOSE_AN_INTERRUPT,
        predict_args(synth_dataset, model_for_predictions, tmp_path),
        str(tmp_path), {"WORKERS_OF": "test", "FAIL_IN": "never",
                        "LOSE_IN": "loading"},
        waits_for_the_interrupt,
        then=lambda process: os.kill(process.pid, signal.SIGINT))
    assert os.path.exists(str(tmp_path / "lost"))
    assert code != 0, output[-3000:]
    assert "KeyboardInterrupt" in output
    assert not os.path.exists(str(tmp_path / "predicted"))
    assert seconds < PROMPT
    assert left == []


@needs_proc
def test_interrupt_lost_after_the_batches_fails_bpnet_predict(
        synth_dataset, model_for_predictions, tmp_path):
    """
        An interrupt that is lost after the last batch (while the
        files are written) does not let the tool end as if it had
        succeeded
    """

    code, output, seconds, left = run_until(
        PREDICT_AND_LOSE_AN_INTERRUPT,
        predict_args(synth_dataset, model_for_predictions, tmp_path),
        str(tmp_path), {"WORKERS_OF": "none", "FAIL_IN": "never",
                        "LOSE_IN": "after the batches"},
        waits_for_the_interrupt,
        then=lambda process: os.kill(process.pid, signal.SIGINT))
    assert os.path.exists(str(tmp_path / "lost"))
    assert code != 0, output[-3000:]
    assert "KeyboardInterrupt" in output
    assert seconds < PROMPT
    assert left == []


@needs_proc
@pytest.mark.parametrize("step", ["train", "val"])
def test_interrupt_lost_in_a_step_ends_bpnet_train(synth_dataset, tmp_path,
                                                   step):
    """
        The training process loses the interrupt that bpnet-train
        passes on to it (in a callback of a training or validation
        step, where TensorFlow discards the KeyboardInterrupt). It
        ends at the next batch all the same, instead of training on
        until bpnet-train terminates it
    """

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    code, output, seconds, left = run_until(
        TRAIN_AND_FAIL, train_args(synth_dataset, model_dir,
                                   **{"--batch-size": 1}),
        str(tmp_path), {"WORKERS_OF": step,
                        "FAIL_IN": "loses an interrupt in " + step},
        waits_for_the_interrupt,
        then=lambda process: os.kill(process.pid, signal.SIGINT))
    assert os.path.exists(str(tmp_path / "lost"))
    assert code != 0, output[-3000:]
    assert "KeyboardInterrupt" in output
    print("ended {:.1f} s after the interrupt".format(seconds))
    # (bpnet-train passes the interrupt on after 2 s, and terminates
    # its training process after another 30)
    assert seconds < PROMPT
    assert left == []
    assert not any(name.startswith("model") for name in os.listdir(
        str(model_dir)))


@needs_proc
def test_interrupt_that_becomes_another_error_is_reported_as_one(
        synth_dataset, model_for_predictions, tmp_path):
    """
        An interrupt that arrives inside the C code of TensorFlow can
        come out as an error of its own (a TypeError without a
        message was seen). bpnet-predict ended with that error; it
        reports the interrupt now, with the error as its cause.
    """

    code, output, seconds, left = run_until(
        PREDICT_AND_LOSE_AN_INTERRUPT,
        predict_args(synth_dataset, model_for_predictions, tmp_path),
        str(tmp_path), {"WORKERS_OF": "test", "FAIL_IN": "another error"},
        waits_for_the_interrupt,
        then=lambda process: os.kill(process.pid, signal.SIGINT))
    assert code != 0, output[-3000:]
    assert "TypeError" in output
    assert "KeyboardInterrupt: interrupted (the interrupt was reported " \
        "as: TypeError())" in output
    assert seconds < PROMPT
    assert left == []


def fails_in_training(how, queue):
    from bpnet.train import training

    def fails(*args, **kwargs):
        if how in ("interrupt reported as another error",
                   "interrupt lost in training"):
            try:
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(5)
            except KeyboardInterrupt:
                if how == "interrupt reported as another error":
                    raise TypeError()
                # (lost: the training goes on, and ends normally)
                return
        raise {"interrupt": KeyboardInterrupt, "error": RuntimeError}[how]()

    training.train_and_validate = fails
    before = signal.getsignal(signal.SIGINT)
    try:
        training.train_and_validate_or_stop()
        result = "no failure"
    except BaseException as failure:
        result = (type(failure).__name__, type(failure.__cause__).__name__)
    ignored = signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    # an interrupt while the process reports and ends does not end it
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(0.2)
    queue.put((result, before != signal.SIG_IGN, ignored))


@pytest.mark.parametrize("how,reported", [
    ("interrupt", ("KeyboardInterrupt", "NoneType")),
    ("error", ("RuntimeError", "NoneType")),
    ("interrupt reported as another error",
     ("KeyboardInterrupt", "TypeError")),
    ("interrupt lost in training", ("KeyboardInterrupt", "NoneType"))])
def test_training_process_on_its_way_out(how, reported):
    """
        The training process of a split that has failed, or was
        interrupted, stops its batch workers and ends with its report.
        The interrupt that bpnet-train passes on to it two seconds
        after its own must not cut that report short (it did on a
        busy machine): once the process is on its way out, it ignores
        interrupts. An interrupt that a library turned into an error
        of its own is reported as an interrupt, and one that was lost
        does not let the training end as if it had succeeded.
    """

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=fails_in_training,
                                      args=(how, queue))
    process.start()
    try:
        assert queue.get(timeout=START_LIMIT) == (reported, True, True)
        process.join(END_LIMIT)
        assert not process.is_alive(), "the training process did not end"
    finally:
        if process.is_alive():
            process.kill()
        process.join()


# --------------------------------------------------------------------
# what bpnet-train does with its training process when it is
# interrupted, on stand-ins for that process
# --------------------------------------------------------------------

def stand_in(kind, ready):
    if kind in ("ignores the interrupt", "ignores everything"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if kind == "ignores everything":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    if kind == "is ending":
        os._exit(3)
    try:
        time.sleep(600)
    except KeyboardInterrupt:
        # what a training process does: clean up, then fail
        os._exit(5)


#: the times of the escalation in these tests: long enough that a
#: busy machine does not blur the steps
NOTICE, STOP = 1.0, 2.0


@pytest.mark.parametrize("kind,exit_code,at_least", [
    # it had the interrupt too and is on its way out: nothing is sent
    ("is ending", 3, 0.0),
    # it ends when the interrupt is passed on
    ("takes the interrupt", 5, NOTICE),
    # it is terminated, then killed
    ("ignores the interrupt", -signal.SIGTERM, NOTICE + STOP),
    ("ignores everything", -signal.SIGKILL, NOTICE + 2 * STOP)])
def test_training_process_is_ended_step_by_step(monkeypatch, kind, exit_code,
                                                at_least):
    """
        The exit code of the stand-in says which step ended it, the
        time that the steps before it were waited for
    """

    from bpnet.train import training

    monkeypatch.setattr(training, "TRAINER_NOTICE_SECONDS", NOTICE)
    monkeypatch.setattr(training, "TRAINER_STOP_SECONDS", STOP)
    ready = multiprocessing.Event()
    process = multiprocessing.Process(target=stand_in, args=(kind, ready))
    process.start()
    try:
        assert ready.wait(60)
        start = time.monotonic()
        training.end_training_process(process)
        seconds = time.monotonic() - start
        assert not process.is_alive()
        assert process.exitcode == exit_code
        assert at_least <= seconds < at_least + 20
    finally:
        if process.is_alive():
            process.kill()
        process.join()


def interrupted_again(queue):
    """
        In a process of its own (it sends itself an interrupt): the
        training process is being ended when a second interrupt
        arrives
    """

    import threading

    from bpnet.train import training

    training.TRAINER_NOTICE_SECONDS, training.TRAINER_STOP_SECONDS = 0.5, 1.0
    handler = signal.getsignal(signal.SIGINT)
    ready = multiprocessing.Event()
    process = multiprocessing.Process(
        target=stand_in, args=("ignores everything", ready))
    process.start()
    ready.wait(60)
    threading.Timer(0.8, os.kill, args=(os.getpid(), signal.SIGINT)).start()
    try:
        training.end_training_process(process)
        result = "ended"
    except KeyboardInterrupt:
        result = "interrupted"
    queue.put((result, process.is_alive(), process.exitcode,
               signal.getsignal(signal.SIGINT) is handler))
    if process.is_alive():
        process.kill()


def test_second_interrupt_does_not_leave_the_training_process():
    queue = multiprocessing.Queue()
    child = multiprocessing.Process(target=interrupted_again, args=(queue,))
    child.start()
    try:
        result = queue.get(timeout=120)
    finally:
        child.join(30)
        if child.is_alive():
            child.kill()
            child.join()
    # the training process was killed, and interrupts are taken again
    assert result == ("ended", False, -signal.SIGKILL, True), result
