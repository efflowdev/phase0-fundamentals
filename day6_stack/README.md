# Day 6 — The local stack, and three wrong measurements

The stack that every later project runs against: Postgres with pgvector, Qdrant, the app itself, and
Ollama reachable but deliberately not containerised. Around it, the three things that make container
work different under an AI workload — a build whose cache behaviour actually matters, model weights
treated as data rather than code, and a memory limit met on purpose so its failure signature is
familiar before it arrives uninvited.

Every claim below is a number from `runs/*.json`. The reason to say that out loud is that three of
this day's answers were wrong on the first pass, and all three were wrong inside the measuring code
rather than the measured code — the same failure mode as day 5, twice as often.

**Layer ordering is half of the trick, and the stage boundary hands the other half back.** The usual
advice — install dependencies from the lockfile before copying source — is necessary and does not
finish the job. Three Dockerfiles, identical application, seconds:

| scenario | naive | multi-stage | split |
|---|---|---|---|
| cold, nothing cached | 21.9 | 17.2 | 14.6 |
| nothing changed | 1.2 | 1.1 | 1.2 |
| **one source line changed** | 17.8 | 9.3 | **1.0** |
| dependency layer invalidated | 17.4 | 12.1 | 10.3 |
| `--no-cache`, uv cache mount warm | 18.2 | 14.1 | 11.8 |
| image, unpacked MB | 875.0 | 530.0 | 530.0 |

`multi-stage` is the textbook build and it ends `COPY --from=builder /app /app`. `docker history`
shows what that produces: **one 244 MB layer holding the virtualenv and the source together**, so a
six-kilobyte edit re-copies a quarter of a gigabyte. The layer ordering saved the dependency
resolution and the stage boundary gave the saving straight back. `split` copies the virtualenv from a
stage that never sees a `.py` file — its layer is byte-identical across source edits and stays
cached — and copies the source separately. That is the whole difference between 9.3s and 1.0s, and
the images it produces are the same size.

The price is in `docs/decisions.md`: `phase0` ends up on `PYTHONPATH` instead of installed, so it has
no distribution metadata inside the image, and the package is installed locally but not in the
container. Nothing here needs metadata today. That is a property to re-check, not a property to
forget.

**The multi-stage split is worth 345 MB, and not for the reason it is usually sold.** 875 MB unpacked
down to 530. The saving is the `uv` binary and the build leftovers — not a compiler toolchain,
because `uv` installs pre-built wheels and there is no toolchain to leave behind. The pattern earns
much more than this on an image that compiles anything, and much less than its reputation on one that
does not.

**Image size is two numbers and reporting one of them is a category error.** `docker image inspect`
says 113.9 MB; `docker images` says 530.0 MB, for the same image. Neither is wrong: colima runs the
containerd image store, so inspect returns compressed content-store bytes and the listing returns the
unpacked size. Compressed is what crosses the network on a pull; unpacked is what the disk holds.
`measure.py` reports both, because the first version reported whichever one `inspect` returned and
called it "image size".

**A container is a packaging boundary, not a portability guarantee.** Docker on macOS is a Linux VM
with no path to Metal, so a containerised Ollama runs on CPU regardless of what the hardware is.
Identical weights, identical prompt, seed 42, temperature 0, 15 runs each:

| | tok/s median | range | wall |
|---|---|---|---|
| host, Metal | 54.04 | 53.60–55.31 | 2.68s |
| container, CPU | 13.44 | 8.81–14.95 | 11.93s |

**4.02x**, which is the entire argument for keeping Ollama on the host and the containerised service
behind a profile. It is also the concrete version of an abstract point: for inference the accelerator
sits outside the container, and putting it inside is a device-passthrough problem
(`nvidia-container-toolkit`, `--gpus all`) rather than something the image can solve.

Two smaller things fell out of it. The first pass reported **7.0x**, using the mean — cold-start
outliers on the CPU arm dragged it, and the summary statistic moved the headline by 75%. And the two
daemons return different token counts, 135 against 159, from an identical seed at temperature 0,
because the CPU and Metal backends order their arithmetic differently and the divergence compounds.
Day 3's finding, met again one layer down. Throughput is per-token so the comparison survives it; an
exact-output comparison would not have.

**Cross-architecture emulation costs 1.45x, which is cheaper than expected.** `--platform
linux/amd64` through Rosetta builds in 21.1s against 14.6s native, and the image reports `x86_64` and
runs. Worth knowing before deploying Apple Silicon work to x86 Linux, and worth knowing that the
wheels are the risk rather than the speed.

**A memory limit is a band, not a threshold, and the process says nothing on the way down.** Day 4's
500-document embedding job, run in the container under `mem_limit`, three times per limit:

| mem_limit | outcome | died during | peak RSS |
|---|---|---|---|
| 256m | killed, killed, killed | load, load, **batch** | — |
| 320m | **survived**, killed, killed | — , batch, batch | **353 MB** when it survived |
| 384m | survived ×3 | — | 358 MB |
| 512m | survived ×3 | — | 356–360 MB |

Three things in that table, in ascending order of how much they change how you would size a
container.

The kill itself is invisible from inside. No exception, no traceback, no final line — the kernel
sends SIGKILL and the process stops mid-sentence. The only evidence is outside: `OOMKilled=true` and
exit 137. At 1am on an embedding service this reads as a hang, not a failure.

The same limit does not produce the same outcome. 320m survived once in three here; separate sweeps
have put it as high as three in five. And at 256m the *shape* moved too — twice it died during model
load, a step, and once it got as far as document 200, a ramp. A single run per limit reports a
threshold that does not reproduce, which is why `oom_sweep.py` takes `--repeat`.

And the surviving run at 320m peaked at **353 MB — 33 MB above its own ceiling**. Resident set is not
the quantity the limiter enforces: most of that RSS is the memory-mapped ONNX model, and clean
file-backed pages are evicted under pressure rather than counted at the moment of the kill. So peak
RSS over-estimates what the container needs, in the direction that looks responsible — size from it,
add headroom on top, and pay for memory the workload never used. The honest number here is that the
job wants ~360 MB and is stable from 384m, so 512m is the setting.



**Alpine fails twice, and the arrangement decides which failure you get.** Alpine is musl libc and
every other image here is glibc:

| Alpine in | fails at | when | the message |
|---|---|---|---|
| the deps stage | resolution | 0.5s, during build | uv names `musllinux_1_2_aarch64` and lists every platform that *does* have a wheel |
| the runtime stage | import | in production, on a green build | `ModuleNotFoundError` for a module that is physically on disk |

`onnxruntime` publishes manylinux wheels and no source distribution, so on musl there is nothing to
install — not slow, impossible. The second case is worse in every way: `uv sync` runs on Debian and
resolves fine, the image builds, and then a virtualenv of glibc-linked objects meets a musl
interpreter. CPython looks for `.cpython-312-aarch64-linux-musl.so`, the file on disk is
`...-linux-gnu.so`, and the import machinery never opens it. numpy's own diagnostic then suggests
checking the Python version, checking the NumPy version, and suspecting a bad install — three things
that are all fine.

A third break sits in front of both: BusyBox has `adduser`, not `useradd`. Alpine is not
Debian-but-smaller, and the conclusion is the reverse of the reason people reach for it — chosen to
shrink an image, it either cannot install the scientific Python stack at all or produces a smaller
image that cannot run. `slim-bookworm` is the correct base, not a compromise.
`Dockerfile.alpine` is committed, broken on purpose, as the reproduction.

## What was wrong, and it was the instruments again

Three times, and none of them failed loudly.

The **control was rigged in my own favour**. `Dockerfile.naive` did not set `UV_COMPILE_BYTECODE=1`
while the other two did, so it was measured doing less work. Uncorrected, it reported the naive
build *beating* the optimised one on a source change, 8.9s against 9.6s — a result that falsifies
this day's entire thesis, produced by the harness rather than by Docker.

The **size metric compared a pull cost against a disk cost**, as above.

And `oom_sweep.py` **concatenated `stdout + stderr`** instead of letting the OS interleave them,
which appends every stderr line after every stdout line regardless of when they happened. The "last
line before the kill" was therefore reliably onnxruntime's startup warning, and every classification
of *where* the process died was wrong. It now runs with `stderr=subprocess.STDOUT`.

Day 5 ended on the same note and it is worth restating more sharply: the code that measures is where
the bugs live, because its output is a plausible number instead of a stack trace. Two of these three
would have been written up as findings.

## Reproducing

Requires colima (or any Docker), and the host Ollama running for the benchmark.

```bash
docker compose -f day6_stack/docker-compose.yml up -d          # postgres, qdrant, app
uv run python day6_stack/healthcheck.py                        # from the host
docker compose -f day6_stack/docker-compose.yml run --rm app   # the same script, inside

uv run python day6_stack/measure.py                            # the build table, ~6 min
uv run python day6_stack/oom_sweep.py --repeat 3               # the OOM band
uv run python day6_stack/bench_ollama.py --runs 15             # needs --profile full ollama
uv run pytest                                                  # 128 tests
```

Two things that will make the stack look broken and are not:

- `init/01-extensions.sql` runs **only** on an empty data directory. Editing it does nothing until
  `docker compose down -v`.
- `docker compose up` starts Postgres, Qdrant and the app — not Ollama. The phase-0 done bar reads as
  one command for all three; here it is `docker compose --profile full up`, and that is a decision
  rather than an omission.
