<div align="center">
  <img src="https://i.ibb.co/RptKysH8/image.png" alt="UnoGPT, a GPT written completely from scratch in pure Python" width="100%" />
</div>

<div align="center">

# UnoGPT
<span style="color:lightgreen">

### No PyTorch. No NumPy. No dependencies. Just Pure Python.
 </span>
</div>

<p align="center">
  <a href="https://github.com/aryanshrai03/UnoGPT"><img alt="Version 1.0" src="https://img.shields.io/badge/version-1.0-E74C3C?style=for-the-badge" /></a>
  <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="Zero dependencies" src="https://img.shields.io/badge/dependencies-zero-00B894?style=for-the-badge" />
  <img alt="5.01M parameters" src="https://img.shields.io/badge/parameters-5.01M-6C5CE7?style=for-the-badge" />
  <a href="license.txt"><img alt="License: UnoGPT Open 1.0" src="https://img.shields.io/badge/license-UnoGPT%20Open%201.0-FDCB6E?style=for-the-badge" /></a>
  <img alt="Made with Python" src="https://img.shields.io/badge/made%20with-Python-FFD43B?style=for-the-badge&logo=python&logoColor=black" />
</p>


---

UnoGPT is a decoder-only transformer with its own reverse-mode autodiff engine, tokenizer, dataset generator, trainer, sampler, and chat CLI. Everything lives in `unogpt.py`, everything runs on the standard library, and every operator's backward pass is verified against numerical gradients.

There is no PyTorch, no NumPy, and no pretrained checkpoint. It's an arithmetic chat model: ask it to add, subtract, multiply, divide, square, or exponentiate numbers and it answers one token at a time, straight out of the network. No `eval()`, no calculator, no answer lookup hidden somewhere in the source. Full config: 5,007,872 parameters.

If you've ever wanted to read a real GPT, this is your chance. Every matrix multiply, attention score, gradient, and optimizer update in here is ordinary Python you can step through.

## Quick start

Python 3.8+ is all you need (tested on 3.13). No pip install, no GPU, no build step.

```bash
git clone https://github.com/aryanshrai03/UnoGPT.git
cd UnoGPT

python unogpt.py info --materialize                 # real parameter counts for every config
python unogpt.py gradcheck --config tiny            # verify all 12 operators + the full model
python unogpt.py generate-data --samples 100000     # build the dataset (seed 1234, deterministic)
python unogpt.py train --config micro --steps 500   # smoke training, minutes on a CPU
python unogpt.py chat --config micro                # chat with the checkpoint
```

> The repo ships code and data, not pretrained weights. A 500-step `micro` run proves the pipeline learns, it does not give reliable answers yet. For that, train `nano` for a few hours.

## What's inside

- **Autodiff engine.** Reverse mode over flat `array('d')` buffers, hand-written analytic backward for every operator, intermediates freed as soon as they're consumed.
- **The model.** Pre-norm blocks with RMSNorm (eps 1e-6), RoPE (theta 10000), causal multi-head attention, SwiGLU feed-forward, and tied input/output embeddings.
- **Tokenizer.** Character level, vocabulary of 85: 81 printable characters plus `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- **Dataset.** A procedural conversational-arithmetic generator, seeded and fully reproducible.
- **Trainer.** Adam with linear warmup (60 steps) and cosine decay to 10% of peak LR, global-norm clip 1.0, optional gradient accumulation, atomic checkpoints, and `--resume` that picks up the same run instead of starting over.
- **Sampler.** KV-cached generation with temperature, top-k, top-p, repetition penalty, and EOS stopping.
- **Tooling.** Chat CLI, exact-match evaluation, throughput benchmark, and a 5-point central-difference gradient checker.

## Configurations

| Config | Parameters | Layers | d_model | Heads | FFN | Ctx | Good for |
|---|---:|---:|---:|---:|---:|---:|---|
| **full** | 5,007,872 | 4 | 256 | 4 | 1,280 | 256 | The real 5M spec, days to weeks of CPU training |
| **nano** | 137,216 | 2 | 64 | 4 | 256 | 160 | A practical demo, hours |
| **micro** | 59,904 | 2 | 48 | 3 | 128 | 128 | Smoke tests, minutes |
| **tiny** | 4,016 | 1 | 16 | 2 | 32 | 24 | Gradient checking |

`full` is the default everywhere and is never silently swapped for a smaller model. `info --materialize` prints the true count for each configuration.

## Training

```bash
python unogpt.py train --config micro --steps 500     # minutes, sanity check
python unogpt.py train --config nano --steps 3000     # hours, practical demo
python unogpt.py train --config full                  # the 5M spec, ETA printed up front
python unogpt.py train --config nano --resume         # continue an interrupted run
```

Checkpoints carry the parameters, Adam moments, step counter, tokenizer, validation history, and the batch RNG state, so a resumed run continues exactly where it stopped.

### Measured smoke run

`micro`, 60 steps, batch size 2, lr 3e-3, one laptop CPU:

| Step | Train loss | Val loss |
|---:|---:|---:|
| 10 | 4.0859 | |
| 20 | 3.4028 | |
| 30 | 3.0115 | 2.7478 |
| 40 | 3.1777 | |
| 50 | 2.9225 | |
| 60 | 2.9327 | 2.3958 |

Loss goes down and validation follows it. That's what this run proves: the data pipeline, forward pass, backward pass, optimizer, and checkpoints all work end to end. It is not evidence of a capable model yet.

## Chat

```bash
python unogpt.py chat --config nano
python unogpt.py chat --checkpoint unogpt_nano.ckpt
python unogpt.py generate --config nano --prompt "What is 5 + 5?"
```

```text
You: What is 12 + 30?
UnoGPT: 12 + 30 = 42.
You: What is 9 squared?
UnoGPT: 9^2 = 81.
You: /exit
UnoGPT: Bye!
```

Inside the chat you get `/help`, `/stats`, `/clear`, `/save [file]`, and `/exit` (also `/quit` and `/q`). Generation is controlled with `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--max-new`, and `--history-turns`.

## Evaluation

```bash
python unogpt.py evaluate --config nano --count 12
```

Exact-match scoring on two tables: a held-out test split drawn from the training distributions, and a generalization probe with magnitudes deliberately pushed outside the training ranges. The probe score is expected to be much lower, and it's reported as measured, not buried.

## Gradient checking

`python unogpt.py gradcheck --config tiny` compares every operator's hand-derived backward against a 5-point central-difference numerical gradient, relative tolerance 2e-4, then checks the full model end to end. This table is what the checker actually prints:

| Check | Verifies | Max relative error |
|---|---|---:|
| `mm` | Dense matrix multiply + bias | 3.86e-12 |
| `mmt` | Tied output projection | 1.58e-11 |
| `silu` | SiLU / SwiGLU activation | 4.03e-11 |
| `add` | Tensor + tensor, tensor + row | 2.35e-12 |
| `elemwise` | Multiply, scale, divide, power, exp, log | 1.26e-11 |
| `softmax` | Row-wise softmax | 5.76e-11 |
| `logsoftmax` | Row-wise log-softmax | 1.37e-10 |
| `rmsnorm` | RMSNorm forward/backward | 1.81e-11 |
| `rope` | Rotary position embeddings | 6.32e-12 |
| `attn` | Causal multi-head attention | 8.22e-11 |
| `emb` | Embedding lookup + scatter-add | 1.48e-05 |
| `crossentropy` | Masked cross-entropy | 2.36e-11 |
| **Full model** | End-to-end loss on `tiny` | **1.45e-05** |

All 13 pass. The checker also verifies the causal mask (future tokens never touch earlier positions) and the tokenizer round-trip.

## Dataset

```bash
python unogpt.py generate-data --samples 100000
```

Generated deterministically with seed `1234` and split with no leakage into **97,000 train / 1,500 validation / 1,500 test** examples. Each one is a two-line conversation, and the loss mask covers only the answer tokens, so the model learns to respond instead of repeating the question:

```text
User: What is 12 + 30?
UnoGPT: 12 + 30 = 42.
```

| Category | Train examples | What it teaches |
|---|---:|---|
| `add` | 26,088 | Addition, negatives and decimals included |
| `sub` | 18,120 | Subtraction |
| `mul` | 20,432 | Multiplication |
| `div` | 15,806 | Division with clean decimal answers |
| `square` | 8,593 | Squares of integers and decimals |
| `pow` | 2,293 | Integer powers |
| `ood` | 5,303 | Polite refusal of off-topic questions |
| `conv` | 365 | Greetings, thanks, goodbyes, help |
| **Total** | **97,000** | |

## Performance and limits

Pure Python means pure CPU speed: on the order of tens of MFLOP/s on a typical laptop. `python unogpt.py benchmark --config full` measures your machine and prints a real ETA instead of a guess.

- `micro` for smoke tests, `nano` for a practical demo, `full` for the long haul.
- UnoGPT is an arithmetic specialist, not a general assistant, and it will confidently get things wrong outside its training ranges.
- There is no safety or moderation layer, so keep it away from real users.
- Slow inference is the price of zero dependencies. That trade is the whole point.

## Project structure

```text
unogpt/
├── unogpt.py          # the entire implementation, about 3,700 lines
├── README.md
├── license.txt        # UnoGPT Open License 1.0
└── data/
    ├── train.json     # 97,000 examples
    ├── val.json       # 1,500 examples
    └── test.json      # 1,500 examples
```

## License

**UnoGPT Open License 1.0 (UOL-1.0).** Anyone may use, copy, modify, merge, publish, distribute, sublicense, or sell UnoGPT, commercially included. The single condition: keep the original [`license.txt`](license.txt), the copyright notice, and the attribution notice in your codebase and in anything you distribute, and mark modified versions as modified. Full terms are in [`license.txt`](license.txt).


## Star History

### If UnoGPT was useful, Please star the Repo! ⭐

[![Star History Chart](https://api.star-history.com/chart?repos=aryanshrai03/UnoGPT&type=date&legend=top-left)](https://www.star-history.com/?repos=aryanshrai03%2FUnoGPT&type=date&legend=top-left)


<div align="center">

## 
### Made by Aryansh Rai

<a href="https://github.com/aryanshrai03">
  <img alt="GitHub: @aryanshrai03" src="https://img.shields.io/badge/GitHub-%40aryanshrai03-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>
<a href="https://github.com/aryanshrai03/UnoGPT">
  <img alt="Repository: UnoGPT" src="https://img.shields.io/badge/Project-UnoGPT-6C5CE7?style=for-the-badge&logo=github&logoColor=white" />
</a>



© 2026 Aryansh Rai, licensed under the [UnoGPT Open License 1.0](license.txt)

</div>
