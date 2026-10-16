# Compiler-Aware Performance Modelling of NVDLA

**A comparison of the native NVDLA compiler and ONNC on three CNNs, and what compiler scheduling means for execution-time prediction.**

University of Manchester · Dissertation, 2024–25 · Samarth Sharma

---

## Materials

| | |
|:--|:--|
| **Dissertation** | [`dissertation.pdf`](./dissertation.pdf) |
| **Presentation slides** | [`slides.pptx`](./slides.pptx) |
| **Presentation video** | [`presentation-video.mp4`](./presentation-video.mp4) |

---

## The question

Accelerators such as NVIDIA's open-source Deep Learning Accelerator (NVDLA) are often described as dataflow architectures, which suggests execution should be predictable. Performance models usually build on that assumption and derive timing from the hardware alone.

This project asks whether that holds once the **compiler** is taken into account. The compiler decides operator scheduling, fusion and memory allocation, so two compilers can turn the same network into very different execution plans.

## What I did

I compiled three networks with two toolchains and compared the resulting `.nvdla` loadable files.

| | Native NVDLA compiler | ONNC |
|:--|:--|:--|
| **Input** | Caffe (`.prototxt` + `.caffemodel`) | ONNX, exported from PyTorch (opset 11) |
| **Output** | `.nvdla` loadable | `.nvdla` loadable |
| **Behaviour** | Layer-by-layer, one-to-one with the model graph | Operator fusion, buffer reuse, reordering |

**Models:** LeNet-5 (MNIST), AlexNet and SqueezeNet 1.0 (ImageNet), chosen to cover increasing complexity.

**Method:** The compiler was the only variable. Weights, preprocessing and datasets were held constant, and batch size was fixed at one. I parsed each loadable to extract the scheduling order, peak memory allocation and instruction count.

**Environment:** NVDLA open-source simulator (small configuration), Ubuntu 20.04, PyTorch 2.x, ONNX 1.14, ONNC.

## Results

ONNC produced smaller, leaner loadables than the native compiler on every model, and the gap grew with model complexity.

| Model | Peak memory (native → ONNC) | Reduction | Instructions (native → ONNC) | Change |
|:--|:-:|:-:|:-:|:-:|
| LeNet-5 | 3.2 MB → 2.8 MB | ~12% | 112 → 96 | −14% |
| AlexNet | 48 MB → 41 MB | ~15% | 1,245 → 1,030 | −17% |
| SqueezeNet | 36 MB → 28 MB | ~22% | 1,620 → 1,180 | −27% |

**Key findings**

- **Fusion drives the gains.** ONNC consistently fused convolution, bias and activation layers. The effect was small for LeNet-5 and large for SqueezeNet's fire modules.
- **Predictability falls as optimisation rises.** The native compiler's schedule maps directly to the model's layers. ONNC's reordering and fusion break that mapping, which makes per-layer timing models less reliable.
- **Correctness is preserved.** Accuracy was unaffected. Intermediate floating-point differences were on the order of 1e-6 to 1e-7 and never changed the final rankings in the tested models.
- **The compiler belongs in the model.** Execution-time prediction for accelerators needs both a hardware model and a compiler scheduling model.

## Limitations

- Results come from NVDLA's open-source **simulator**, not silicon, so real-hardware effects are not captured.
- Only **three models** were studied, and ONNC did not support every ONNX operator, which limited model choice.
- The loadable format is sparsely documented, so scheduling had to be reconstructed by manual parsing.

## Future work

Extending the study to other compilers (such as TVM) and accelerators, validating predictions on physical hardware, studying quantised models, and building tools that expose compiler decisions for system-level modelling.

## Reproducing the compilation step

```bash
# Native compiler (Caffe model in, .nvdla loadable out)
./nvdla_compiler --prototxt lenet.prototxt --caffemodel lenet.caffemodel --output lenet.nvdla

# ONNC (ONNX model in, .nvdla loadable out)
onnc -m onnx lenet.onnx -o lenet_onnc.nvdla
```

PyTorch export used fixed input dimensions, a static batch size and opset 11, which ONNC supports reliably. Full details are in Appendices A–F of the dissertation.

## Tech

`Python` · `PyTorch` · `ONNX` · `ONNC` · `Caffe` · `NVDLA`

## Notes

This is academic work submitted to the University of Manchester. Some figures and tools referenced in the dissertation belong to third parties; see the copyright statement inside the PDF.
