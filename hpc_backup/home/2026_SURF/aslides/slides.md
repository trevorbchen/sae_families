# Discovering Interpretable Visual Features in High-Fidelity Generative Models

This is a cleaned narrative outline, not final slide copy. Each slide has one
main claim. Keep text sparse in the actual deck and let the figures carry the
technical detail.

## Slide 1 - Title

**Discovering Interpretable Visual Features in High-Fidelity Generative Models**

Trevor Chen  
Mentors: Pietro Perona and Ilona Demler

**Visual:** A clean row of controlled face edits discovered by an SAE.

## Slide 2 - Roadmap

1. Introduction
3. Our method
4. Experiments
5. Future work

## Slide 3 - Introduction

Section title only: **From generation to internal features**

## Slide 4 - What Does a Generative Model Learn?

- A generative model learns a distribution $p_\theta(x)$ that approximates the
  data distribution.
- Autoregressive models factor the distribution into next-token or next-pixel
  conditionals: $p(x)=\prod_i p(x_i\mid x_{<i})$.
- Diffusion and flow models transform a simple noise distribution into the data
  distribution through a sequence of learned updates.
- The final output is an image, but the model must internally compute reusable
  visual structure such as color, texture, edges, shape, and object identity.

**Visual:** Noise-to-image trajectory beside an autoregressive generation.

**Takeaway:** Image generation is supported by intermediate visual
representations, not a literal database lookup.

## Slide 5 - Generation Is More Than an Input-Output Map

- Mathematically, a diffusion sampler maps noise and conditioning to an image.
- But this is not a simple one-to-one correspondence between input coordinates
  and semantic properties such as hair, edges, lighting, or identity.
- The same visual concept can emerge from many noise samples, and its computation
  is distributed across layers and denoising steps.
- Intermediate representations detect and refine useful structure while the
  image progresses from noise to coarse shape to fine detail.

**Visual:** Pair a denoising trajectory with internal feature examples for face
boundaries, hair, lighting, and texture. Use arrows to show these features being
refined across the trajectory rather than copied directly from the input noise.

**Takeaway:** To understand generation, we need to inspect the model's internal
features, not only the starting noise and final image.

## Slide 6 - Neural Networks Build Internal Features

- Inputs and hidden states are represented as vectors.
- Each layer applies learned linear and nonlinear transformations.
- Directions in a hidden representation can encode useful properties: an edge,
  hair color, lighting, pose, or a more abstract concept.
- We want to identify these directions and test what they causally control.

**Visual:** Image -> encoder/layers -> hidden vector -> output, with a few
candidate feature directions highlighted.

## Slide 7 - The Capacity Puzzle

- Modern networks represent far more useful concepts than they have hidden
  dimensions.
- A width-$d$ layer cannot assign one orthogonal axis to every possible visual
  feature.
- Even a conservative accounting of classes, parts, textures, colors, and
  contexts quickly exceeds the layer width.

**Visual:** A network layer with a few thousand dimensions facing a much larger
set of possible concepts. AlexNet can be used as historical context, but the
point should be architectural rather than model-specific.

## Slide 8 - Superposition

- The superposition hypothesis says models reuse dimensions by representing
  sparse features along many non-orthogonal directions.
- High-dimensional spaces can contain very many approximately orthogonal
  directions, so the number of represented features can exceed the width.
- This works when only a small subset of features is active for any one input.

**Visual:** Standard superposition diagram: orthogonal basis vectors versus many
near-orthogonal feature directions packed into the same space.

**Do not claim a single exact capacity formula.** The capacity depends on the
allowed interference, sparsity, and geometry; the important point is that it can
grow rapidly with dimension.

## Slide 9 - Why Superposition Makes Interpretation Hard

- A neuron is not necessarily a feature.
- One neuron may participate in several concepts, and one concept may be spread
  across many neurons.
- Moving along a raw coordinate can therefore change several visual properties
  at once.

**Question on slide:** Can we recover a larger, sparse feature dictionary from
the model's smaller hidden representation?

## Slide 10 - Sparse Autoencoders

- An SAE maps a hidden state $h\in\mathbb{R}^d$ into an overcomplete feature
  vector $z\in\mathbb{R}^n$, where $n\gg d$.
- A TopK SAE permits only $k$ active features for each input.
- The decoder reconstructs the original hidden state from those sparse features.

\[
z=\operatorname{TopK}(E(h)),\qquad \hat h=D(z)
\]

**Visual:** $d$-dimensional activation -> large sparse dictionary ->
reconstructed activation. Explicitly show that only $k$ entries are active.

## Slide 11 - What SAEs Give Us

- **Detection:** find examples that strongly activate a feature.
- **Description:** inspect what those examples have in common.
- **Causal editing:** change one feature coefficient and decode the result.
- **Auditing:** search for failure modes, biases, or unwanted internal concepts.

**Visual:** For one feature, show activating examples and a negative-to-positive
intervention sweep.

**Caveat:** A visually correlated activation is not automatically causal; edits
must be tested through the frozen decoder or generator.

## Slide 12 - The Missing Degree of Freedom

- A standard SAE is trained to minimize ordinary reconstruction error.
- That objective defines which reconstruction mistakes are expensive.
- Different finite sparse dictionaries can reconstruct equally well while
  exposing different features.
- Therefore an SAE vocabulary is not unique, and ordinary MSE may reveal only
  one useful decomposition of the same representation.

**Visual:** One latent cloud reconstructed by two different sparse dictionaries.

## Slide 13 - Our Research Question

Large text:

> Can we deliberately change the SAE's reconstruction geometry to discover
> different, still useful visual vocabularies?

Small supporting line:

> Instead of accepting whichever features MSE prefers, can we ask the SAE to
> care about errors in edges, coarse structure, or selected frequency bands?

## Slide 14 - Our Work

Section title only: **A family of objective-shaped sparse autoencoders**

## Slide 15 - Experimental Substrate

- Freeze a face autoencoder and its high-fidelity decoder.
- Encode FFHQ/CelebA images into a fixed latent representation.
- Train matched TopK SAEs on the same latent data.
- Change only the SAE objective; keep the latent source, data split, dictionary
  budget, active-feature budget, and evaluation protocol matched.
- Interpret a feature by changing its SAE coefficient and decoding the edited
  latent through the same frozen image decoder.

**Visual:** Image -> frozen encoder -> latent -> SAE -> edited latent -> frozen
decoder -> image.

**Takeaway:** Any vocabulary change comes from SAE training, not retraining the
underlying image model.

## Slide 16 - From One Objective to a Family

Let $z_x$ be the frozen image latent, $\hat z_x$ its SAE reconstruction,
and $G$ the frozen image decoder.

Ordinary SAE:

\[
\mathcal{L}_{\mathrm{MSE}}=\lVert z_x-\hat z_x\rVert_2^2
\]

Objective-shaped SAE:

\[
\mathcal{L}_{\mathrm{proper}}
=\mathcal{L}_{\mathrm{MSE}}
+\lambda\,\lVert T(G(z_x)-G(\hat z_x))\rVert_2^2
\]

We use transforms $T$ that emphasize:

- Laplacian/edge structure
- Low Fourier frequencies
- Mid Fourier frequencies
- High Fourier frequencies

**Visual:** The same image error viewed in pixel, Laplacian, and Fourier-band
spaces.

## Slide 17 - Why Keep the MSE Anchor?

- A pure transformed objective may ignore reconstruction errors in the
  transform's nullspace.
- This happened experimentally: pure transformed SAEs reduced their target loss
  but produced substantially worse latent explained variance.
- Adding MSE keeps the full reconstruction problem constrained while the
  transformed term changes which errors are prioritized.

**Visual:** Proper-versus-pure EV graph:
`diffusion_autoencoder/results/family_sae/grouped_objective_sae/standalone_objectives/convae_4096_k32/figures/proper_ev_comparison.png`

**Takeaway:** The transformed term shapes the vocabulary; MSE prevents the SAE
from exploiting blind spots in the transform.

## Slide 18 - What We Can Prove

In the idealized linear case, the transformed objective is a new metric on the
reconstruction residual:

\[
\mathcal{L}(r)=r^T(I+\lambda T^\top T)r
\]

Machine-checked Lean results establish:

1. Identity weighting recovers ordinary MSE.
2. A non-isotropic metric can reverse which finite-capacity dictionary is
   preferred.
3. Proper transformed loss and MSE have the same zero-loss solutions when exact
   reconstruction is realizable.
4. Pure transformed loss has that guarantee only when the transform is
   injective.

**Important wording:** This proves that changed geometry *can* select a
different finite vocabulary without sacrificing zero-loss expressivity. It does
not prove that SGD always finds that vocabulary or that it is more interpretable.

## Slide 19 - Did the Vocabulary Actually Change?

Train three independent seeds per objective and compare dictionaries using four
complementary measurements:

- **Decoder cosine:** do matched features write in the same latent direction?
- **Activation correlation:** do matched features activate on the same images?
- **Linear CKA:** do the full feature spaces encode similar relationships?
- **Shared-atom/Hungarian matching:** how much of one dictionary has a close
  counterpart in another?

Controls separate:

- same objective, different seed
- different objective, matched seed
- different objective, different seed

**Visual:** A compact diagram explaining the three comparison categories before
showing the heatmaps.

## Slide 20 - Result: Objective and Seed Both Matter

- Changed objectives produce measurably different learned dictionaries.
- Random seed also changes the exact atoms, so individual feature IDs are not
  stable scientific objects.
- The strongest claim comes from distributions, subspaces, and matched controls,
  not from one visually appealing feature.

**Visual:** Use the clearest two vocabulary-alignment plots in the main deck.
Move the remaining decoder/activation/CKA/shared-atom plots to backup unless
there is enough presentation time.

**Takeaway:** We are changing the vocabulary, but vocabulary identity is
statistical rather than one-to-one.

## Slide 21 - Qualitative Features: Semantics and Geometry

Show a small set of human-marked, repeatable interventions such as:

- face width or length
- eye size or eye shadow
- hair density, color, or coarse hair mass
- teeth, ears, hats, or clothing color
- head turn or tilt

For each feature, show several identities and the same coefficient sweep.

**Rule:** Do not present a feature based on one face. The effect must recur while
preserving unrelated identity details.

## Slide 22 - Qualitative Features: Scale and Frequency

Show the cleaner objective-relevant examples:

- global illumination or color cast
- broad face shading versus small localized shadows
- subject edges versus background edges
- coarse hair mass versus individual strands
- structured noise or channel-specific fine detail
- selective sharpening of eyes, teeth, hair, or face boundaries

Avoid calling generic perturbation blur a discovered feature. Slight blur is a
common intervention side effect and is not informative by itself.

**Visual:** Group examples by spatial scale: global, regional, small-local, and
fine detail.

## Slide 23 - One Model, Multiple Vocabularies

Scientific goal:

> Preserve an ordinary semantic vocabulary while allocating additional sparse
> capacity to Laplacian and Fourier geometries.

Engineering problems encountered:

- Shared TopK causes objective groups to compete for activation slots.
- Unrouted gradients let one objective update every dictionary group.
- Differently scaled losses create unstable or dominating gradients.
- Shared partial reconstructions let branches compensate for one another.

**Visual:** Failed shared-write design with arrows showing cross-objective
interference.

## Slide 24 - Stable Solution: Independently Routed Branches

- Five complete branches see the same latent and minibatch.
- One MSE branch: 8,192 features, $k=32$.
- Four specialist branches: 2,048 features and $k=8$ each.
- Each branch owns its parameters and receives only its assigned objective.
- Branches train concurrently; the deployed reconstruction is their uniform
  mean.

Total: 16,384 dictionary features and $k=64$.

**Visual:** One input feeding five parallel SAE branches, then a mean operation.
Make parameter-local gradient routing explicit.

## Slide 25 - Experiments

Section title only: **Are the new vocabularies reconstructive, detectable, and
useful?**

## Slide 26 - Reconstruction and Specialization

- Canonical all-four model: final validation EV **48.87%**
  (peak **48.93%**).
- Architecture-matched five-branch MSE control: **50.08%** EV.
- Cost of specialization: **1.21 EV percentage points**.
- The transformed branches reduced their own target losses relative to matched
  MSE branches by **7.68% to 16.58%**.
- 16,241 of 16,384 features were used during training.

**Visual:**
`diffusion_autoencoder/results/family_sae/grouped_objective_sae/independent_routed_tandem/figures/non_residual_all_four_union_ev.png`

**Takeaway:** Multiple objective-specific vocabularies can coexist with only a
small reconstruction tradeoff.

## Slide 27 - Planted-Feature Recovery

Question: if a known visual feature is injected into only 10% of examples, does
the corresponding objective make it easier for a small SAE feature set to
recover?

- Plant 20 controlled effects: brightness, hue, contrast, Fourier sinusoids,
  local patterns, occlusion, JPEG artifacts, and others.
- Measure how much planted residual variance is explained by the best 1, 5, or
  10 SAE features.
- Proper changed objectives improved top-1 recovery in **11/20** conditions and
  top-10 recovery in **8/20**.
- Gains were selective, not universal. Strong examples include hue rotation and
  mid-frequency sinusoids; several high-frequency conditions became worse.

**Visual:** A compact MSE-versus-best-objective comparison, plus one success and
one failure case.

## Slide 28 - Linear-Probe Benchmark

Question: what information becomes linearly accessible from each sparse
vocabulary?

Use identity-disjoint CelebA splits and matched architecture controls.

Probe families:

- 40 CelebA semantic attributes
- decoded-image geometry: luminance, contrast, gradients, Laplacian power, and
  Fourier-band power
- source-image versions of the same geometry targets
- controlled transformation classification

Also measure how many SAE features are needed to retain 90% or 95% of full-probe
performance.

**Visual:** Experimental design diagram, not the result graph yet.

## Slide 29 - Linear-Probe Results

- The all-four vocabulary preserved most semantic accessibility, but was
  slightly below its matched MSE control: AUROC delta **-0.0052**,
  95% CI **[-0.0095, -0.0009]**.
- Mean decoder-geometry $R^2$ improved by **+0.0297**,
  95% CI **[+0.0075, +0.0565]**.
- The clearest effect was mid-frequency power: decoder $R^2$ delta **+0.1055**
  and source-image $R^2$ delta **+0.1048**, both with intervals above zero.
- Single specialists were inconsistent; not every named objective improved its
  matching downstream probe.

**Visuals:**
- `diffusion_autoencoder/results/family_sae/linear_probe_benchmark/linear_probe_summary.png`
- `diffusion_autoencoder/results/family_sae/linear_probe_benchmark/aligned_geometry_effects.png`

**Takeaway:** Objective shaping changes which information is linearly
accessible, but it is a tradeoff rather than a universal improvement.

## Feature-Edit Output Geometry

Question: do features from each objective-shaped vocabulary causally write the
kind of image geometry emphasized during training?

- Start from the original ConvAE latent and perturb one naturally active SAE
  feature by $\pm2$ active RMS; do not decode the SAE reconstruction.
- Let $y_i=G(h+\Delta z_i d_i)$ and isolate the feature write with
  $\Delta y_i=y_i-G(h)$.
- Normalized Laplacian energy is
  $\|K*I\|_2^2/(\|I\|_2^2+\varepsilon)$.
- Fourier low/mid/high are each measured as their fraction of total non-DC
  spectral power.
- Evaluate both the full edited image $I=y_i$ and causal difference
  $I=\Delta y_i$ across three matched seeds.

The causal-difference result is clearer: Laplacian and Fourier-mid rank first
on their matching measurements; Fourier-high ranks second behind Laplacian;
Fourier-low is effectively inconclusive. The full edited image is dominated by
the source image and shows much weaker alignment.

**Visual:**
`diffusion_autoencoder/results/family_sae/feature_output_geometry/aggregate/causal_delta_geometry_alignment.png`

## Slide 30 - What the Evidence Supports

Supported:

- Reconstruction geometry changes the learned SAE vocabulary.
- Proper MSE-anchored objectives remain reconstructive.
- Multiple objective-specific branches can train together stably.
- The all-four vocabulary improves some frequency-aligned downstream readouts.

Not yet established:

- Every transformed feature is more human-interpretable.
- Every specialist improves its named task.
- Individual atoms reproduce exactly across seeds.
- These results automatically transfer to language or arbitrary model families.

This slide is important: it makes the conclusion credible rather than broad.

## Slide 31 - Future Work

- Finish systematic labels for the top activity-ranked features in each family.
- Quantify which effects are enriched by objective: global lighting, local
  shadows, edge support, texture scale, structured noise, and color detail.
- Validate labels with held-out identities and causal interventions.
- Search for objective-specific failure-mode features that an ordinary SAE
  misses.
- Extend the framework to other representations and modalities, including
  language-model activations.
- Study learned or task-derived transforms instead of only hand-selected
  Laplacian/Fourier operators.

## Slide 32 - Conclusion

Large text:

> Sparse autoencoders do not expose a unique vocabulary.

Then reveal:

> By changing reconstruction geometry, we can train a family of sparse
> vocabularies that preserve most reconstruction quality while making different
> visual information accessible.

Final line:

> Interpretability is not only about reading the features a model gives us; it
> is also about choosing the geometry in which we ask for them.

## Slide 33 - Acknowledgments

- Pietro Perona
- Ilona Demler
- Robert K. and Alice L. Roney SURF Fellowship
- Caltech Student-Faculty Programs and the computing support used for this work

**Closing:** Questions?

## Slide 34 - References

Keep this readable and move detailed citations to speaker notes if necessary.

- Elhage et al. (2022), *Toy Models of Superposition*
- Bricken et al. (2023), *Towards Monosemanticity*
- Gao et al. (2024), *Scaling and Evaluating Sparse Autoencoders*
- Preechakul et al. (2022), *Diffusion Autoencoders: Toward a Meaningful and
  Decodable Representation*
- *Vision Sparse Autoencoders: A Benchmark Study* (2025)
- *Are Sparse Autoencoder Benchmarks Reliable?* (2026)

# Suggested Backup Slides

1. Full decoder-cosine, activation-correlation, CKA, and shared-atom matrices
2. Proper-versus-pure loss curves for each objective
3. Full 20-condition planted-feature table
4. All 40 CelebA attribute probes
5. Feature-efficiency curves and effective support
6. Failed shared-routing architecture and gradient-instability diagnostics
7. Lean theorem statements and assumptions
