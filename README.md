# Awesome Image & Video Deblurring

A curated, structured collection of research papers, implementations, datasets, and benchmarks for image and video deblurring.

**Coverage:** 2019–2026 · **Papers:** 154 · **Datasets:** 18

> This file is generated from the structured research data. Edit the YAML catalog, not generated tables.

## Browse

- [By task](docs/by-task.md) — motion, defocus, blind, video, blur synthesis, 3D reconstruction, and related problems.
- [By signal / sensor](docs/by-signal.md) — events, gyro, dual/quad pixel, spike, stereo, RAW, and more.
- [By method](docs/by-method.md) — diffusion, transformers, state-space models, kernel estimation, Gaussian splatting, and more.
- [Datasets & benchmarks](#datasets--benchmarks)

### By year

[2026](#2026-papers) (15) | [2025](#2025-papers) (24) | [2024](#2024-papers) (27) | [2023](#2023-papers) (30) | [2022](#2022-papers) (22) | [2021](#2021-papers) (10) | [2020](#2020-papers) (21) | [2019](#2019-papers) (5)

### Research map

| Dimension | Most represented categories | Full index |
|---|---|---|
| Tasks | Motion Deblurring (103), Video Deblurring (24), Blind Deblurring (24), Defocus Deblurring (18), 3D Reconstruction (9), Deconvolution (8) | [Browse tasks](docs/by-task.md) |
| Signals | RGB (120), Events (26), Video (25), Gyro (3), Spike (3), Dual Pixel (3) | [Browse signals](docs/by-signal.md) |
| Methods | Kernel Estimation (21), Frequency Domain (13), CNN (13), Degradation Modeling (12), Diffusion (11), Transformer (10) | [Browse methods](docs/by-method.md) |

## 2026 Papers

| Venue | Paper | Task | Resource |
|---|---|---|---|
| CVPR | [Gyro-based Deep Video Deblurring](https://cg.postech.ac.kr/researches/GyroDVD/) | Motion · Video | [Code](https://github.com/rimchang/GyroDVD) |
| CVPR | [Time-Specialized Event-Image Alignment for Blur-to-Video Decomposition](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Time-Specialized_Event-Image_Alignment_for_Blur-to-Video_Decomposition_CVPR_2026_paper.html) | Blur-to-Video | — |
| CVPR | [Spatio-Temporal Difference Guided Motion Deblurring with the Complementary Vision Sensor](https://openaccess.thecvf.com/content/CVPR2026/html/Meng_Spatio-Temporal_Difference_Guided_Motion_Deblurring_with_the_Complementary_Vision_Sensor_CVPR_2026_paper.html) | Motion | [Code](https://github.com/Tianmouc/tmcDeblur) |
| CVPR | [Event-based Motion Deblurring with Unpaired Data](https://openaccess.thecvf.com/content/CVPR2026/html/Cho_Event-based_Motion_Deblurring_with_Unpaired_Data_CVPR_2026_paper.html) | Motion | [Code](https://github.com/Chohoonhee/EMP) |
| CVPR | [Event-Based Motion Deblurring Using Task-Oriented 3D Gaussian Event Representations](https://openaccess.thecvf.com/content/CVPR2026/html/Xue_Event-Based_Motion_Deblurring_Using_Task-Oriented_3D_Gaussian_Event_Representations_CVPR_2026_paper.html) | Motion | — |
| CVPR | [Seeing Through Blur: Tackling Defocus in Spike-Based Imaging](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Seeing_Through_Blur_Tackling_Defocus_in_Spike-Based_Imaging_CVPR_2026_paper.html) | Defocus | — |
| CVPR | [MSCD-GS: Motion-Separated Cooperative Deblurring Dynamic Reconstruction via Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2026/html/Liao_MSCD-GS_Motion-Separated_Cooperative_Deblurring_Dynamic_Reconstruction_via_Gaussian_Splatting_CVPR_2026_paper.html) | Motion · 3D Reconstruction | — |
| CVPR Findings | [MVSSM: Motion-aware Visual State Space Model for Efficient Video Deblurring](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhou_MVSSM_Motion-aware_Visual_State_Space_Model_for_Efficient_Video_Deblurring_CVPRF_2026_paper.html) | Motion · Video | — |
| ECCV | [CogSENet: Blind Image Deblurring with Blur-Conditioned Semantic Routing and Explicit Frequency Fusion](https://arxiv.org/abs/2606.30030) | Blind · Motion | — |
| ECCV | [Event-Driven Motion Deblurring via Trajectory-Based Kernel Reconstruction](https://link.springer.com/book/10.1007/978-3-032-37314-4) | Motion | — |
| ECCV | [Leveraging Phase Information to Boost Unrolled Network Learning for Image Deblurring](https://arxiv.org/abs/2607.00251) | Image | — |
| ECCV | [A Benchmark for Heterogeneous Stereo Deblurring with Physically- and Epipolar-Constrained Cross Attention](https://arxiv.org/abs/2606.25962) | Motion | [Code](https://github.com/shinhoju/PECA) |
| ECCV | [FMA-Net++: Motion- and Exposure-Aware Joint Video Super-Resolution and Deblurring](https://kaist-viclab.github.io/fmanetpp_site/) | Motion · Video · Super-Resolution | [Code](https://github.com/KAIST-VICLab/FMA-Net-PlusPlus) |
| ECCV | [PRISM3D: Probabilistic Refinement and Robust Initialization for Physically Consistent Scene Modeling under Extreme Motion Blur](https://arxiv.org/abs/2607.03855) | Motion · 3D Reconstruction | [Code](https://github.com/GopiRajuMatta/PRISM3D) |
| ECCV | [Realistic Compound-Lens Defocus Blur Synthesis](https://arxiv.org/abs/2607.05837) | Defocus · Blur Synthesis | [Code](https://github.com/lykelee/CLDefocus) |

## 2025 Papers

| Venue | Paper | Task | Resource |
|---|---|---|---|
| WACV | [Blind Image Deblurring with FFT-ReLU Sparsity Prior](https://openaccess.thecvf.com/content/WACV2025/html/Al_Radi_Blind_Image_Deblurring_with_FFT-ReLU_Sparsity_Prior_WACV_2025_paper.html) | Blind | [Code](https://github.com/Metalicana/Blind-Image-Deblurring-using-FFT-ReLU-with-Deep-Learning-Pipeline-Integration) |
| WACV | [Deep Joint Unrolling for Deblurring and Low-Light Image Enhancement (JUDE)](https://openaccess.thecvf.com/content/WACV2025/html/Vo_Deep_Joint_Unrolling_for_Deblurring_and_Low-Light_Image_Enhancement_JUDE_WACV_2025_paper.html) | Motion · Low-light | [Project](https://jude.kc-ml2.com/) |
| WACV Workshops | [DaBiT: Depth and Blur informed Transformer for Video Deblurring](https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/html/Morris_DaBiT_Depth_and_Blur_informed_Transformer_for_Video_Deblurring_WACVW_2025_paper.html) | Video · Defocus | — |
| AAAI | [Motion-adaptive Transformer for Event-based Image Deblurring](https://ojs.aaai.org/index.php/AAAI/article/view/32967) | Motion | — |
| AAAI | [Residual Diffusion Deblurring Model for Single Image Defocus Deblurring](https://ojs.aaai.org/index.php/AAAI/article/view/32303) | Defocus | — |
| AAAI | [Asymmetric Hierarchical Difference-aware Interaction Network for Event-guided Motion Deblurring](https://ojs.aaai.org/index.php/AAAI/article/view/33003) | Motion | — |
| CVPR | [Quad-Pixel Image Defocus Deblurring: A New Benchmark and Model](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Quad-Pixel_Image_Defocus_Deblurring_A_New_Benchmark_and_Model_CVPR_2025_paper.html) | Defocus | [Code](https://github.com/excllent123/QPD-Application) |
| CVPR | [Efficient Visual State Space Model for Image Deblurring](https://openaccess.thecvf.com/content/CVPR2025/html/Kong_Efficient_Visual_State_Space_Model_for_Image_Deblurring_CVPR_2025_paper.html) | Motion | [Code](https://github.com/kkkls/EVSSM) |
| CVPR | [Gyro-based Neural Single Image Deblurring](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Gyro-based_Neural_Single_Image_Deblurring_CVPR_2025_paper.html) | Motion | [Code](https://github.com/hmyang0727/GyroDeblurNet) |
| CVPR | [Diffusion-based Event Generation for High-Quality Image Deblurring](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Diffusion-based_Event_Generation_for_High-Quality_Image_Deblurring_CVPR_2025_paper.html) | Motion | [Code](https://github.com/XinanXie/EGDeblurring) |
| CVPR | [Parameterized Blur Kernel Prior Learning for Local Motion Deblurring](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_Parameterized_Blur_Kernel_Prior_Learning_for_Local_Motion_Deblurring_CVPR_2025_paper.html) | Local Motion · Blind | — |
| CVPR | [A Polarization-Aided Transformer for Image Deblurring via Motion Vector Decomposition](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_A_Polarization-Aided_Transformer_for_Image_Deblurring_via_Motion_Vector_Decomposition_CVPR_2025_paper.html) | Motion | — |
| CVPR | [Exploiting Deblurring Networks for Radiance Fields](https://openaccess.thecvf.com/content/CVPR2025/html/Choi_Exploiting_Deblurring_Networks_for_Radiance_Fields_CVPR_2025_paper.html) | Motion · 3D Reconstruction | [Code](https://github.com/haeyun-choi/DeepDeblurRF) |
| ICCV | [Blind Noisy Image Deblurring Using Residual Guidance Strategy](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Blind_Noisy_Image_Deblurring_Using_Residual_Guidance_Strategy_ICCV_2025_paper.html) | Blind | — |
| ICCV | [Performing Defocus Deblurring by Modeling its Formation Process](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Performing_Defocus_Deblurring_by_Modeling_its_Formation_Process_ICCV_2025_paper.html) | Defocus | — |
| ICCV | [Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Learning_Deblurring_Texture_Prior_from_Unpaired_Data_with_Diffusion_Model_ICCV_2025_paper.html) | Motion | [Code](https://github.com/ChengxuLiu/TP-Diff) |
| ICCV | [ClearSight: Human Vision-Inspired Solutions for Event-Based Motion Deblurring](https://openaccess.thecvf.com/content/ICCV2025/html/Lin_ClearSight_Human_Vision-Inspired_Solutions_for_Event-Based_Motion_Deblurring_ICCV_2025_paper.html) | Motion | — |
| ICCV | [EVDM: Event-based Real-world Video Deblurring with Mamba](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_EVDM_Event-based_Real-world_Video_Deblurring_with_Mamba_ICCV_2025_paper.html) | Motion · Video | [Code](https://github.com/ZhijingS/EVDM) |
| ICCV | [Efficient Concertormer for Image Deblurring and Beyond](https://openaccess.thecvf.com/content/ICCV2025/html/Kuo_Efficient_Concertormer_for_Image_Deblurring_and_Beyond_ICCV_2025_paper.html) | Motion | — |
| ICCV | [Separation for Better Integration: Disentangling Edge and Motion in Event-based Deblurring](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Separation_for_Better_Integration_Disentangling_Edge_and_Motion_in_Event-based_ICCV_2025_paper.html) | Motion | — |
| ICCV | [Splat-based 3D Scene Reconstruction with Extreme Motion-blur](https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Splat-based_3D_Scene_Reconstruction_with_Extreme_Motion-blur_ICCV_2025_paper.html) | Motion · 3D Reconstruction | [Code](https://github.com/KAISTVCLAB/gs-extreme-motion-blur) |
| NeurIPS | [DeblurDiff: Real-World Image Deblurring with Generative Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e393677793767624f2821cec8bdd02f1-Abstract-Conference.html) | Motion | [Code](https://github.com/kkkls/DeblurDiff) |
| NeurIPS | [BlurDM: A Blur Diffusion Model for Image Deblurring](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4b43f14df70be3b93e8c415d46df0598-Abstract-Conference.html) | Motion | [Code](https://github.com/Jin-Ting-He/BlurDM) |
| NeurIPS | [Asymmetric Dual-Lens Video Deblurring](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3c8290b9d484baa0435f31d11e01b5b8-Abstract-Conference.html) | Motion · Video | — |

<a id="2024-papers"></a>

<details>
<summary><strong>2024 Papers (27)</strong></summary>

| Venue | Paper | Task | Resource |
|---|---|---|---|
| arXiv | [Fast Diffusion EM: a diffusion model for blind inverse problems with application to deconvolution](https://arxiv.org/abs/2309.00287v2) | Blind · Deconvolution | [Code](https://github.com/claroche-r/fastdiffusionem) |
| SPIE | [Estimation of motion blur kernel parameters using regression convolutional neural networks](https://arxiv.org/abs/2308.01381v3) | Blind | [Code](https://github.com/duckduckpig/regression_blur) |
| SIGGRAPH | [Deep Hybrid Camera Deblurring for Smartphone Cameras](https://graphics.postech.ac.kr/researches/HCDeblur/) | Motion | [Code](https://github.com/rimchang/HCDeblur) |
| CVPR | [A Unified Framework for Microscopy Defocus Deblur with Multi-Pyramid Transformer and Contrastive Learning](https://arxiv.org/abs/2403.02611) | Defocus | [Code](https://github.com/PieceZhang/MPT-CataBlur) |
| CVPR | [AdaRevD: Adaptive Patch Exiting Reversible Decoder Pushes the Limit of Image Deblurring](https://arxiv.org/abs/2406.09135) | Motion | [Code](https://github.com/INVOKERer/AdaRevD) |
| CVPR | [Blur2Blur: Blur Conversion for Unsupervised Image Deblurring on Unknown Domains](https://arxiv.org/abs/2403.16205) | Motion · Blur Synthesis | [Code](https://github.com/VinAIResearch/Blur2Blur) |
| CVPR | [Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light Enhancement and Deblurring](https://openaccess.thecvf.com/content/CVPR2024/html/Lv_Fourier_Priors-Guided_Diffusion_for_Zero-Shot_Joint_Low-Light_Enhancement_and_Deblurring_CVPR_2024_paper.html) | Motion · Low-light | [Code](https://github.com/aipixel/FourierDiff) |
| CVPR | [ID-Blau: Image Deblurring by Implicit Diffusion-based reBLurring AUgmentation](https://arxiv.org/abs/2312.10998) | Motion · Blur Synthesis | [Code](https://github.com/plusgood-steven/ID-Blau) |
| CVPR | [LDP: Language-driven Dual-Pixel Image Defocus Deblurring Network](https://arxiv.org/abs/2307.09815) | Defocus | [Code](https://github.com/noxsine/LDP) |
| CVPR | [Mitigating Motion Blur in Neural Radiance Fields with Events and Frames](https://arxiv.org/abs/2403.19780) | Motion · 3D Reconstruction | [Code](https://github.com/uzh-rpg/EvDeblurNeRF) |
| CVPR | [Motion-adaptive Separable Collaborative Filters for Blind Motion Deblurring](https://arxiv.org/abs/2404.13153) | Blind · Motion | [Code](https://github.com/ChengxuLiu/MISCFilter) |
| CVPR | [Motion Blur Decomposition with Cross-shutter Guidance](https://arxiv.org/abs/2404.01120) | Blur-to-Video | [Code](https://github.com/jixiang2016/dualBR) |
| CVPR | [Spike-guided Motion Deblurring with Unknown Modal Spatiotemporal Alignment](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Spike-guided_Motion_Deblurring_with_Unknown_Modal_Spatiotemporal_Alignment_CVPR_2024_paper.html) | Motion | [Code](https://github.com/Leozhangjiyuan/UaSDN) |
| CVPR | [Blur-aware Spatio-temporal Sparse Transformer for Video Deblurring](https://arxiv.org/abs/2406.07551) | Motion · Video | [Code](https://github.com/huicongzhang/BSSTNet) |
| CVPR | [Unsupervised Blind Image Deblurring Based on Self-Enhancement](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Unsupervised_Blind_Image_Deblurring_Based_on_Self-Enhancement_CVPR_2024_paper.html) | Blind | — |
| CVPR | [Real-World Efficient Blind Motion Deblurring via Blur Pixel Discretization](https://arxiv.org/abs/2404.12168) | Blind · Motion | — |
| CVPR | [EVS-assisted Joint Deblurring Rolling-Shutter Correction and Video Frame Interpolation through Sensor Inverse Modeling](https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_EVS-assisted_Joint_Deblurring_Rolling-Shutter_Correction_and_Video_Frame_Interpolation_through_CVPR_2024_paper.pdf) | Motion · Rolling Shutter · Frame Interpolation | — |
| CVPR | [Latency Correction for Event-guided Deblurring and Frame Interpolation](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Latency_Correction_for_Event-guided_Deblurring_and_Frame_Interpolation_CVPR_2024_paper.pdf) | Motion · Frame Interpolation | — |
| CVPR | [Frequency-aware Event-based Video Deblurring for Real-World Motion Blur](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Frequency-aware_Event-based_Video_Deblurring_for_Real-World_Motion_Blur_CVPR_2024_paper.html) | Motion · Video | — |
| arXiv | [Gyroscope-Assisted Motion Deblurring Network](https://arxiv.org/abs/2402.06854) | Motion | — |
| ECCV | [BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting](https://arxiv.org/abs/2403.11831) | Motion · 3D Reconstruction | [Code](https://github.com/WU-CVGL/BAD-Gaussians) |
| ECCV | [BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](https://arxiv.org/abs/2407.02174v2) | Motion · 3D Reconstruction | [Code](https://github.com/WU-CVGL/BeNeRF) |
| ECCV | [Blind Image Deblurring with Noise-Robust Kernel Estimation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3024_ECCV_2024_paper.php) | Blind | [Code](https://github.com/csleemooo/BD_noise_robust_kernel_estimation) |
| ECCV | [Domain-adaptive Video Deblurring via Test-time Blurring](https://arxiv.org/abs/2407.09059) | Motion · Video · Blur Synthesis | [Code](https://github.com/Jin-Ting-He/DADeblur) |
| ECCV | [Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion](https://arxiv.org/abs/2403.13327) | Motion · Rolling Shutter · 3D Reconstruction | [Code](https://github.com/SpectacularAI/3dgs-deblur) |
| ECCV | [Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](http://vi.kaist.ac.kr/2024/07/02/towards-real-world-event-guided-low-light-video-enhancement-and-deblurring/) | Motion · Video · Low-light | [Code](https://github.com/intelpro/ELEDNet) |
| ECCV | [UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation](https://arxiv.org/abs/2305.15078) | Motion · Rolling Shutter · Frame Interpolation | [Code](https://github.com/yunfanLu/UniINR) |

</details>

<a id="2023-papers"></a>

<details>
<summary><strong>2023 Papers (30)</strong></summary>

| Venue | Paper | Task | Resource |
|---|---|---|---|
| ICML | [GibbsDDRM: A Partially Collapsed Gibbs Sampler for Solving Blind Inverse Problems with Denoising Diffusion Restoration](https://arxiv.org/abs/2301.12686v2) | Blind · Deconvolution | [Code](https://github.com/sony/gibbsddrm) |
| IJCV | [Blind Image Deblurring with Unknown Kernel Size and Substantial Noise](https://arxiv.org/abs/2208.09483v2) | Blind | [Code](https://github.com/sun-umn/Blind-Image-Deblurring) |
| TIP | [INFWIDE: Image and Feature Space Wiener Deconvolution Network for Non-blind Image Deblurring in Low-Light Conditions](https://ieeexplore.ieee.org/document/10047966) | Non-blind · Low-light · Deconvolution | [Code](https://github.com/zhihongz/infwide) |
| AAAI | [Real-World Deep Local Motion Deblurring](https://arxiv.org/abs/2204.08179) | Local Motion | [Code](https://github.com/LeiaLi/ReLoBlur) |
| ICCV | [Multi-scale Residual Low-Pass Filter Network for Image Deblurring](https://ieeexplore.ieee.org/document/10377577) | Motion | — |
| TCSVT | [Multi-Scale Frequency Separation Network for Image Deblurring](https://arxiv.org/abs/2206.00798) | Motion | [Code](https://github.com/LiQiang0307/MSFS-Net) |
| ICML | [IRNeXt: Rethinking Convolutional Network Design for Image Restoration](https://dl.acm.org/doi/10.5555/3618408.3618669) | Image | [Code](https://github.com/c-yn/IRNeXt) |
| CVPR | [Structured Kernel Estimation for Photon-Limited Deconvolution](https://arxiv.org/abs/2303.03472) | Blind · Deconvolution | [Code](https://github.com/sanghviyashiitb/structured-kernel-cvpr23) |
| CVPR | [Blur Interpolation Transformer for Real-World Motion from Blur](https://arxiv.org/abs/2211.11423) | Blur-to-Video | [Code](https://github.com/zzh-tech/BiT) |
| CVPR | [Neumann Network with Recursive Kernels for Single Image Defocus Deblurring](https://openaccess.thecvf.com/content/CVPR2023/papers/Quan_Neumann_Network_With_Recursive_Kernels_for_Single_Image_Defocus_Deblurring_CVPR_2023_paper.pdf) | Defocus | [Code](https://github.com/csZcWu/NRKNet) |
| CVPR | [Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring](https://arxiv.org/abs/2211.12250) | Motion | [Code](https://github.com/kkkls/FFTformer) |
| CVPR | [Hybrid Neural Rendering for Large-Scale Scenes with Motion Blur](https://arxiv.org/abs/2304.12652) | Motion · 3D Reconstruction | [Code](https://github.com/CVMI-Lab/HybridNeuralRendering) |
| CVPR | [Self-Supervised Non-Uniform Kernel Estimation With Flow-Based Motion Prior for Blind Image Deblurring](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_Self-Supervised_Non-Uniform_Kernel_Estimation_With_Flow-Based_Motion_Prior_for_Blind_CVPR_2023_paper.html) | Blind · Motion | [Code](https://github.com/Fangzhenxuan/UFPDeblur) |
| CVPR | [Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior](https://arxiv.org/abs/2210.05361) | Motion | [Code](https://github.com/xl-tang01/UAUDeblur) |
| CVPR | [K3DN: Disparity-Aware Kernel Estimation for Dual-Pixel Defocus Deblurring](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_K3DN_Disparity-Aware_Kernel_Estimation_for_Dual-Pixel_Defocus_Deblurring_CVPR_2023_paper.html) | Defocus | — |
| CVPR | [Self-Supervised Blind Motion Deblurring With Deep Expectation Maximization](https://ieeexplore.ieee.org/document/10203880) | Blind · Motion | — |
| CVPR | [HyperCUT: Video Sequence from a Single Blurry Image using Unsupervised Ordering](https://arxiv.org/abs/2304.01686) | Blur-to-Video | [Code](https://github.com/VinAIResearch/HyperCUT) |
| CVPR | [Deep Discriminative Spatial and Temporal Network for Efficient Video Deblurring](https://ieeexplore.ieee.org/document/10204041) | Motion · Video | [Code](https://github.com/xuboming8/DSTNet) |
| AAAI | [Dual-Domain Attention for Image Deblurring](https://ojs.aaai.org/index.php/AAAI/article/view/25122) | Motion | [Code](https://github.com/c-yn/DDANet) |
| AAAI | [Learning Single Image Defocus Deblurring with Misaligned Training Pairs](https://ojs.aaai.org/index.php/AAAI/article/view/25235) | Defocus | [Code](https://github.com/liyucs/JDRL) |
| AAAI | [Intriguing Findings of Frequency Selection for Image Deblurring](https://arxiv.org/abs/2111.11745) | Motion | [Code](https://github.com/INVOKERer/DeepRFT/tree/AAAI2023) |
| AAAI | [Learnable Blur Kernel for Single-Image Defocus Deblurring in the Wild](https://ojs.aaai.org/index.php/AAAI/article/view/25446) | Defocus | — |
| ICCV | [Multiscale Structure Guided Diffusion for Image Deblurring](https://arxiv.org/abs/2212.01789) | Motion | — |
| ICCV | [Single Image Defocus Deblurring via Implicit Neural Inverse Kernels](https://openaccess.thecvf.com/content/ICCV2023/papers/Quan_Single_Image_Defocus_Deblurring_via_Implicit_Neural_Inverse_Kernels_ICCV_2023_paper.pdf) | Defocus | [Code](https://github.com/xinyao240/INIKNet) |
| ICCV | [Single Image Deblurring with Row-dependent Blur Magnitude](https://openaccess.thecvf.com/content/ICCV2023/html/Ji_Single_Image_Deblurring_with_Row-dependent_Blur_Magnitude_ICCV_2023_paper.html) | Motion | [Code](https://github.com/jixiang2016/RSS-T) |
| ICCV | [Non-Coaxial Event-Guided Motion Deblurring with Spatial Alignment](https://openaccess.thecvf.com/content/ICCV2023/html/Cho_Non-Coaxial_Event-Guided_Motion_Deblurring_with_Spatial_Alignment_ICCV_2023_paper.html) | Motion | — |
| ICCV | [Generalizing Event-Based Motion Deblurring in Real-World Scenarios](https://arxiv.org/abs/2308.05932) | Motion | [Code](https://github.com/XiangZ-0/GEM) |
| ICCV | [Exploring Temporal Frequency Spectrum in Deep Video Deblurring](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Exploring_Temporal_Frequency_Spectrum_in_Deep_Video_Deblurring_ICCV_2023_paper.pdf) | Motion · Video | — |
| NeurIPS | [Hierarchical Integration Diffusion Model for Realistic Image Deblurring](https://arxiv.org/abs/2305.12966) | Motion | [Code](https://github.com/zhengchen1999/HI-Diff) |
| NeurIPS | [Enhancing Motion Deblurring in High-Speed Scenes with Spike Streams](https://openreview.net/forum?id=cAyLnMxiTl) | Motion | — |

</details>

<a id="2022-papers"></a>

<details>
<summary><strong>2022 Papers (22)</strong></summary>

| Venue | Paper | Task | Resource |
|---|---|---|---|
| ECCVW | [MSSNet: Multi-Scale-Stage Network for Single Image Deblurring](https://arxiv.org/abs/2202.09652) | Motion | [Code](https://github.com/kky7/MSSNet) |
| CVPRW | [HINet: Half Instance Normalization Network for Image Restoration](https://arxiv.org/abs/2105.06086) | Image | [Code](https://github.com/megvii-model/HINet) |
| TIP | [BANet: A Blur-Aware Attention Network for Dynamic Scene Deblurring](https://arxiv.org/abs/2101.07518) | Motion | [Code](https://github.com/pp00704831/BANet-TIP-2022) |
| CVPR | [Learning to Deblur using Light Field Generated and Real Defocus Images](https://arxiv.org/abs/2204.00367) | Defocus | [Code](https://github.com/lingyanruan/DRBNet) |
| CVPR | [Pixel Screening Based Intermediate Correction for Blind Deblurring](https://ieeexplore.ieee.org/document/9878959) | Blind | — |
| CVPR | [Deblurring via Stochastic Refinement](https://openaccess.thecvf.com/content/CVPR2022/html/Whang_Deblurring_via_Stochastic_Refinement_CVPR_2022_paper.html) | Motion | — |
| CVPR | [XYDeblur: Divide and Conquer for Single Image Deblurring](https://ieeexplore.ieee.org/document/9880408) | Motion | — |
| CVPR | [Unifying Motion Deblurring and Frame Interpolation with Events](https://arxiv.org/abs/2203.12178) | Motion · Frame Interpolation | [Code](https://github.com/XiangZ-0/EVDI) |
| CVPR | [E-CIR: Event-Enhanced Continuous Intensity Recovery](https://arxiv.org/abs/2203.01935) | Motion | [Code](https://github.com/chensong1995/E-CIR) |
| CVPR | [Multi-Scale Memory-Based Video Deblurring](https://arxiv.org/abs/2204.02977) | Motion · Video | [Code](https://github.com/jibo27/MemDeblur) |
| ECCV | [Learning Degradation Representations for Image Deblurring](https://arxiv.org/abs/2208.05244) | Motion | [Code](https://github.com/dasongli1/Learning_degradation) |
| ECCV | [Stripformer: Strip Transformer for Fast Image Deblurring](https://arxiv.org/abs/2204.04627) | Motion | [Code](https://github.com/pp00704831/Stripformer-ECCV-2022-) |
| ECCV | [Animation from Blur: Multi-modal Blur Decomposition with Motion Guidance](https://arxiv.org/abs/2207.10123) | Blur-to-Video | [Code](https://github.com/zzh-tech/Animation-from-Blur) |
| ECCV | [United Defocus Blur Detection and Deblurring via Adversarial Promoting Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3308_ECCV_2022_paper.php) | Defocus | [Code](https://github.com/wdzhao123/APL) |
| ECCV | [Realistic Blur Synthesis for Learning Image Deblurring](https://arxiv.org/abs/2202.08771) | Motion · Blur Synthesis | [Code](https://github.com/rimchang/RSBlur) |
| ECCV | [Event-based Fusion for Motion Deblurring with Cross-modal Attention](https://arxiv.org/abs/2112.00167) | Motion | [Code](https://github.com/AHupuJR/EFNet) |
| ECCV | [Event-Guided Deblurring of Unknown Exposure Time Videos](https://arxiv.org/abs/2112.06988) | Motion · Video | [Code](https://github.com/intelpro/UEVD_public) |
| ECCV | [Spatio-Temporal Deformable Attention Network for Video Deblurring](https://arxiv.org/abs/2207.10852) | Motion · Video | [Code](https://github.com/huicongzhang/STDAN) |
| ECCV | [Efficient Video Deblurring Guided by Motion Magnitude](https://arxiv.org/abs/2207.13374) | Motion · Video | [Code](https://github.com/sollynoay/MMP-RNN) |
| ECCV | [ERDN: Equivalent Receptive Field Deformable Network for Video Deblurring](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4085_ECCV_2022_paper.php) | Motion · Video | [Code](https://github.com/TencentCloud/ERDN) |
| ECCV | [DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting](https://arxiv.org/abs/2111.09985) | Motion · Video · Frame Interpolation | [Code](https://github.com/JihyongOh/DeMFI) |
| ECCV | [Towards Real-World Video Deblurring by Exploring Blur Formation Process](https://arxiv.org/abs/2208.13184) | Motion · Video · Blur Synthesis | [Code](https://github.com/ljzycmd/rawblur) |

</details>

<a id="2021-papers"></a>

<details>
<summary><strong>2021 Papers (10)</strong></summary>

| Venue | Paper | Task | Resource |
|---|---|---|---|
| CVPR | [Explore Image Deblurring via Encoded Blur Kernel Space](https://openaccess.thecvf.com/content/CVPR2021/html/Tran_Explore_Image_Deblurring_via_Encoded_Blur_Kernel_Space_CVPR_2021_paper.html) | Blind · Motion | [Code](https://github.com/VinAIResearch/blur-kernel-space-exploring) |
| CVPR | [Multi-Stage Progressive Image Restoration](https://arxiv.org/abs/2102.02808) | Image | [Code](https://github.com/swz30/MPRNet) |
| CVPR | [DeFMO: Deblurring and Shape Recovery of Fast Moving Objects](https://arxiv.org/abs/2012.00595) | Local Motion | [Code](https://github.com/rozumden/DeFMO) |
| CVPR | [ARVo: Learning All-Range Volumetric Correspondence for Video Deblurring](https://arxiv.org/abs/2103.04260) | Motion · Video | — |
| CVPR | [Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes](https://arxiv.org/abs/2104.01601) | Motion · Rolling Shutter | [Code](https://github.com/zzh-tech/RSCD) |
| CVPR | [Digital Gimbal: End-to-end Deep Image Stabilization with Learnable Exposure Times](https://arxiv.org/abs/2012.04515) | Motion | [Code](https://github.com/omer11a/digital-gimbal) |
| ICCV | [Bringing Events into Video Deblurring with Non-consecutively Blurry Frames](https://ieeexplore.ieee.org/document/9711143) | Motion · Video | [Code](https://github.com/shangwei5/D2Net) |
| ICCV | [Rethinking Coarse-to-Fine Approach in Single Image Deblurring](https://arxiv.org/abs/2108.05054) | Motion | [Code](https://github.com/chosj95/MIMO-UNet) |
| ICCV | [Single Image Defocus Deblurring Using Kernel-Sharing Parallel Atrous Convolutions](https://arxiv.org/abs/2108.09108) | Defocus | [Code](https://github.com/HyeongseokSon1/KPAC) |
| NeurIPS | [Gaussian Kernel Mixture Network for Single Image Defocus Deblurring](https://openreview.net/forum?id=kSR-_SVzDR-) | Defocus | [Code](https://github.com/csZcWu/GKMNet) |

</details>

<a id="2020-papers"></a>

<details>
<summary><strong>2020 Papers (21)</strong></summary>

| Venue | Paper | Task | Resource |
|---|---|---|---|
| NeurIPS | [Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring](https://arxiv.org/abs/2103.09962v1) | Non-blind · Deconvolution | [Code](https://gitlab.mpi-klsb.mpg.de/jdong/dwdn) |
| IEEE | [Raw Image Deblurring](https://arxiv.org/abs/2012.04264v1) | Motion | [Code](https://github.com/bob831009/raw_image_deblurring) |
| TCSVT | [A Simple Local Minimal Intensity Prior and An Improved Algorithm for Blind Image Deblurring](https://arxiv.org/abs/1906.06642v5) | Blind | [Code](https://github.com/FWen/deblur-pmp) |
| ECCV | [End-to-end Interpretable Learning of Non-blind Image Deblurring](https://arxiv.org/abs/2007.01769v2) | Non-blind | [Code](https://github.com/teboli/CPCR) |
| IJCV | [Spatially-Adaptive Filter Units for Compact and Efficient Deep Neural Networks](https://arxiv.org/abs/1902.07474v2) | Image | [Code](https://github.com/skokec/DAU-ConvNet) |
| TNNLS | [Learning Deep Gradient Descent Optimization for Image Deconvolution](https://arxiv.org/abs/1804.03368v2) | Deconvolution · Non-blind | [Code](https://github.com/donggong1/learn-optimizer-rgdn) |
| TCSVT | [Deep Convolutional-Neural-Network-Based Channel Attention for Single Image Dynamic Scene Blind Deblurring](https://ieeexplore.ieee.org/document/9247132) | Blind · Motion | — |
| CVPR | [Cascaded Deep Video Deblurring Using Temporal Sharpness Prior](https://arxiv.org/abs/2004.02501) | Motion · Video | [Code](https://github.com/csbhr/CDVD-TSP) |
| CVPR | [Learning Event-Based Motion Deblurring](https://ieeexplore.ieee.org/document/9156741) | Motion | — |
| CVPR | [Variational-EM-Based Deep Learning for Noise-Blind Image Deblurring](https://ieeexplore.ieee.org/document/9157497) | Blind | [Code](https://github.com/ysnan/VEM-NBD) |
| CVPR | [Efficient Dynamic Scene Deblurring Using Spatially Variant Deconvolution Network With Optical Flow Guided Training](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Efficient_Dynamic_Scene_Deblurring_Using_Spatially_Variant_Deconvolution_Network_With_CVPR_2020_paper.pdf) | Motion · Deconvolution | — |
| CVPR | [Deblurring by Realistic Blurring](https://arxiv.org/abs/2004.01860) | Motion · Blur Synthesis | [Code](https://github.com/HDCVLab/Deblurring-by-Realistic-Blurring) |
| CVPR | [Spatially-Attentive Patch-Hierarchical Network for Adaptive Motion Deblurring](https://arxiv.org/abs/2004.05343) | Motion | — |
| CVPR | [Deblurring Using Analysis-Synthesis Networks Pair](https://arxiv.org/abs/2004.02956) | Motion | — |
| ECCV | [Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf) | Motion · Video | [Code](https://github.com/zzh-tech/ESTRNN) |
| ECCV | [Multi-Temporal Recurrent Neural Networks For Progressive Non-Uniform Single Image Deblurring With Incremental Temporal Training](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510324.pdf) | Motion | [Code](https://github.com/Dong1P/MTRNN) |
| ECCV | [Learning Event-Driven Video Deblurring and Interpolation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530681.pdf) | Motion · Video · Frame Interpolation | [Code](https://github.com/Lynn0306/LEDVDI) |
| ECCV | [Defocus Deblurring Using Dual-Pixel Data](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550120.pdf) | Defocus | [Code](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) |
| ECCV | [Real-World Blur Dataset for Learning and Benchmarking Deblurring Algorithms](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf) | Motion | [Code](https://github.com/rimchang/RealBlur) |
| ECCV | [OID: Outlier Identifying and Discarding in Blind Image Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700596.pdf) | Blind | [Project](https://drive.google.com/file/d/19PCEXVs6imWqqae37r5By-OCUlrNGHnd/view) |
| ECCV | [Enhanced Sparse Model for Blind Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700630.pdf) | Blind | [Project](https://drive.google.com/file/d/1HgLrWWh0Lx69kRh8xkm_peqNha1lNIOG/view) |

</details>

<a id="2019-papers"></a>

<details>
<summary><strong>2019 Papers (5)</strong></summary>

| Venue | Paper | Task | Resource |
|---|---|---|---|
| ICCV | [DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/abs/1908.03826v1) | Motion | [Code](https://github.com/VITA-Group/DeblurGANv2) |
| BMVC | [Blind Image Deconvolution using Pretrained Generative Priors](https://arxiv.org/abs/1908.07404v1) | Blind · Deconvolution | [Code](https://github.com/axium/Blind-Image-Deconvolution-using-Deep-Generative-Priors) |
| arXiv | [Efficient Blind Deblurring under High Noise Levels](https://arxiv.org/abs/1904.09154v2) | Blind | [Code](https://github.com/kidanger/high-noise-deblurring) |
| CVPR | [Deep Stacked Hierarchical Multi-Patch Network for Image Deblurring](https://arxiv.org/abs/1904.03468) | Motion | [Code](https://github.com/HongguangZhang/DMPHN-cvpr19-master) |
| CVPR | [Dynamic Scene Deblurring with Parameter Selective Sharing and Nested Skip Connections](https://ieeexplore.ieee.org/document/8953950) | Motion | [Code](https://github.com/firenxygao/deblur) |

</details>

## Datasets & Benchmarks

| Dataset | Focus | Signals | Capture | Link |
|---|---|---|---|---|
| GoPro | Motion | RGB | synthetic-from-high-fps | [Resource](https://seungjunnah.github.io/Datasets/gopro) |
| REDS | Video · Motion | Video | high-fps-video | [Resource](https://seungjunnah.github.io/Datasets/reds) |
| DPDD | Defocus | Dual Pixel | real | [Resource](https://abuolaim.nowaty.com/eccv_2020_dp_defocus_deblurring/) |
| HIDE | Motion | RGB | synthetic-from-high-fps | [Resource](https://github.com/joanshen0508/HA_deblur) |
| RealBlur | Motion | RGB · RAW | real | [Resource](https://cg.postech.ac.kr/research/realblur/) |
| Deblur-NeRF | Motion · Defocus · 3D Reconstruction | RGB | mixed | [Resource](https://limacv.github.io/deblurnerf/) |
| RSBlur | Motion · Blur Synthesis | RGB | mixed | [Resource](https://cg.postech.ac.kr/research/rsblur/) |
| ReLoBlur | Local Motion | RGB | real | [Resource](https://leiali.github.io/ReLoBlur_homepage/index.html) |
| HCBlur | Motion | Stereo | mixed | [Resource](https://github.com/rimchang/HCDeblur) |
| DAVIS-Blur | Video · Defocus | Video | synthetic | [Resource](https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/html/Morris_DaBiT_Depth_and_Blur_informed_Transformer_for_Video_Deblurring_WACVW_2025_paper.html) |
| QPDD | Defocus | Quad Pixel | real | [Resource](https://github.com/excllent123/QPD-Application) |
| GyroBlur | Motion | RGB · Gyro | mixed | [Resource](https://github.com/hmyang0727/GyroDeblurNet) |
| BlurRF-Synth | Motion · 3D Reconstruction | RGB | synthetic | [Resource](https://github.com/haeyun-choi/DeepDeblurRF) |
| T-RED | Motion · Video | Video · Events | real | [Resource](https://github.com/ZhijingS/EVDM) |
| Extreme Motion-Blur RGB-D | Motion · 3D Reconstruction | RGB-D | real | [Resource](https://github.com/KAISTVCLAB/gs-extreme-motion-blur) |
| GyroVD | Motion · Video | Video · Gyro | mixed | [Resource](https://github.com/rimchang/GyroDVD) |
| HSD | Motion | Stereo | real | [Resource](https://github.com/shinhoju/PECA) |
| CLDefocus | Defocus · Blur Synthesis | RGB | synthetic | [Resource](https://github.com/lykelee/CLDefocus) |

## Data & Maintenance

Structured migration: **2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026** complete; **0** pending years.

- Research data: [`data/`](data/)
- Taxonomy: [`data/taxonomy.yaml`](data/taxonomy.yaml)
- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Validation: `python scripts/validate_data.py`
- Generation: `python scripts/generate_catalog.py --output-dir .`

## Contributing

Contributions are welcome for missing papers, official code/project links, datasets, and corrections. Please follow the contribution policy and prefer primary sources.
