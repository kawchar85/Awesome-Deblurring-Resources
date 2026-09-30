# Papers by Method

Generated from the structured research catalog.

[← Back to main catalog](../README.md)

## Index

- [Kernel Estimation](#kernel-estimation) — 21 papers
- [CNN](#cnn) — 13 papers
- [Frequency Domain](#frequency-domain) — 13 papers
- [Degradation Modeling](#degradation-modeling) — 12 papers
- [Diffusion](#diffusion) — 11 papers
- [Transformer](#transformer) — 10 papers
- [Physics Based](#physics-based) — 8 papers
- [Gaussian Splatting](#gaussian-splatting) — 6 papers
- [Unsupervised](#unsupervised) — 6 papers
- [Radiance Field](#radiance-field) — 4 papers
- [Recurrent](#recurrent) — 4 papers
- [Unrolled](#unrolled) — 4 papers
- [State Space](#state-space) — 3 papers
- [Depth Aware](#depth-aware) — 2 papers
- [Self Supervised](#self-supervised) — 2 papers
- [GAN](#gan) — 1 paper
- [Generative Prior](#generative-prior) — 1 paper

## CNN

Convolutional neural networks.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2024 | SPIE | [Estimation of motion blur kernel parameters using regression convolutional neural networks](https://arxiv.org/abs/2308.01381v3) | [Code](https://github.com/duckduckpig/regression_blur) |
| 2023 | ICML | [IRNeXt: Rethinking Convolutional Network Design for Image Restoration](https://dl.acm.org/doi/10.5555/3618408.3618669) | [Code](https://github.com/c-yn/IRNeXt) |
| 2022 | CVPRW | [HINet: Half Instance Normalization Network for Image Restoration](https://arxiv.org/abs/2105.06086) | [Code](https://github.com/megvii-model/HINet) |
| 2021 | CVPR | [Multi-Stage Progressive Image Restoration](https://arxiv.org/abs/2102.02808) | [Code](https://github.com/swz30/MPRNet) |
| 2021 | ICCV | [Rethinking Coarse-to-Fine Approach in Single Image Deblurring](https://arxiv.org/abs/2108.05054) | [Code](https://github.com/chosj95/MIMO-UNet) |
| 2021 | ICCV | [Single Image Defocus Deblurring Using Kernel-Sharing Parallel Atrous Convolutions](https://arxiv.org/abs/2108.09108) | [Code](https://github.com/HyeongseokSon1/KPAC) |
| 2020 | CVPR | [Deblurring Using Analysis-Synthesis Networks Pair](https://arxiv.org/abs/2004.02956) | — |
| 2020 | CVPR | [Efficient Dynamic Scene Deblurring Using Spatially Variant Deconvolution Network With Optical Flow Guided Training](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Efficient_Dynamic_Scene_Deblurring_Using_Spatially_Variant_Deconvolution_Network_With_CVPR_2020_paper.pdf) | — |
| 2020 | IEEE | [Raw Image Deblurring](https://arxiv.org/abs/2012.04264v1) | [Code](https://github.com/bob831009/raw_image_deblurring) |
| 2020 | IJCV | [Spatially-Adaptive Filter Units for Compact and Efficient Deep Neural Networks](https://arxiv.org/abs/1902.07474v2) | [Code](https://github.com/skokec/DAU-ConvNet) |
| 2020 | TCSVT | [Deep Convolutional-Neural-Network-Based Channel Attention for Single Image Dynamic Scene Blind Deblurring](https://ieeexplore.ieee.org/document/9247132) | — |
| 2019 | CVPR | [Deep Stacked Hierarchical Multi-Patch Network for Image Deblurring](https://arxiv.org/abs/1904.03468) | [Code](https://github.com/HongguangZhang/DMPHN-cvpr19-master) |
| 2019 | CVPR | [Dynamic Scene Deblurring with Parameter Selective Sharing and Nested Skip Connections](https://ieeexplore.ieee.org/document/8953950) | [Code](https://github.com/firenxygao/deblur) |

## Degradation Modeling

Explicit modeling/representation of the degradation process.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2026 | ECCV | [Realistic Compound-Lens Defocus Blur Synthesis](https://arxiv.org/abs/2607.05837) | [Code](https://github.com/lykelee/CLDefocus) |
| 2025 | ICCV | [Performing Defocus Deblurring by Modeling its Formation Process](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Performing_Defocus_Deblurring_by_Modeling_its_Formation_Process_ICCV_2025_paper.html) | — |
| 2024 | CVPR | [Blur2Blur: Blur Conversion for Unsupervised Image Deblurring on Unknown Domains](https://arxiv.org/abs/2403.16205) | [Code](https://github.com/VinAIResearch/Blur2Blur) |
| 2024 | CVPR | [ID-Blau: Image Deblurring by Implicit Diffusion-based reBLurring AUgmentation](https://arxiv.org/abs/2312.10998) | [Code](https://github.com/plusgood-steven/ID-Blau) |
| 2024 | CVPR | [Real-World Efficient Blind Motion Deblurring via Blur Pixel Discretization](https://arxiv.org/abs/2404.12168) | — |
| 2024 | ECCV | [Domain-adaptive Video Deblurring via Test-time Blurring](https://arxiv.org/abs/2407.09059) | [Code](https://github.com/Jin-Ting-He/DADeblur) |
| 2023 | ICCV | [Single Image Deblurring with Row-dependent Blur Magnitude](https://openaccess.thecvf.com/content/ICCV2023/html/Ji_Single_Image_Deblurring_with_Row-dependent_Blur_Magnitude_ICCV_2023_paper.html) | [Code](https://github.com/jixiang2016/RSS-T) |
| 2022 | ECCV | [Learning Degradation Representations for Image Deblurring](https://arxiv.org/abs/2208.05244) | [Code](https://github.com/dasongli1/Learning_degradation) |
| 2022 | ECCV | [Realistic Blur Synthesis for Learning Image Deblurring](https://arxiv.org/abs/2202.08771) | [Code](https://github.com/rimchang/RSBlur) |
| 2022 | ECCV | [Towards Real-World Video Deblurring by Exploring Blur Formation Process](https://arxiv.org/abs/2208.13184) | [Code](https://github.com/ljzycmd/rawblur) |
| 2021 | CVPR | [Digital Gimbal: End-to-end Deep Image Stabilization with Learnable Exposure Times](https://arxiv.org/abs/2012.04515) | [Code](https://github.com/omer11a/digital-gimbal) |
| 2020 | CVPR | [Deblurring by Realistic Blurring](https://arxiv.org/abs/2004.01860) | [Code](https://github.com/HDCVLab/Deblurring-by-Realistic-Blurring) |

## Depth Aware

Depth is used explicitly for deblurring.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2025 | WACV Workshops | [DaBiT: Depth and Blur informed Transformer for Video Deblurring](https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/html/Morris_DaBiT_Depth_and_Blur_informed_Transformer_for_Video_Deblurring_WACVW_2025_paper.html) | — |
| 2023 | CVPR | [K3DN: Disparity-Aware Kernel Estimation for Dual-Pixel Defocus Deblurring](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_K3DN_Disparity-Aware_Kernel_Estimation_for_Dual-Pixel_Defocus_Deblurring_CVPR_2023_paper.html) | — |

## Diffusion

Diffusion/generative diffusion methods.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2025 | AAAI | [Residual Diffusion Deblurring Model for Single Image Defocus Deblurring](https://ojs.aaai.org/index.php/AAAI/article/view/32303) | — |
| 2025 | CVPR | [Diffusion-based Event Generation for High-Quality Image Deblurring](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Diffusion-based_Event_Generation_for_High-Quality_Image_Deblurring_CVPR_2025_paper.html) | [Code](https://github.com/XinanXie/EGDeblurring) |
| 2025 | ICCV | [Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Learning_Deblurring_Texture_Prior_from_Unpaired_Data_with_Diffusion_Model_ICCV_2025_paper.html) | [Code](https://github.com/ChengxuLiu/TP-Diff) |
| 2025 | NeurIPS | [BlurDM: A Blur Diffusion Model for Image Deblurring](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4b43f14df70be3b93e8c415d46df0598-Abstract-Conference.html) | [Code](https://github.com/Jin-Ting-He/BlurDM) |
| 2025 | NeurIPS | [DeblurDiff: Real-World Image Deblurring with Generative Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e393677793767624f2821cec8bdd02f1-Abstract-Conference.html) | [Code](https://github.com/kkkls/DeblurDiff) |
| 2024 | CVPR | [Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light Enhancement and Deblurring](https://openaccess.thecvf.com/content/CVPR2024/html/Lv_Fourier_Priors-Guided_Diffusion_for_Zero-Shot_Joint_Low-Light_Enhancement_and_Deblurring_CVPR_2024_paper.html) | [Code](https://github.com/aipixel/FourierDiff) |
| 2024 | CVPR | [ID-Blau: Image Deblurring by Implicit Diffusion-based reBLurring AUgmentation](https://arxiv.org/abs/2312.10998) | [Code](https://github.com/plusgood-steven/ID-Blau) |
| 2024 | arXiv | [Fast Diffusion EM: a diffusion model for blind inverse problems with application to deconvolution](https://arxiv.org/abs/2309.00287v2) | [Code](https://github.com/claroche-r/fastdiffusionem) |
| 2023 | ICCV | [Multiscale Structure Guided Diffusion for Image Deblurring](https://arxiv.org/abs/2212.01789) | — |
| 2023 | ICML | [GibbsDDRM: A Partially Collapsed Gibbs Sampler for Solving Blind Inverse Problems with Denoising Diffusion Restoration](https://arxiv.org/abs/2301.12686v2) | [Code](https://github.com/sony/gibbsddrm) |
| 2023 | NeurIPS | [Hierarchical Integration Diffusion Model for Realistic Image Deblurring](https://arxiv.org/abs/2305.12966) | [Code](https://github.com/zhengchen1999/HI-Diff) |

## Frequency Domain

Fourier/frequency-domain priors or processing central to the method.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2026 | ECCV | [CogSENet: Blind Image Deblurring with Blur-Conditioned Semantic Routing and Explicit Frequency Fusion](https://arxiv.org/abs/2606.30030) | — |
| 2026 | ECCV | [Leveraging Phase Information to Boost Unrolled Network Learning for Image Deblurring](https://arxiv.org/abs/2607.00251) | — |
| 2025 | WACV | [Blind Image Deblurring with FFT-ReLU Sparsity Prior](https://openaccess.thecvf.com/content/WACV2025/html/Al_Radi_Blind_Image_Deblurring_with_FFT-ReLU_Sparsity_Prior_WACV_2025_paper.html) | [Code](https://github.com/Metalicana/Blind-Image-Deblurring-using-FFT-ReLU-with-Deep-Learning-Pipeline-Integration) |
| 2024 | CVPR | [Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light Enhancement and Deblurring](https://openaccess.thecvf.com/content/CVPR2024/html/Lv_Fourier_Priors-Guided_Diffusion_for_Zero-Shot_Joint_Low-Light_Enhancement_and_Deblurring_CVPR_2024_paper.html) | [Code](https://github.com/aipixel/FourierDiff) |
| 2024 | CVPR | [Frequency-aware Event-based Video Deblurring for Real-World Motion Blur](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Frequency-aware_Event-based_Video_Deblurring_for_Real-World_Motion_Blur_CVPR_2024_paper.html) | — |
| 2023 | AAAI | [Dual-Domain Attention for Image Deblurring](https://ojs.aaai.org/index.php/AAAI/article/view/25122) | [Code](https://github.com/c-yn/DDANet) |
| 2023 | AAAI | [Intriguing Findings of Frequency Selection for Image Deblurring](https://arxiv.org/abs/2111.11745) | [Code](https://github.com/INVOKERer/DeepRFT/tree/AAAI2023) |
| 2023 | CVPR | [Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring](https://arxiv.org/abs/2211.12250) | [Code](https://github.com/kkkls/FFTformer) |
| 2023 | ICCV | [Exploring Temporal Frequency Spectrum in Deep Video Deblurring](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Exploring_Temporal_Frequency_Spectrum_in_Deep_Video_Deblurring_ICCV_2023_paper.pdf) | — |
| 2023 | ICCV | [Multi-scale Residual Low-Pass Filter Network for Image Deblurring](https://ieeexplore.ieee.org/document/10377577) | — |
| 2023 | TCSVT | [Multi-Scale Frequency Separation Network for Image Deblurring](https://arxiv.org/abs/2206.00798) | [Code](https://github.com/LiQiang0307/MSFS-Net) |
| 2023 | TIP | [INFWIDE: Image and Feature Space Wiener Deconvolution Network for Non-blind Image Deblurring in Low-Light Conditions](https://ieeexplore.ieee.org/document/10047966) | [Code](https://github.com/zhihongz/infwide) |
| 2020 | NeurIPS | [Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring](https://arxiv.org/abs/2103.09962v1) | [Code](https://gitlab.mpi-klsb.mpg.de/jdong/dwdn) |

## GAN

Generative-adversarial-network-based restoration.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2019 | ICCV | [DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/abs/1908.03826v1) | [Code](https://github.com/VITA-Group/DeblurGANv2) |

## Gaussian Splatting

3D Gaussian splatting.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2026 | CVPR | [Event-Based Motion Deblurring Using Task-Oriented 3D Gaussian Event Representations](https://openaccess.thecvf.com/content/CVPR2026/html/Xue_Event-Based_Motion_Deblurring_Using_Task-Oriented_3D_Gaussian_Event_Representations_CVPR_2026_paper.html) | — |
| 2026 | CVPR | [MSCD-GS: Motion-Separated Cooperative Deblurring Dynamic Reconstruction via Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2026/html/Liao_MSCD-GS_Motion-Separated_Cooperative_Deblurring_Dynamic_Reconstruction_via_Gaussian_Splatting_CVPR_2026_paper.html) | — |
| 2026 | ECCV | [PRISM3D: Probabilistic Refinement and Robust Initialization for Physically Consistent Scene Modeling under Extreme Motion Blur](https://arxiv.org/abs/2607.03855) | [Code](https://github.com/GopiRajuMatta/PRISM3D) |
| 2025 | ICCV | [Splat-based 3D Scene Reconstruction with Extreme Motion-blur](https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Splat-based_3D_Scene_Reconstruction_with_Extreme_Motion-blur_ICCV_2025_paper.html) | [Code](https://github.com/KAISTVCLAB/gs-extreme-motion-blur) |
| 2024 | ECCV | [BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting](https://arxiv.org/abs/2403.11831) | [Code](https://github.com/WU-CVGL/BAD-Gaussians) |
| 2024 | ECCV | [Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion](https://arxiv.org/abs/2403.13327) | [Code](https://github.com/SpectacularAI/3dgs-deblur) |

## Generative Prior

Pretrained generative models used explicitly as image/degradation priors.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2019 | BMVC | [Blind Image Deconvolution using Pretrained Generative Priors](https://arxiv.org/abs/1908.07404v1) | [Code](https://github.com/axium/Blind-Image-Deconvolution-using-Deep-Generative-Priors) |

## Kernel Estimation

Explicit or parameterized blur-kernel estimation/modeling.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2026 | ECCV | [Event-Driven Motion Deblurring via Trajectory-Based Kernel Reconstruction](https://link.springer.com/book/10.1007/978-3-032-37314-4) | — |
| 2025 | CVPR | [Parameterized Blur Kernel Prior Learning for Local Motion Deblurring](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_Parameterized_Blur_Kernel_Prior_Learning_for_Local_Motion_Deblurring_CVPR_2025_paper.html) | — |
| 2024 | CVPR | [Motion-adaptive Separable Collaborative Filters for Blind Motion Deblurring](https://arxiv.org/abs/2404.13153) | [Code](https://github.com/ChengxuLiu/MISCFilter) |
| 2024 | ECCV | [Blind Image Deblurring with Noise-Robust Kernel Estimation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3024_ECCV_2024_paper.php) | [Code](https://github.com/csleemooo/BD_noise_robust_kernel_estimation) |
| 2024 | SPIE | [Estimation of motion blur kernel parameters using regression convolutional neural networks](https://arxiv.org/abs/2308.01381v3) | [Code](https://github.com/duckduckpig/regression_blur) |
| 2023 | AAAI | [Learnable Blur Kernel for Single-Image Defocus Deblurring in the Wild](https://ojs.aaai.org/index.php/AAAI/article/view/25446) | — |
| 2023 | CVPR | [K3DN: Disparity-Aware Kernel Estimation for Dual-Pixel Defocus Deblurring](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_K3DN_Disparity-Aware_Kernel_Estimation_for_Dual-Pixel_Defocus_Deblurring_CVPR_2023_paper.html) | — |
| 2023 | CVPR | [Neumann Network with Recursive Kernels for Single Image Defocus Deblurring](https://openaccess.thecvf.com/content/CVPR2023/papers/Quan_Neumann_Network_With_Recursive_Kernels_for_Single_Image_Defocus_Deblurring_CVPR_2023_paper.pdf) | [Code](https://github.com/csZcWu/NRKNet) |
| 2023 | CVPR | [Self-Supervised Blind Motion Deblurring With Deep Expectation Maximization](https://ieeexplore.ieee.org/document/10203880) | — |
| 2023 | CVPR | [Self-Supervised Non-Uniform Kernel Estimation With Flow-Based Motion Prior for Blind Image Deblurring](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_Self-Supervised_Non-Uniform_Kernel_Estimation_With_Flow-Based_Motion_Prior_for_Blind_CVPR_2023_paper.html) | [Code](https://github.com/Fangzhenxuan/UFPDeblur) |
| 2023 | CVPR | [Structured Kernel Estimation for Photon-Limited Deconvolution](https://arxiv.org/abs/2303.03472) | [Code](https://github.com/sanghviyashiitb/structured-kernel-cvpr23) |
| 2023 | ICCV | [Single Image Defocus Deblurring via Implicit Neural Inverse Kernels](https://openaccess.thecvf.com/content/ICCV2023/papers/Quan_Single_Image_Defocus_Deblurring_via_Implicit_Neural_Inverse_Kernels_ICCV_2023_paper.pdf) | [Code](https://github.com/xinyao240/INIKNet) |
| 2023 | IJCV | [Blind Image Deblurring with Unknown Kernel Size and Substantial Noise](https://arxiv.org/abs/2208.09483v2) | [Code](https://github.com/sun-umn/Blind-Image-Deblurring) |
| 2021 | CVPR | [Explore Image Deblurring via Encoded Blur Kernel Space](https://openaccess.thecvf.com/content/CVPR2021/html/Tran_Explore_Image_Deblurring_via_Encoded_Blur_Kernel_Space_CVPR_2021_paper.html) | [Code](https://github.com/VinAIResearch/blur-kernel-space-exploring) |
| 2021 | ICCV | [Single Image Defocus Deblurring Using Kernel-Sharing Parallel Atrous Convolutions](https://arxiv.org/abs/2108.09108) | [Code](https://github.com/HyeongseokSon1/KPAC) |
| 2021 | NeurIPS | [Gaussian Kernel Mixture Network for Single Image Defocus Deblurring](https://openreview.net/forum?id=kSR-_SVzDR-) | [Code](https://github.com/csZcWu/GKMNet) |
| 2020 | CVPR | [Variational-EM-Based Deep Learning for Noise-Blind Image Deblurring](https://ieeexplore.ieee.org/document/9157497) | [Code](https://github.com/ysnan/VEM-NBD) |
| 2020 | ECCV | [Enhanced Sparse Model for Blind Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700630.pdf) | [Project](https://drive.google.com/file/d/1HgLrWWh0Lx69kRh8xkm_peqNha1lNIOG/view) |
| 2020 | ECCV | [OID: Outlier Identifying and Discarding in Blind Image Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700596.pdf) | [Project](https://drive.google.com/file/d/19PCEXVs6imWqqae37r5By-OCUlrNGHnd/view) |
| 2020 | TCSVT | [A Simple Local Minimal Intensity Prior and An Improved Algorithm for Blind Image Deblurring](https://arxiv.org/abs/1906.06642v5) | [Code](https://github.com/FWen/deblur-pmp) |
| 2019 | arXiv | [Efficient Blind Deblurring under High Noise Levels](https://arxiv.org/abs/1904.09154v2) | [Code](https://github.com/kidanger/high-noise-deblurring) |

## Physics Based

Explicit image/sensor/blur-formation physics is central to the method.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2026 | ECCV | [A Benchmark for Heterogeneous Stereo Deblurring with Physically- and Epipolar-Constrained Cross Attention](https://arxiv.org/abs/2606.25962) | [Code](https://github.com/shinhoju/PECA) |
| 2026 | ECCV | [Event-Driven Motion Deblurring via Trajectory-Based Kernel Reconstruction](https://link.springer.com/book/10.1007/978-3-032-37314-4) | — |
| 2026 | ECCV | [PRISM3D: Probabilistic Refinement and Robust Initialization for Physically Consistent Scene Modeling under Extreme Motion Blur](https://arxiv.org/abs/2607.03855) | [Code](https://github.com/GopiRajuMatta/PRISM3D) |
| 2026 | ECCV | [Realistic Compound-Lens Defocus Blur Synthesis](https://arxiv.org/abs/2607.05837) | [Code](https://github.com/lykelee/CLDefocus) |
| 2025 | ICCV | [Performing Defocus Deblurring by Modeling its Formation Process](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Performing_Defocus_Deblurring_by_Modeling_its_Formation_Process_ICCV_2025_paper.html) | — |
| 2024 | CVPR | [EVS-assisted Joint Deblurring Rolling-Shutter Correction and Video Frame Interpolation through Sensor Inverse Modeling](https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_EVS-assisted_Joint_Deblurring_Rolling-Shutter_Correction_and_Video_Frame_Interpolation_through_CVPR_2024_paper.pdf) | — |
| 2023 | CVPR | [Structured Kernel Estimation for Photon-Limited Deconvolution](https://arxiv.org/abs/2303.03472) | [Code](https://github.com/sanghviyashiitb/structured-kernel-cvpr23) |
| 2021 | CVPR | [Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes](https://arxiv.org/abs/2104.01601) | [Code](https://github.com/zzh-tech/RSCD) |

## Radiance Field

NeRF or other neural radiance-field approaches.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2025 | CVPR | [Exploiting Deblurring Networks for Radiance Fields](https://openaccess.thecvf.com/content/CVPR2025/html/Choi_Exploiting_Deblurring_Networks_for_Radiance_Fields_CVPR_2025_paper.html) | [Code](https://github.com/haeyun-choi/DeepDeblurRF) |
| 2024 | CVPR | [Mitigating Motion Blur in Neural Radiance Fields with Events and Frames](https://arxiv.org/abs/2403.19780) | [Code](https://github.com/uzh-rpg/EvDeblurNeRF) |
| 2024 | ECCV | [BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](https://arxiv.org/abs/2407.02174v2) | [Code](https://github.com/WU-CVGL/BeNeRF) |
| 2023 | CVPR | [Hybrid Neural Rendering for Large-Scale Scenes with Motion Blur](https://arxiv.org/abs/2304.12652) | [Code](https://github.com/CVMI-Lab/HybridNeuralRendering) |

## Recurrent

Recurrent temporal architectures.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2022 | ECCV | [DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting](https://arxiv.org/abs/2111.09985) | [Code](https://github.com/JihyongOh/DeMFI) |
| 2022 | ECCV | [Efficient Video Deblurring Guided by Motion Magnitude](https://arxiv.org/abs/2207.13374) | [Code](https://github.com/sollynoay/MMP-RNN) |
| 2020 | ECCV | [Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf) | [Code](https://github.com/zzh-tech/ESTRNN) |
| 2020 | ECCV | [Multi-Temporal Recurrent Neural Networks For Progressive Non-Uniform Single Image Deblurring With Incremental Temporal Training](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510324.pdf) | [Code](https://github.com/Dong1P/MTRNN) |

## Self Supervised

Self-supervised learning is central to training.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2023 | CVPR | [Self-Supervised Blind Motion Deblurring With Deep Expectation Maximization](https://ieeexplore.ieee.org/document/10203880) | — |
| 2023 | CVPR | [Self-Supervised Non-Uniform Kernel Estimation With Flow-Based Motion Prior for Blind Image Deblurring](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_Self-Supervised_Non-Uniform_Kernel_Estimation_With_Flow-Based_Motion_Prior_for_Blind_CVPR_2023_paper.html) | [Code](https://github.com/Fangzhenxuan/UFPDeblur) |

## State Space

State-space/Mamba-style sequence or vision models.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2026 | CVPR Findings | [MVSSM: Motion-aware Visual State Space Model for Efficient Video Deblurring](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhou_MVSSM_Motion-aware_Visual_State_Space_Model_for_Efficient_Video_Deblurring_CVPRF_2026_paper.html) | — |
| 2025 | CVPR | [Efficient Visual State Space Model for Image Deblurring](https://openaccess.thecvf.com/content/CVPR2025/html/Kong_Efficient_Visual_State_Space_Model_for_Image_Deblurring_CVPR_2025_paper.html) | [Code](https://github.com/kkkls/EVSSM) |
| 2025 | ICCV | [EVDM: Event-based Real-world Video Deblurring with Mamba](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_EVDM_Event-based_Real-world_Video_Deblurring_with_Mamba_ICCV_2025_paper.html) | [Code](https://github.com/ZhijingS/EVDM) |

## Transformer

Transformer/attention-based architectures.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2026 | ECCV | [A Benchmark for Heterogeneous Stereo Deblurring with Physically- and Epipolar-Constrained Cross Attention](https://arxiv.org/abs/2606.25962) | [Code](https://github.com/shinhoju/PECA) |
| 2025 | AAAI | [Motion-adaptive Transformer for Event-based Image Deblurring](https://ojs.aaai.org/index.php/AAAI/article/view/32967) | — |
| 2025 | CVPR | [A Polarization-Aided Transformer for Image Deblurring via Motion Vector Decomposition](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_A_Polarization-Aided_Transformer_for_Image_Deblurring_via_Motion_Vector_Decomposition_CVPR_2025_paper.html) | — |
| 2025 | ICCV | [Efficient Concertormer for Image Deblurring and Beyond](https://openaccess.thecvf.com/content/ICCV2025/html/Kuo_Efficient_Concertormer_for_Image_Deblurring_and_Beyond_ICCV_2025_paper.html) | — |
| 2025 | WACV Workshops | [DaBiT: Depth and Blur informed Transformer for Video Deblurring](https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/html/Morris_DaBiT_Depth_and_Blur_informed_Transformer_for_Video_Deblurring_WACVW_2025_paper.html) | — |
| 2024 | CVPR | [A Unified Framework for Microscopy Defocus Deblur with Multi-Pyramid Transformer and Contrastive Learning](https://arxiv.org/abs/2403.02611) | [Code](https://github.com/PieceZhang/MPT-CataBlur) |
| 2024 | CVPR | [Blur-aware Spatio-temporal Sparse Transformer for Video Deblurring](https://arxiv.org/abs/2406.07551) | [Code](https://github.com/huicongzhang/BSSTNet) |
| 2023 | CVPR | [Blur Interpolation Transformer for Real-World Motion from Blur](https://arxiv.org/abs/2211.11423) | [Code](https://github.com/zzh-tech/BiT) |
| 2023 | CVPR | [Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring](https://arxiv.org/abs/2211.12250) | [Code](https://github.com/kkkls/FFTformer) |
| 2022 | ECCV | [Stripformer: Strip Transformer for Fast Image Deblurring](https://arxiv.org/abs/2204.04627) | [Code](https://github.com/pp00704831/Stripformer-ECCV-2022-) |

## Unrolled

Model-based/unrolled optimization networks.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2026 | ECCV | [Leveraging Phase Information to Boost Unrolled Network Learning for Image Deblurring](https://arxiv.org/abs/2607.00251) | — |
| 2025 | WACV | [Deep Joint Unrolling for Deblurring and Low-Light Image Enhancement (JUDE)](https://openaccess.thecvf.com/content/WACV2025/html/Vo_Deep_Joint_Unrolling_for_Deblurring_and_Low-Light_Image_Enhancement_JUDE_WACV_2025_paper.html) | [Project](https://jude.kc-ml2.com/) |
| 2020 | ECCV | [End-to-end Interpretable Learning of Non-blind Image Deblurring](https://arxiv.org/abs/2007.01769v2) | [Code](https://github.com/teboli/CPCR) |
| 2020 | TNNLS | [Learning Deep Gradient Descent Optimization for Image Deconvolution](https://arxiv.org/abs/1804.03368v2) | [Code](https://github.com/donggong1/learn-optimizer-rgdn) |

## Unsupervised

Unsupervised/unpaired learning is central to training.

| Year | Venue | Paper | Resource |
|---:|---|---|---|
| 2026 | CVPR | [Event-based Motion Deblurring with Unpaired Data](https://openaccess.thecvf.com/content/CVPR2026/html/Cho_Event-based_Motion_Deblurring_with_Unpaired_Data_CVPR_2026_paper.html) | [Code](https://github.com/Chohoonhee/EMP) |
| 2025 | ICCV | [Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Learning_Deblurring_Texture_Prior_from_Unpaired_Data_with_Diffusion_Model_ICCV_2025_paper.html) | [Code](https://github.com/ChengxuLiu/TP-Diff) |
| 2024 | CVPR | [Blur2Blur: Blur Conversion for Unsupervised Image Deblurring on Unknown Domains](https://arxiv.org/abs/2403.16205) | [Code](https://github.com/VinAIResearch/Blur2Blur) |
| 2024 | CVPR | [Unsupervised Blind Image Deblurring Based on Self-Enhancement](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Unsupervised_Blind_Image_Deblurring_Based_on_Self-Enhancement_CVPR_2024_paper.html) | — |
| 2023 | CVPR | [HyperCUT: Video Sequence from a Single Blurry Image using Unsupervised Ordering](https://arxiv.org/abs/2304.01686) | [Code](https://github.com/VinAIResearch/HyperCUT) |
| 2023 | CVPR | [Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior](https://arxiv.org/abs/2210.05361) | [Code](https://github.com/xl-tang01/UAUDeblur) |
