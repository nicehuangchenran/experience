# 模型整体架构

  DeepFusionAVGen， T2I (T2V,TV2V,T2IA,T2A)

## 文本 [B, 512, 2560] \* 36 

  基于 Qwen3-VL-4B-Instruct，冻结参数，只做特征提取。

  输入: text_list (List[str])
    └── Tokenize (max_length=512)
    └── input_ids: [B, 512], attention_mask: [B, 512]
    └── Qwen3VL Language Model (36 Decoder Layers)
        hidden_size = 2560
        num_attention_heads = 16
        num_key_value_heads = 8
  输出: 36 x Tensor[B, 512, 2560]  (每层 hidden state)
       attention_mask: [B, 512]

## 视频 latent embed [B, L_v, 2560]

  img_in_image:
    输入: [B, 65, T, H, W]   # 65 = 32(video) + 32(cond) + 1(mask)
    Conv3d(65 → 2560, kernel=[1,1,1], stride=[1,1,1])
    展平 → [B, T\*H\*W, 2560]    # L_v = T * H * W

## 音频embed [B, L_a, 2560] 

  img_in_audio:
    输入: [B, 20, T_a]
    Conv1d(20 → 2560, k=7, p=3) → SiLU → ConvMLP(2560, 2560*4=10240)
    输出: [B, T_a, 2560]       # L_a = T_a

## condition embed

### time_step + task [B, 256]

  time_in (TimestepEmbedder):
    t: [B] (float, 0~1 normalized)
    → sinusoidal encoding: [B, 256]
    → Linear(256,256) → SiLU → Linear(256,256)
    → vec: [B, 256]

  task_embedder (TaskEmbedder): # 生成视频，生成音频，生成视频和音频
    task_id: [B]  (0=video, 1=audio, 2=both)
    → Embedding(3, 256) → Linear(256,256) → SiLU → Linear(256,256)
    → [B, 256],  加到 vec

### audio or video 模态生成任务 [B, 2560] 

  modality_embedder (ModalityEmbedder):
    → Embedding(2, 2560)      # 0=audio, 1=video
    → [B, 2560] 广播后加到 token 上

## OmniFusionDiTLayer

36层，每一层输入为 [B, L_v , 2560]

### concat(text,video)

 understanding_projs: ModuleList of 36 × Linear(2560 → 2560, bias=False)
    每层独立: [B, 512, 2560] → [B, 512, 2560]

输入 hidden_states: [B, L_v + 512, 2560]

### time_step + task -> shift, scale, gate [B, 2560]

mod (ModulateDiT):
  Linear(256 → 2560*6) → chunk(6) 得到:
  shift1, scale1, gate1, shift2, scale2, gate2  各 [B, 2560]

### layernorm and attention

input_layernorm (RMSNorm, dim=2560)
modulate: * scale1 + shift1

```python
self_attn (Qwen3VL GQA):
  q_proj: Linear(2560 → 32*128 = 4096)
  k_proj: Linear(2560 →  8*128 = 1024)
  v_proj: Linear(2560 →  8*128 = 1024)
  q_norm, k_norm: RMSNorm(128)
  RoPE (rope_dim=[32,48,48], head_dim=128)
  Flash Attention 2 (full attention, is_causal=False)
  o_proj: Linear(4096 → 2560)
residual + gate1 * attn_output
```

### layernorm and MLP

    post_attention_layernorm (RMSNorm, dim=2560)
    modulate: * scale2 + shift2
    mlp (SwiGLU):
      gate_proj: Linear(2560 → 9728)
      up_proj:   Linear(2560 → 9728)
      down_proj: Linear(9728 → 2560)
    residual + gate2 * mlp_output

输出: [B, L_v + 512, 2560]
只取视频token: [B, L_v, 2560]

## video/audio embedding to latent

 输入: [B, L_v, 2560]  →  输出: [B, L_v, 32]

  final_layer_image:
    LayerNorm(2560, elementwise_affine=False)
    adaLN_modulation: SiLU → Linear(256 → 5120)
    linear: Linear(2560 → 32)   # patch_size=[1,1,1], out_channels=32

​    unpatchify: [B, L_v, 32] → [B, 32, T, H, W]

  final_layer_audio:
    同上，linear: Linear(2560 → 20)
    输入: [B, L_a, 2560]  →  输出: [B, 20, T_a]

## Video VAE

  AutoencoderKLConv3D  HunyuanVideo 1.5 VAE，空间压缩比 16。

  decode 输入: [B, 32, T, H/16, W/16]
  decode 输出: [B, 3, T, H, W]  (float, [-1,1] → [0,1])

## Aduio VAE

TOD-VAE BigGAN 

# 推理

  （T2V，默认 resolution=256, video_length=1）

  输入: prompt (str)

  Step 1 - 初始化噪声
    latents: [1, 32, 1, 16, 16]   # C=32, T=1, H=W=256//16=16

  Step 2 - 文本编码 (Qwen3VLTextExtractor Qwen3-VL-4B-Instruct)
    cond_emb:   36 × [1, 512, 2560]
    cond_mask:  [1, 512]

  Step 3 - 扩散采样 (30步, FlowMatchEulerDiscreteScheduler)
    每步:
      t: [1]  (float, normalized)
      vec = time_in(t) + task_embedder(TASK_VIDEO): [1, 256]

      video input: cat(latents, zeros, mask): [1, 65, 1, 16, 16]
      video_tokens = img_in_image: [1, 1*16*16=256, 2560]
      + modality_embed(video): [1, 256, 2560]
    
      gen_tokens = video_tokens: [1, 256, 2560]
    
      for layer_i in 36:
          text_tokens = understanding_projs[i](cond_emb[i]): [1, 512, 2560]
          hidden = cat([gen_tokens, text_tokens]): [1, 768, 2560]
          hidden = OmniFusionDiTLayer(hidden, vec, RoPE, mask)
          gen_tokens = hidden[:, :256, :]: [1, 256, 2560]
    
      video_tok = final_layer_image(gen_tokens, vec): [1, 256, 32]
      pred_noise = unpatchify: [1, 32, 1, 16, 16]
    
      latents = scheduler.step(pred_noise, t, latents)

  Step 4 - VAE 解码
    latents: [1, 32, 1, 16, 16]
    → AutoencoderKLConv3D.decode
    → video_frames: [1, 3, 1, 256, 256]  → PNG 图片

---
  注意事项

  - CFG (guidance_scale=3.5) 时每步做两次前向（uncond + cond），使用 guidance_scale * (cond - uncond) 组合
  - qwen_features 每步需实时编码（Qwen3VLTextExtractor 在 @torch.no_grad() 下运行）
  - Attention mask 格式：Flash Attention 2 使用 2D mask [B, L_total]，L_total = L_v + 512

✻ Sautéed for 2m 23s

❯ text encoder的输出是固定[B, 512, 2560] ，还是根据输入的prompt长度动态调整？                                   

● 直接看 omni_encoder_4b.py 的 forward 方法即可回答。

  关键代码在第 115-122 行：

  inputs = self.processor.tokenizer(
      text_prompts,
      return_tensors="pt",
      padding="max_length",   # <-- 强制 pad 到最大长度
      truncation=True,
      max_length=512,         # <-- 固定 512
      add_special_tokens=False
  ).to(self.device)

  结论：固定 [B, 512, 2560]，不随 prompt 长度变化。

  - 短 prompt → 右侧 padding 补零到 512
  - 长 prompt → 截断到 512

  实际有效 token 的范围由 attention_mask: [B, 512]（1=有效，0=padding）标记，并一同返回给 OmniFusionDiT。

  在 OmniFusionDiT forward 中（第 1049-1050 行），这个 mask 用来构建 attention mask：

  gen_mask = torch.ones(batch_size, L_gen, ...)
  full_mask = torch.cat([gen_mask, qwen_mask.to(device)], dim=1)  # [B, L_gen + 512]

  padding 位置会被 Flash Attention 2 正确地屏蔽掉，不参与注意力计算。



# 流程

## 条件嵌入

### time and task ([B,256])
  [B, 256]   vec = time_emb + task_emb 后续全连接变成scale bia gate

  time_in (TimestepEmbedder):
    t: [B] (float, 0~1 normalized)
    → sinusoidal encoding: [B, 256]
    → Linear(256,256) → SiLU → Linear(256,256)
    → vec: [B, 256]

  task_embedder (TaskEmbedder):
    task_id: [B]  (0=video, 1=audio, 2=both)
    → Embedding(3, 256) → Linear(256,256) → SiLU → Linear(256,256)
    → [B, 256],  加到 vec

### video or audio modality ([B.2560] added to each token embedding)
  modality_embedder (ModalityEmbedder):
    → Embedding(2, 2560)      # 0=audio, 1=video
    → [B, 2560] 广播后加到 token 上

  video 任务加到video token上
## pipeline

以下用默认推理参数（B=1, resolution=256, video_length=1, num_inference_steps=30）展示完整数据流。VAE 空间压缩 16 倍，所以 latent 空间
  H=W=16。
                                                                                                                                               
---
  阶段一：文本编码（Qwen3VLTextExtractor，执行一次）                                                                                           
                                                                                                                                             
  输入                                                                                                                                         
    prompt: str  →  包装成 chat template 后 tokenize

  [B, 512]  input_ids
  [B, 512]  attention_mask          (1=有效 token, 0=padding)
       │
       ▼
  Qwen3-VL-4B Language Model
    36 × Decoder Layer
    hidden_size = 2560
       │
       ▼  output_hidden_states=True，取 36 层 hidden state
       │
  36 × [B, 512, 2560]   qwen_features   (每层一个张量)
       [B, 512]          qwen_mask

---
  阶段二：初始化噪声（执行一次）

  随机采样

  [B, 32, 1, 16, 16]   latents     (C=32, T=1, H=16, W=16)

---
  阶段三：扩散采样循环（重复 30 步）

  每步执行一次 OmniFusionDiT forward（CFG 时执行两次，uncond + cond）。

  3.1 准备视频输入

  [B, 32, 1, 16, 16]   video_latents
  [B, 32, 1, 16, 16]   video_cond_latents  (全零)
  [B,  1, 1, 16, 16]   video_mask          (全零)
       │
       ▼  torch.cat(dim=1)
  [B, 65, 1, 16, 16]   x_video             (65 = 32+32+1)

  3.2 Patch Embedding（视频）

  [B, 65, 1, 16, 16]   x_video
       │
       ▼  Conv3d(65→2560, kernel=[1,1,1], stride=[1,1,1])
       ▼  flatten spatial → L_v = 1×16×16 = 256
  [B, 256, 2560]        video_tokens
       │
       ▼  + ModalityEmbedder(1=video): Embedding(2,2560) → [B, 2560]
          广播后加到每个 token
  [B, 256, 2560]        video_tokens   (含 modality 信息)

  3.3 条件向量合成

  [B]   t  (float, 0~1 normalized)
       │
       ▼  TimestepEmbedder
          sinusoidal(t) → [B, 256]
          Linear(256→256) → SiLU → Linear(256→256)
  [B, 256]   time_emb

  [B]   task_id = 0  (TASK_VIDEO)
       │
       ▼  TaskEmbedder
          Embedding(3,256) → Linear(256→256) → SiLU → Linear(256→256)
  [B, 256]   task_emb

  [B, 256]   vec = time_emb + task_emb

  3.4 计算 RoPE

  video_grid_sizes = (T=1, H=16, W=16)
       │
       ▼  get_nd_rotary_pos_embed(rope_dim_list=[32,48,48], head_dim=128)
  [256, 128]   video_cos
  [256, 128]   video_sin

  text tokens 使用位置 0 的 RoPE:
  [512, 128]   text_cos = ones(1,128).expand(512,-1)
  [512, 128]   text_sin = zeros(1,128).expand(512,-1)

       ▼  cat(dim=0)
  [768, 128]   unified_rope_cos    (L_gen + L_text = 256 + 512)
  [768, 128]   unified_rope_sin

  3.5 Fusion Layers（36 层循环）

  第 i 层（i = 0..35）：

  ── 文本特征投影 ──────────────────────────────────

  36 × [B, 512, 2560]   qwen_features[i]
       │
       ▼  understanding_projs[i]: Linear(2560→2560, bias=False)
  [B, 512, 2560]   text_tokens

  ── 拼接 gen + text ──────────────────────────────

  [B, 256, 2560]   gen_tokens   (i=0 时 = video_tokens)
  [B, 512, 2560]   text_tokens
       │
       ▼  torch.cat(dim=1)
  [B, 768, 2560]   hidden_states

  ── 构建 Attention Mask ───────────────────────────

  [B, 256]   gen_mask  = ones
  [B, 512]   qwen_mask
       │
       ▼  cat(dim=1)
  [B, 768]   full_mask          (Flash Attention 2 格式)

  ── Modulation ───────────────────────────────────

  [B, 256]   vec
       │
       ▼  ModulateDiT: Linear(256 → 2560×6)
  [B, 15360]  → chunk(6)
  shift1, scale1, gate1   各 [B, 2560]
  shift2, scale2, gate2   各 [B, 2560]

  ── Attention 分支 ────────────────────────────────

  [B, 768, 2560]   hidden_states
       │
       ▼  input_layernorm: RMSNorm(2560)
  [B, 768, 2560]   normed
       │
       ▼  modulate: normed * (1 + scale1) + shift1
  [B, 768, 2560]   modulated

       ▼  q_proj: Linear(2560 → 32×128 = 4096)
  [B, 768, 4096]   Q
       ▼  k_proj: Linear(2560 →  8×128 = 1024)
  [B, 768, 1024]   K
       ▼  v_proj: Linear(2560 →  8×128 = 1024)
  [B, 768, 1024]   V

       ▼  reshape Q → [B, 32, 768, 128]
          reshape K → [B,  8, 768, 128]
          reshape V → [B,  8, 768, 128]
       ▼  q_norm, k_norm: RMSNorm(128)  (per-head)
       ▼  apply RoPE (unified_rope_cos/sin, [768,128])
       ▼  Flash Attention 2
          GQA: 8 kv-heads 广播给 32 q-heads
          mask = full_mask [B, 768]
          is_causal = False (全量 attention)
       ▼  reshape → [B, 768, 4096]
       ▼  o_proj: Linear(4096 → 2560)
  [B, 768, 2560]   attn_output

       ▼  residual + gate1 * attn_output
  [B, 768, 2560]   hidden_states

  ── MLP 分支 ─────────────────────────────────────

  [B, 768, 2560]   hidden_states
       │
       ▼  post_attention_layernorm: RMSNorm(2560)
       ▼  modulate: * (1 + scale2) + shift2
       ▼  SwiGLU MLP:
          gate_proj: Linear(2560 → 9728)
          up_proj:   Linear(2560 → 9728)
          SiLU(gate) * up → [B, 768, 9728]
          down_proj: Linear(9728 → 2560)
  [B, 768, 2560]   mlp_output

       ▼  residual + gate2 * mlp_output
  [B, 768, 2560]   hidden_states

  ── 提取 gen_tokens ──────────────────────────────

  [B, 768, 2560]   hidden_states
       │
       ▼  [:, :256, :]
  [B, 256, 2560]   gen_tokens    (进入下一层)

  3.6 Final Layer

  [B, 256, 2560]   gen_tokens  (36 层后)
  [B, 256]         vec

       ▼  final_layer_image:
          adaLN_modulation: SiLU → Linear(256→5120) → shift,scale 各[B,2560]
          LayerNorm(2560, elementwise_affine=False)
          modulate
          Linear(2560 → 32)      # patch_size=[1,1,1], out_channels=32
  [B, 256, 32]   video_tok

       ▼  unpatchify: reshape [B, T=1, H=16, W=16, C=32, 1, 1, 1]
          einsum → [B, 32, 1, 16, 16]
  [B, 32, 1, 16, 16]   pred_noise

  3.7 Scheduler Step

  [B, 32, 1, 16, 16]   pred_noise
  [B, 32, 1, 16, 16]   latents (当前)
  timestep t
       │
       ▼  FlowMatchEulerDiscreteScheduler.step
  [B, 32, 1, 16, 16]   latents (更新后，进入下一步)

---
  阶段四：VAE 解码（执行一次）

  [B, 32, 1, 16, 16]   latents (采样完成)
       │
       ▼  反归一化: latents * latents_std + latents_mean
  [B, 32, 1, 16, 16]   latents (原始尺度)
       │
       ▼  AutoencoderKLConv3D.decode (空间上采样 ×16)
  [B, 3, 1, 256, 256]  video_frames  (float, [-1, 1])
       │
       ▼  (x / 2 + 0.5).clamp(0, 1) → uint8
       ▼  einops: [C,T,H,W] → [T,H,W,C]
  [1, 256, 256, 3]     输出图片（video_length=1 保存为 PNG）

# 训练

## 代码入口

**train_qwen3vl_fusion.py（纯图像训练）**

  task_ids = torch.zeros(B, dtype=torch.long)      # 全 0，TASK_VIDEO

**train_qwen3vl_mixed.py（混合训练）**

图像 batch

  task_ids = torch.zeros(B, dtype=torch.long)      # TASK_VIDEO = 0

音频 batch

  task_ids = torch.ones(B, dtype=torch.long)       # TASK_AUDIO = 1

 一、模型初始化

  冻结部分（不参与反向传播）

  Qwen3VLTextExtractor (理解侧)
    - Qwen3-VL-4B 全部参数冻结
    - 不经过 accelerator.prepare()
    - 每步 @torch.no_grad() 提取特征

  AutoencoderKLConv3D (Video VAE)
    - 全部参数冻结
    - 仅用于 encode 原始图像 → latent

  FeaturesUtils (Audio VAE, 混合训练)
    - TOD-VAE + BigVGAN 全部冻结
    - 仅用于 encode 音频波形 → latent

  可训练部分

  OmniFusionDiT (生成侧)
    - 全部参数可训练
    - 经过 accelerator.prepare() 包装为 DDP/DeepSpeed
    - 默认 bf16 混合精度

  ---
  二、数据流水线

  图像数据（MultiModalMapDataset）

  Parquet 文件（按分辨率分桶）
    例：bucket=256_256, bucket=192_320, bucket=320_192 ...

  MultiModalBatchSampler
    - 按 bucket 分组，保证一个 batch 内分辨率相同
    - 支持多分辨率训练（不同 batch 分辨率可不同）

  DataLoader → batch:
    "image_path": [B, 3, H, W]   uint8 图像
    "caption":    List[str]       文本描述

  音频数据（混合训练，TimedAudioDataset）

  JSONL 文件（音频元数据 + 语音元数据）

  HomogeneousTaskBatchSampler
    按任务类型分组，每个 batch 内任务同质：
      tts:              30%   (文本转语音)
      t2a_single:       30%   (文本转单音)
      t2a_mix:          20%   (文本转混合音)
      speech_audio_mix: 10%
      t2a_timeline:     10%

  DataLoader → batch:
    raw_waveforms: [B, 1, T_samples]
    prompts:       List[str]

  混合模式的模态调度

  # 每步用确定性随机决定使用哪个模态（所有 DDP rank 保持一致）
  rng = random.Random(seed + global_step * 31337)
  use_image = rng.random() < image_prob   # 默认 image_prob=0.9

  ---
  三、单步训练流程（以图像为例）

  Step 1：VAE 编码（@no_grad）

  raw_image: [B, 3, H, W]  (float, [-1,1])
       │
       ▼  image.unsqueeze(2): [B, 3, 1, H, W]
       ▼  AutoencoderKLConv3D.encode().latent_dist.mode()
       ▼  * scaling_factor (0.5)
  latents: [B, 32, 1, H/16, W/16]   (bf16)

  Step 2：加噪（Flow Matching）

  noise = randn_like(latents): [B, 32, 1, H', W']

  # 用 logit-normal 分布采样时间步（偏向中间时间步，避免过多简单/困难样本）
  u ~ LogitNormal(mean=0, std=1)         # u ∈ (0,1)
  timestep_index = int(u * 1000)         # 映射到 0~999
  sigma = scheduler.sigmas[index]        # Flow Matching 的噪声权重

  # Flow Matching 加噪公式（线性插值）：
  noisy_latents = (1 - sigma) * latents + sigma * noise

  model_timesteps = timestep / 1000.0    # 归一化到 [0,1]，输入模型

  Step 3：文本特征提取（@no_grad）

  captions: List[str]
       │
       ▼  Qwen3VLTextExtractor（冻结）
       │  tokenize: padding="max_length", max_length=512
       │  Qwen3-VL-4B forward，取 36 层 hidden state
       │
  qwen_features: 36 × [B, 512, 2560]
  qwen_mask:     [B, 512]

  Step 4：模型前向

  输入:
    video_latents = noisy_latents: [B, 32, 1, H', W']
    t = model_timesteps:           [B]   (0~1)
    qwen_features:                 36 × [B, 512, 2560]
    qwen_mask:                     [B, 512]
    task_ids = TASK_VIDEO (=0):    [B]

  OmniFusionDiT.forward(...)
    （详见推理结构，训练与推理前向完全相同）

  输出:
    model_pred: [B, 32, 1, H', W']   (预测的速度场)

  Step 5：损失计算

  # Flow Matching 的训练目标是预测「速度」(velocity)：
  # 速度 = 从 latent 到 noise 的方向
  target = noise - latents: [B, 32, 1, H', W']

  # MSE Loss（先在空间维度展平，再对 batch 取均值）
  loss = mean( (model_pred.float() - target.float())² )
       = mean over [B, 32*1*H'*W'] → scalar

  Step 6：反向传播与优化

  accelerator.backward(loss)

  # 梯度同步后：
  clip_grad_norm_(model.parameters(), max_norm=1.0)

  optimizer.step()      # AdamW, lr=1e-4, β=(0.9,0.999), wd=0.01
  lr_scheduler.step()   # constant_with_warmup, warmup=500步
  optimizer.zero_grad()

  ---
  四、音频分支差异（混合训练）

  target_waveforms: [B, 1, T_samples]
       │
       ▼  GPUAudioProcessor.process_batch()
          - 重采样到 16kHz
          - 时间截断/padding 到 max_duration=10s
          - 随机 time dropout (prob=0.9)
       │
       ▼  audio_vae.wrapped_encode(waveforms) * 0.5
  audio_latents: [B, 20, T']     # T' = T_samples / 时间压缩比

  # 后续加噪、编码文本、模型前向、loss 计算与图像完全一致
  # 区别：
  #   - model(audio_latents=noisy_audio, video_latents=None, ...)
  #   - task_ids = TASK_AUDIO (=1)
  #   - target = noise - audio_latents

  ---
  五、训练配置总结

  ┌──────────────┬───────────────────────────────────────────────────────┐
  │    配置项    │                          值                           │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ 精度         │ bf16                                                  │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ 优化器       │ AdamW                                                 │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ 学习率       │ 1e-4，constant_with_warmup                            │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ warmup steps │ 500                                                   │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ 梯度裁剪     │ max_norm=1.0                                          │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ 梯度累积     │ 可配置（默认 1）                                      │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ 噪声调度器   │ FlowMatchEulerDiscreteScheduler，T=1000，shift=2.0    │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ 时间步采样   │ logit-normal，mean=0，std=1                           │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ 损失函数     │ MSE（预测速度场 v = noise - latents）                 │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ CFG 训练     │ prob_uncond=0.1，以 10% 概率将 caption 替换为空字符串 │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ Checkpoint   │ 每 N 步保存，支持断点续训                             │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │ 分布式       │ 单节点用 accelerate；多节点用 DeepSpeed ZeRO-2        │
  └──────────────┴───────────────────────────────────────────────────────┘

  
