模型整体架构

  该模型是 DeepFusionAVGen，基于 Qwen3-VL 的多模态生成 DiT（Diffusion
  Transformer），支持文本驱动的视频/音频生成。推理流水线由三个主要组件构成。

  ---
  1. 文本编码器：Qwen3VLTextExtractor

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

  ---
  2. 生成主干：OmniFusionDiT

  2.1 关键超参数

  ┌─────────────────────┬─────────────────────────────────────┐
  │        参数         │                 值                  │
  ├─────────────────────┼─────────────────────────────────────┤
  │ hidden_size         │ 2560                                │
  ├─────────────────────┼─────────────────────────────────────┤
  │ num_layers          │ 36                                  │
  ├─────────────────────┼─────────────────────────────────────┤
  │ num_attention_heads │ 32                                  │
  ├─────────────────────┼─────────────────────────────────────┤
  │ num_key_value_heads │ 8                                   │
  ├─────────────────────┼─────────────────────────────────────┤
  │ head_dim            │ 128 (rope_dim_list=[32,48,48] 之和) │
  ├─────────────────────┼─────────────────────────────────────┤
  │ mlp_width_ratio     │ 3.8 → intermediate = 9728           │
  ├─────────────────────┼─────────────────────────────────────┤
  │ cond_embed_dim      │ 256                                 │
  ├─────────────────────┼─────────────────────────────────────┤
  │ patch_size          │ [1, 1, 1]                           │
  ├─────────────────────┼─────────────────────────────────────┤
  │ image_in_channels   │ 32                                  │
  ├─────────────────────┼─────────────────────────────────────┤
  │ audio_in_channels   │ 20                                  │
  └─────────────────────┴─────────────────────────────────────┘

  2.2 各子模块结构与维度

  Patch Embedding

  img_in_image (视频):
    输入: [B, 65, T, H, W]   # 65 = 32(video) + 32(cond) + 1(mask)
    Conv3d(65 → 2560, kernel=[1,1,1], stride=[1,1,1])
    展平 → [B, T*H*W, 2560]    # L_v = T * H * W

  img_in_audio (音频):
    输入: [B, 20, T_a]
    Conv1d(20 → 2560, k=7, p=3) → SiLU → ConvMLP(2560, 2560*4=10240)
    输出: [B, T_a, 2560]       # L_a = T_a

  条件嵌入

  time_in (TimestepEmbedder):
    t: [B] (float, 0~1 normalized)
    → sinusoidal encoding: [B, 256]
    → Linear(256,256) → SiLU → Linear(256,256)
    → vec: [B, 256]

  task_embedder (TaskEmbedder): # 生成视频，生成音频，生成视频和音频
    task_id: [B]  (0=video, 1=audio, 2=both)
    → Embedding(3, 256) → Linear(256,256) → SiLU → Linear(256,256)
    → [B, 256],  加到 vec

  modality_embedder (ModalityEmbedder):
    → Embedding(2, 2560)      # 0=audio, 1=video
    → [B, 2560] 广播后加到 token 上

  文本特征投影 (per_layer 模式)

  understanding_projs: ModuleList of 36 × Linear(2560 → 2560, bias=False)
    每层独立: [B, 512, 2560] → [B, 512, 2560]

  Fusion Layers (36 × OmniFusionDiTLayer)

  每层结构：

  OmniFusionDiTLayer (i):
    输入 hidden_states: [B, L_v + 512, 2560]
    vec:               [B, 256]

    mod (ModulateDiT):
      Linear(256 → 2560*6) → chunk(6) 得到:
      shift1, scale1, gate1, shift2, scale2, gate2  各 [B, 2560]

    -- Attention 分支 --
    input_layernorm (RMSNorm, dim=2560)
    modulate: * scale1 + shift1
    self_attn (Qwen3VL GQA):
      q_proj: Linear(2560 → 32*128 = 4096)
      k_proj: Linear(2560 →  8*128 = 1024)
      v_proj: Linear(2560 →  8*128 = 1024)
      q_norm, k_norm: RMSNorm(128)
      RoPE (rope_dim=[32,48,48], head_dim=128)
      Flash Attention 2 (full attention, is_causal=False)
      o_proj: Linear(4096 → 2560)
    residual + gate1 * attn_output

    -- MLP 分支 --
    post_attention_layernorm (RMSNorm, dim=2560)
    modulate: * scale2 + shift2
    mlp (SwiGLU):
      gate_proj: Linear(2560 → 9728)
      up_proj:   Linear(2560 → 9728)
      down_proj: Linear(9728 → 2560)
    residual + gate2 * mlp_output

    输出: [B, L_v + 512, 2560]
    取前 L_v: gen_tokens [B, L_v, 2560]

  Final Layers

  final_layer_image:
    LayerNorm(2560, elementwise_affine=False)
    adaLN_modulation: SiLU → Linear(256 → 5120)
    linear: Linear(2560 → 32)   # patch_size=[1,1,1], out_channels=32
    输入: [B, L_v, 2560]  →  输出: [B, L_v, 32]
    unpatchify: [B, L_v, 32] → [B, 32, T, H, W]

  final_layer_audio:
    同上，linear: Linear(2560 → 20)
    输入: [B, L_a, 2560]  →  输出: [B, 20, T_a]

  ---
  3. Video VAE：AutoencoderKLConv3D

  HunyuanVideo 1.5 VAE，空间压缩比 16。

  decode 输入: [B, 32, T, H/16, W/16]
  decode 输出: [B, 3, T, H, W]  (float, [-1,1] → [0,1])

  ---
  # 推理数据流
  （T2V，默认 resolution=256, video_length=1）

  输入: prompt (str)

  Step 1 - 初始化噪声
    latents: [1, 32, 1, 16, 16]   # C=32, T=1, H=W=256//16=16

  Step 2 - 文本编码 (Qwen3VLTextExtractor)
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
