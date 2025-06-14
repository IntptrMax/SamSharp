using TorchSharp;
using TorchSharp.Modules;
using static SamSharp.Modeling.Common;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SamSharp.Modeling
{
	/// <summary>
	/// This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/Modeling/backbone/vit.py 
	/// </summary>
	internal class ImageEncoderViT : Module<Tensor, Tensor>
	{
		private readonly PatchEmbed patch_embed;
		private readonly Parameter? pos_embed;
		private readonly ModuleList<Block> blocks;
		private readonly Sequential neck;
		public readonly int img_size;

		/// <summary>
		/// This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/Modeling/backbone/vit.py 
		/// </summary>
		/// <param name="img_size">Input image size.</param>
		/// <param name="patch_size">Patch size.</param>
		/// <param name="in_chans">Number of input image channels.</param>
		/// <param name="embed_dim">Patch embedding dimension.</param>
		/// <param name="depth">Depth of ViT.</param>
		/// <param name="num_heads">Number of attention heads in each ViT block.</param>
		/// <param name="mlp_ratio">Ratio of mlp hidden dim to embedding dim.</param>
		/// <param name="out_chans">Number of output channels.</param>
		/// <param name="qkv_bias">If True, add a learnable bias to query, key, value.</param>
		/// <param name="use_abs_pos">If True, use absolute positional embeddings.</param>
		/// <param name="use_rel_pos">If True, add relative positional embeddings to the attention map.</param>
		/// <param name="rel_pos_zero_init">If True, zero initialize relative positional parameters.</param>
		/// <param name="window_size">Window size for window attention blocks.</param>
		/// <param name="global_attn_indexes">Indexes for blocks using global attention.</param>
		public ImageEncoderViT(int img_size = 1024,
		int patch_size = 16,
		int in_chans = 3,
		int embed_dim = 768,
		int depth = 12,
		int num_heads = 12,
		float mlp_ratio = 4.0f,
		int out_chans = 256,
		bool qkv_bias = true,
		bool use_abs_pos = true,
		bool use_rel_pos = false,
		bool rel_pos_zero_init = true,
		int window_size = 0,
		int[]? global_attn_indexes = null) : base(nameof(ImageEncoderViT))
		{
			this.img_size = img_size;
			patch_embed = new PatchEmbed(kernel_size: patch_size, stride: patch_size, in_chans: in_chans, embed_dim: embed_dim);
			pos_embed = null;
			if (use_abs_pos)
			{
				// Initialize absolute positional embedding with pretrain image size.
				pos_embed = Parameter(zeros(1, img_size / patch_size, img_size / patch_size, embed_dim));
			}

			blocks = new ModuleList<Block>();

			for (int i = 0; i < depth; i++)
			{
				Block block = new Block(
								   dim: embed_dim,
								   num_heads: num_heads,
								   mlp_ratio: mlp_ratio,
								   qkv_bias: qkv_bias,
								   use_rel_pos: use_rel_pos,
								   rel_pos_zero_init: rel_pos_zero_init,
								   window_size: !global_attn_indexes!.Contains(i) ? window_size : 0,
								   input_size: (img_size / patch_size, img_size / patch_size));
				blocks.append(block);
			}

			neck = Sequential(
				Conv2d(embed_dim, out_chans, kernel_size: 1, bias: false),
				new LayerNorm2d(out_chans),
				Conv2d(out_chans, out_chans, kernel_size: 3, padding: 1, bias: false),
				new LayerNorm2d(out_chans)
				);

			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			x = this.patch_embed.forward(x);
			if (this.pos_embed is not null)
			{
				x = x + this.pos_embed;
			}
			foreach (Block blk in this.blocks)
			{
				x = blk.forward(x);
			}

			x = this.neck.forward(x.permute(new long[] { 0, 3, 1, 2 }));
			return x;
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				// Dispose of all components
				patch_embed.Dispose();
				pos_embed?.Dispose();
				foreach (var block in blocks)
				{
					block?.Dispose();
				}
				neck?.Dispose();
			}
			base.Dispose(disposing);
		}


		/// <summary>  
		/// Image to Patch Embedding.  
		/// </summary>  
		private class PatchEmbed : Module<Tensor, Tensor>
		{
			private readonly Conv2d proj;

			/// <summary>
			/// Patch Embedding
			/// </summary>
			/// <param name="kernel_size">kernel size of the projection layer.</param>
			/// <param name="stride">stride of the projection layer.</param>
			/// <param name="padding">padding size of the projection layer.</param>
			/// <param name="in_chans">Number of input image channels.</param>
			/// <param name="embed_dim">Patch embedding dimension.</param>
			internal PatchEmbed(long kernel_size = 16, long stride = 16, long padding = 0, int in_chans = 3, int embed_dim = 768) : base(nameof(PatchEmbed))
			{
				proj = Conv2d(in_channels: in_chans, out_channels: embed_dim, kernel_size: (kernel_size, kernel_size), stride: (stride, stride), padding: (padding, padding));
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				x = proj.forward(x);
				// B C H W -> B H W C  
				x = x.permute(0, 2, 3, 1);
				return x;
			}
			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					proj?.Dispose();
				}
				base.Dispose(disposing);
			}
		}

		/// <summary>
		/// Multi-head Attention block with relative position embeddings.
		/// </summary>
		private class Attention : Module<Tensor, Tensor>
		{
			private readonly int num_heads;
			private readonly int head_dim;
			private readonly float scale;
			private readonly Linear qkv;
			private readonly Linear proj;
			private readonly bool use_rel_pos;
			private readonly Tensor? rel_pos_h;
			private readonly Tensor? rel_pos_w;

			/// <summary>
			/// Multi-head Attention block with relative position embeddings.
			/// </summary>
			/// <param name="dim">Number of input channels.</param>
			/// <param name="num_heads">Number of attention heads.</param>
			/// <param name="qkv_bias">If True, add a learnable bias to query, key, value.</param>
			/// <param name="use_rel_pos">If True, add relative positional embeddings to the attention map.</param>
			/// <param name="rel_pos_zero_init">If True, zero initialize relative positional parameters.</param>
			/// <param name="input_size">Input resolution for calculating the relative positional parameter size.</param>
			/// <exception cref="ArgumentNullException"></exception>
			internal Attention(int dim, int num_heads = 8, bool qkv_bias = true, bool use_rel_pos = false, bool rel_pos_zero_init = true, (int, int)? input_size = null) : base(nameof(Attention))
			{
				// Initialize the attention module with the given parameters
				// This is a placeholder for the actual implementation

				this.num_heads = num_heads;
				head_dim = dim / num_heads;
				scale = MathF.Pow(head_dim, -0.5f);
				qkv = Linear(dim, dim * 3, hasBias: qkv_bias);
				proj = Linear(dim, dim);
				this.use_rel_pos = use_rel_pos;

				if (this.use_rel_pos)
				{
					if (input_size is null)
					{
						throw new ArgumentNullException(nameof(input_size), "input_size must be provided when use_rel_pos is true.");
					}
					rel_pos_h = Parameter(zeros(2 * input_size.Value.Item1 - 1, head_dim));
					rel_pos_w = Parameter(zeros(2 * input_size.Value.Item2 - 1, head_dim));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				long B = x.shape[0];
				long H = x.shape[1];
				long W = x.shape[2];
				// qkv with shape (3, B, nHead, H * W, C)
				Tensor qkv = this.qkv.forward(x).reshape(B, H * W, 3, num_heads, -1).permute(2, 0, 3, 1, 4);
				// q, k, v with shape (B * nHead, H * W, C)
				Tensor[] qkv_mix = qkv.reshape(3, B * num_heads, H * W, -1).unbind(0);
				Tensor q = qkv_mix[0];
				Tensor k = qkv_mix[1];
				Tensor v = qkv_mix[2];
				Tensor attn = (q * scale).matmul(k.transpose(-2, -1));
				if (use_rel_pos)
				{
					attn = add_decomposed_rel_pos(attn, q, rel_pos_h!, rel_pos_w!, (H, W), (H, W));
				}
				attn = attn.softmax(dim: -1);
				x = attn.matmul(v).view(B, num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1);
				x = proj.forward(x);
				return x.MoveToOuterDisposeScope();
			}

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					qkv?.Dispose();
					proj?.Dispose();
					rel_pos_h?.Dispose();
					rel_pos_w?.Dispose();
				}
				base.Dispose(disposing);
			}

			/// <summary>
			/// Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
			/// </summary>
			/// <param name="attn">attention map.</param>
			/// <param name="q">query q in the attention layer with shape (B, q_h * q_w, C).</param>
			/// <param name="rel_pos_h">relative position embeddings (Lh, C) for height axis.</param>
			/// <param name="rel_pos_w">relative position embeddings (Lw, C) for width axis.</param>
			/// <param name="q_size">spatial sequence size of query q with (q_h, q_w).</param>
			/// <param name="k_size">spatial sequence size of key k with (k_h, k_w).</param>
			/// <returns>attention map with added relative positional embeddings.</returns>
			private Tensor add_decomposed_rel_pos(Tensor attn, Tensor q, Tensor rel_pos_h, Tensor rel_pos_w, (long, long) q_size, (long, long) k_size)
			{
				using var _ = NewDisposeScope();

				(long q_h, long q_w) = q_size;
				(long k_h, long k_w) = k_size;

				Tensor Rh = get_rel_pos(q_h, k_h, rel_pos_h);

				Tensor Rw = get_rel_pos(q_w, k_w, rel_pos_w);

				long B = q.shape[0];
				long dim = q.shape[2];

				Tensor r_q = q.reshape(B, q_h, q_w, dim);
				Tensor rel_h = einsum("bhwc,hkc->bhwk", r_q, Rh);
				Tensor rel_w = einsum("bhwc,wkc->bhwk", r_q, Rw);
				attn = (
					attn.view(B, q_h, q_w, k_h, k_w) + rel_h[.., .., .., .., TensorIndex.None] + rel_w[.., .., .., TensorIndex.None, ..]
				).view(B, q_h * q_w, k_h * k_w);
				return attn.MoveToOuterDisposeScope();

			}

			/// <summary>
			/// Get relative positional embeddings according to the relative positions of query and key sizes.
			/// </summary>
			/// <param name="q_size">size of query q.</param>
			/// <param name="k_size">size of key k.</param>
			/// <param name="rel_pos">relative position embeddings (L, C).</param>
			/// <returns>Extracted positional embeddings according to relative positions.</returns>
			private Tensor get_rel_pos(long q_size, long k_size, Tensor rel_pos)
			{
				using var _ = NewDisposeScope();

				int max_rel_dist = (int)(2 * Math.Max(q_size, k_size) - 1);
				// Interpolate rel pos if needed.

				Tensor rel_pos_resized = zeros(0);
				if (rel_pos.shape[0] != max_rel_dist)
				{
					// Interpolate rel pos.
					rel_pos_resized = functional.interpolate(
							rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
							size: new long[] { max_rel_dist },
							mode: InterpolationMode.Linear);
					rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0);
				}
				else
				{
					rel_pos_resized = rel_pos;
				}

				// Scale the coords with short length if shapes for q and k are different.
				Tensor q_coords = arange(q_size)[.., TensorIndex.None] * max(k_size / q_size, 1.0);
				Tensor k_coords = arange(k_size)[TensorIndex.None, ..] * max(q_size / k_size, 1.0);
				Tensor relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0);
				return rel_pos_resized[relative_coords.@long()].MoveToOuterDisposeScope();
			}

		}

		/// <summary>
		/// Transformer blocks with support of window attention and residual propagation blocks
		/// </summary>
		private class Block : Module<Tensor, Tensor>
		{
			private readonly LayerNorm norm1;
			private readonly Attention attn;
			private readonly LayerNorm norm2;
			private readonly MLPBlock mlp;
			private readonly int window_size;

			/// <summary>
			/// Transformer blocks with support of window attention and residual propagation blocks
			/// </summary>
			/// <param name="dim">Number of input channels.</param>
			/// <param name="num_heads">Number of attention heads in each ViT block.</param>
			/// <param name="mlp_ratio">Ratio of mlp hidden dim to embedding dim.</param>
			/// <param name="qkv_bias">If True, add a learnable bias to query, key, value.</param>
			/// <param name="use_rel_pos">If True, add relative positional embeddings to the attention map.</param>
			/// <param name="rel_pos_zero_init">If True, zero initialize relative positional parameters.</param>
			/// <param name="window_size">indow size for window attention blocks. If it equals 0, then use global attention.</param>
			/// <param name="input_size">Input resolution for calculating the relative positional parameter size.</param>
			public Block(int dim, int num_heads, float mlp_ratio = 4.0f, bool qkv_bias = true, bool use_rel_pos = false, bool rel_pos_zero_init = true, int window_size = 0, (int, int)? input_size = null) : base(nameof(Block))
			{
				this.window_size = window_size;
				norm1 = LayerNorm(dim, eps: 1e-6);
				attn = new Attention(dim, num_heads: num_heads, qkv_bias: qkv_bias, use_rel_pos: use_rel_pos, rel_pos_zero_init: rel_pos_zero_init, input_size = (window_size == 0 ? input_size : (window_size, window_size)));
				norm2 = LayerNorm(dim, eps: 1e-6);
				mlp = new MLPBlock(embedding_dim: dim, mlp_dim: (int)(dim * mlp_ratio));
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				Tensor shortcut = x;
				x = norm1.forward(x);
				// Window partition
				long H = 0, W = 0;
				(int, int) pad_hw = (0, 0);
				if (window_size > 0)
				{
					(H, W) = (x.shape[1], x.shape[2]);
					(x, pad_hw) = window_partition(x, window_size);
				}

				x = attn.forward(x);
				// Reverse window partition
				if (window_size > 0)
				{
					x = window_unpartition(x, window_size, pad_hw, ((int)H, (int)W));
				}
				x = shortcut + x;
				x = x + mlp.forward(norm2.forward(x));
				return x.MoveToOuterDisposeScope();
			}

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					norm1?.Dispose();
					attn?.Dispose();
					norm2?.Dispose();
					mlp?.Dispose();
				}
				base.Dispose(disposing);
			}
		}

		/// <summary>
		/// Partition into non-overlapping windows with padding if needed.
		/// </summary>
		/// <param name="x">input tokens with [B, H, W, C].</param>
		/// <param name="window_size">window size.</param>
		/// <returns>windows:windows after partition with [B * num_windows, window_size, window_size, C].(Hp, Wp): padded height and width before partition</returns>
		private static (Tensor, (int, int)) window_partition(Tensor x, int window_size)
		{
			int B = (int)x.shape[0];
			int H = (int)x.shape[1];
			int W = (int)x.shape[2];
			int C = (int)x.shape[3];

			int pad_h = (window_size - H % window_size) % window_size;
			int pad_w = (window_size - W % window_size) % window_size;

			if (pad_h > 0 || pad_w > 0)
			{
				x = functional.pad(x, new long[] { 0, 0, 0, pad_w, 0, pad_h });
			}
			(int Hp, int Wp) = (H + pad_h, W + pad_w);
			x = x.view(B, Hp / window_size, window_size, Wp / window_size, window_size, C);

			Tensor windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C);
			return (windows, (Hp, Wp));
		}

		/// <summary>
		/// Window unpartition into original sequences and removing padding.
		/// </summary>
		/// <param name="windows">input tokens with [B * num_windows, window_size, window_size, C].</param>
		/// <param name="window_size">window size.</param>
		/// <param name="pad_hw">padded height and width (Hp, Wp).</param>
		/// <param name="hw">original height and width (H, W) before padding.</param>
		/// <returns>unpartitioned sequences with [B, H, W, C].</returns>
		private static Tensor window_unpartition(Tensor windows, int window_size, (int, int) pad_hw, (int, int) hw)
		{
			(int Hp, int Wp) = pad_hw;
			(int H, int W) = hw;

			int B = (int)windows.shape[0] / (Hp * Wp / window_size / window_size);
			Tensor x = windows.view(B, Hp / window_size, Wp / window_size, window_size, window_size, -1);
			x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1);
			if (Hp > H || Wp > W)
			{
				x = x[.., ..H, ..W, ..].contiguous();
			}
			return x;

		}

	}
}
