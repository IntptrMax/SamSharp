using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SamSharp.Modeling
{
	internal class Transformer
	{
		public class TwoWayTransformer : Module<Tensor, Tensor, Tensor, (Tensor, Tensor)>
		{
			private readonly int depth;
			private readonly int embedding_dim;
			private readonly int num_heads;
			private readonly int mlp_dim;
			private readonly ModuleList<TwoWayAttentionBlock> layers;
			private readonly Attention final_attn_token_to_image;
			private readonly LayerNorm norm_final_attn;

			/// <summary>
			/// A transformer decoder that attends to an input image using queries whose positional embedding is supplied.
			/// </summary>
			/// <param name="depth">number of layers in the transformer</param>
			/// <param name="embedding_dim">the channel dimension for the input embeddings</param>
			/// <param name="num_heads">the number of heads for multihead attention. Must divide embedding_dim</param>
			/// <param name="mlp_dim">the channel dimension internal to the MLP block</param>
			/// <param name="attention_downsample_rate"></param>
			public TwoWayTransformer(int depth, int embedding_dim, int num_heads, int mlp_dim, int attention_downsample_rate = 2) : base(nameof(TwoWayTransformer))
			{
				this.depth = depth;
				this.embedding_dim = embedding_dim;
				this.num_heads = num_heads;
				this.mlp_dim = mlp_dim;
				this.layers = new ModuleList<TwoWayAttentionBlock>();
				for (int i = 0; i < depth; i++)
				{
					this.layers.append(new TwoWayAttentionBlock(
					embedding_dim: embedding_dim,
					num_heads: num_heads,
					mlp_dim: mlp_dim,
					attention_downsample_rate: attention_downsample_rate,
					skip_first_layer_pe: (i == 0)));
				}
				this.final_attn_token_to_image = new Attention(embedding_dim, num_heads, downsample_rate: attention_downsample_rate);
				this.norm_final_attn = nn.LayerNorm(embedding_dim);

				RegisterComponents();
			}

			/// <summary>
			/// Forwards the Transformer, attending to an image embedding with positional encoding and a point embedding.
			/// </summary>
			/// <param name="image_embedding">image to attend to. Should be shape B X embedding_dim X h X w for any h and w.</param>
			/// <param name="image_pe">the positional encoding to add to the image. Must have the same shape as image_embedding.</param>
			/// <param name="point_embedding">the embedding to add to the query points. Must have shape B X N_points X embedding_dim for any N_points.</param>
			/// <returns>torch.Tensor: the processed point_embedding.<br/>torch.Tensor: the processed image_embedding</returns>
			public override (Tensor, Tensor) forward(Tensor image_embedding, Tensor image_pe, Tensor point_embedding)
			{
				using var _ = NewDisposeScope();
				// BxCxHxW -> BxHWxC == B X N_image_tokens X C
				long bs = image_embedding.shape[0];
				long c = image_embedding.shape[1];
				long h = image_embedding.shape[2];
				long w = image_embedding.shape[3];

				image_embedding = image_embedding.flatten(2).permute(0, 2, 1);
				image_pe = image_pe.flatten(2).permute(0, 2, 1);

				// Prepare queries
				Tensor queries = point_embedding;
				Tensor keys = image_embedding;

				// Apply Transformer blocks and final layernorm
				foreach (var layer in this.layers)
				{
					(queries, keys) = layer.forward(queries: queries, keys: keys, query_pe: point_embedding, key_pe: image_pe);
				}
				// Apply the final attention layer from the points to the image
				Tensor q = queries + point_embedding;
				Tensor k = keys + image_pe;
				Tensor attn_out = this.final_attn_token_to_image.forward(q: q, k: k, v: keys);
				queries = queries + attn_out;
				queries = this.norm_final_attn.forward(queries);
				return (queries.MoveToOuterDisposeScope(), keys.MoveToOuterDisposeScope());
			}
		}


		public class TwoWayAttentionBlock : Module<Tensor, Tensor, Tensor, Tensor, (Tensor, Tensor)>
		{
			private readonly Attention self_attn;
			private readonly LayerNorm norm1;
			private readonly Attention cross_attn_token_to_image;
			private readonly LayerNorm norm2;
			private readonly Common.MLPBlock mlp;
			private readonly LayerNorm norm3;
			private readonly LayerNorm norm4;
			private readonly Attention cross_attn_image_to_token;
			private readonly bool skip_first_layer_pe;

			/// <summary>
			/// A Transformer block with four layers: <br/>
			/// (1) self-attention of sparse inputs, <br/>
			/// (2) cross attention of sparse inputs to dense inputs, <br/>
			/// (3) mlp block on sparse inputs, and<br/>
			/// (4) cross attention of dense inputs to sparse inputs.
			/// </summary>
			/// <param name="embedding_dim">the channel dimension of the embeddings</param>
			/// <param name="num_heads">the number of heads in the attention layers</param>
			/// <param name="mlp_dim">the hidden dimension of the mlp block</param>
			/// <param name="attention_downsample_rate"></param>
			/// <param name="skip_first_layer_pe">skip the PE on the first layer</param>
			public TwoWayAttentionBlock(int embedding_dim, int num_heads, int mlp_dim = 2048, int attention_downsample_rate = 2, bool skip_first_layer_pe = false) : base(nameof(TwoWayAttentionBlock))
			{
				this.self_attn = new Attention(embedding_dim, num_heads);
				this.norm1 = nn.LayerNorm(embedding_dim);
				this.cross_attn_token_to_image = new Attention(embedding_dim, num_heads, downsample_rate: attention_downsample_rate);
				this.norm2 = nn.LayerNorm(embedding_dim);
				this.mlp = new Common.MLPBlock(embedding_dim, mlp_dim, Common.MLPBlock.ActivationType.ReLU);
				this.norm3 = nn.LayerNorm(embedding_dim);
				this.norm4 = nn.LayerNorm(embedding_dim);
				this.cross_attn_image_to_token = new Attention(embedding_dim, num_heads, downsample_rate: attention_downsample_rate);
				this.skip_first_layer_pe = skip_first_layer_pe;
				RegisterComponents();
			}

			public override (Tensor, Tensor) forward(Tensor queries, Tensor keys, Tensor query_pe, Tensor key_pe)
			{
				using var _ = NewDisposeScope();
				Tensor q = torch.zeros(0);
				Tensor attn_out = torch.zeros(0);
				if (this.skip_first_layer_pe)
				{
					queries = this.self_attn.forward(q: queries, k: queries, v: queries);
				}
				else
				{
					q = queries + query_pe;
					attn_out = this.self_attn.forward(q: q, k: q, v: queries);
					queries = queries + attn_out;
				}

				queries = this.norm1.forward(queries);

				// Cross attention block, tokens attending to image embedding
				q = queries + query_pe;
				Tensor k = keys + key_pe;
				attn_out = this.cross_attn_token_to_image.forward(q: q, k: k, v: keys);
				queries = queries + attn_out;
				queries = this.norm2.forward(queries);

				// MLP block
				Tensor mlp_out = this.mlp.forward(queries);
				queries = queries + mlp_out;
				queries = this.norm3.forward(queries);

				// Cross attention block, image embedding attending to tokens
				q = queries + query_pe;
				k = keys + key_pe;
				attn_out = this.cross_attn_image_to_token.forward(q: k, k: q, v: queries);
				keys = keys + attn_out;
				keys = this.norm4.forward(keys);
				return (queries.MoveToOuterDisposeScope(), keys.MoveToOuterDisposeScope());
			}
		}


		public class Attention : Module<Tensor, Tensor, Tensor, Tensor>
		{
			private readonly int embedding_dim;
			private readonly int internal_dim;
			private readonly int num_heads;
			private readonly Linear q_proj;
			private readonly Linear k_proj;
			private readonly Linear v_proj;
			private readonly Linear out_proj;

			public Attention(int embedding_dim, int num_heads, int downsample_rate = 1) : base(nameof(Attention))
			{
				this.embedding_dim = embedding_dim;
				this.internal_dim = embedding_dim / downsample_rate;
				this.num_heads = num_heads;

				if (this.internal_dim % num_heads != 0)
				{
					throw new ArgumentException("num_heads must divide embedding_dim.", nameof(num_heads));
				}

				this.q_proj = nn.Linear(embedding_dim, this.internal_dim);
				this.k_proj = nn.Linear(embedding_dim, this.internal_dim);
				this.v_proj = nn.Linear(embedding_dim, this.internal_dim);
				this.out_proj = nn.Linear(this.internal_dim, embedding_dim);

				RegisterComponents();
			}

			private Tensor _separate_heads(Tensor x, int num_heads)
			{
				long b = x.shape[0];
				long n = x.shape[1];
				long c = x.shape[2];
				x = x.reshape(b, n, num_heads, c / num_heads);
				return x.transpose(1, 2);  // B X N_heads X N_tokens X C_per_head
			}

			private Tensor _recombine_heads(Tensor x)
			{
				long b = x.shape[0];
				long n_heads = x.shape[1];
				long n_tokens = x.shape[2];
				long c_per_head = x.shape[3];
				x = x.transpose(1, 2);  // B X N_tokens X N_heads X C_per_head
				return x.reshape(b, n_tokens, n_heads * c_per_head);  // B X N_tokens X C
			}

			public override Tensor forward(Tensor q, Tensor k, Tensor v)
			{
				using var _ = NewDisposeScope();
				q = this.q_proj.forward(q);
				k = this.k_proj.forward(k);
				v = this.v_proj.forward(v);

				// Separate into heads
				q = this._separate_heads(q, this.num_heads);
				k = this._separate_heads(k, this.num_heads);
				v = this._separate_heads(v, this.num_heads);

				// Attention
				long c_per_head = q.shape[3];

				Tensor attn = q.matmul(k.permute(0, 1, 3, 2));  // B X N_heads X N_tokens X N_tokens;
				attn = attn / Math.Sqrt(c_per_head);
				attn = torch.softmax(attn, dim: -1);

				// Get output
				Tensor @out = attn.matmul(v);
				@out = this._recombine_heads(@out);
				@out = this.out_proj.forward(@out);
				return @out.MoveToOuterDisposeScope();
			}

		}


	}
}
