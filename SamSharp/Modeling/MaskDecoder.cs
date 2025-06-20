using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SamSharp.Modeling
{
	internal class MaskDecoder : Module<Tensor, Tensor, Tensor, Tensor, bool, (Tensor, Tensor)>
	{
		private readonly Module<Tensor, Tensor, Tensor, (Tensor, Tensor)> transformer;
		private readonly int num_multimask_outputs;
		private readonly int num_mask_tokens;
		private readonly Embedding iou_token;
		private readonly Embedding mask_tokens;
		private readonly Sequential output_upscaling;
		private readonly ModuleList<MLP> output_hypernetworks_mlps;
		private readonly MLP iou_prediction_head;

		/// <summary>
		/// Predicts masks given an image and prompt embeddings, using a Transformer architecture.
		/// </summary>
		/// <param name="transformer_dim">the channel dimension of the Transformer</param>
		/// <param name="transformer">the Transformer used to predict masks</param>
		/// <param name="num_multimask_outputs">the number of masks to predict when disambiguating masks</param>
		/// <param name="iou_head_depth">the depth of the MLP used to predict mask quality</param>
		/// <param name="iou_head_hidden_dim">the hidden dimension of the MLP used to predict mask quality</param>
		public MaskDecoder(int transformer_dim, Module<Tensor, Tensor, Tensor, (Tensor, Tensor)> transformer, int num_multimask_outputs = 3, int iou_head_depth = 3, int iou_head_hidden_dim = 256) : base(nameof(MaskDecoder))
		{
			this.transformer = transformer;
			this.num_multimask_outputs = num_multimask_outputs;
			this.iou_token = Embedding(1, transformer_dim);
			this.num_mask_tokens = num_multimask_outputs + 1;
			this.mask_tokens = Embedding(this.num_mask_tokens, transformer_dim);

			this.output_upscaling = Sequential(
				ConvTranspose2d(transformer_dim, transformer_dim / 4, kernel_size: 2, stride: 2),
					new Common.LayerNorm2d(transformer_dim / 4),
					GELU(),
					ConvTranspose2d(transformer_dim / 4, transformer_dim / 8, kernel_size: 2, stride: 2),
					GELU());


			this.output_hypernetworks_mlps = new ModuleList<MLP>();
			for (int i = 0; i < this.num_mask_tokens; i++)
			{
				output_hypernetworks_mlps.Add(new MLP(transformer_dim, transformer_dim, transformer_dim / 8, 3));
			}
			this.iou_prediction_head = new MLP(transformer_dim, iou_head_hidden_dim, this.num_mask_tokens, iou_head_depth);
			RegisterComponents();

		}

		/// <summary>
		/// Predict masks given image and prompt embeddings.
		/// </summary>
		/// <param name="image_embeddings">the embeddings from the image encoder</param>
		/// <param name="image_pe">positional encoding with the shape of image_embeddings</param>
		/// <param name="sparse_prompt_embeddings">the embeddings of the points and boxes</param>
		/// <param name="dense_prompt_embeddings">the embeddings of the mask inputs</param>
		/// <param name="multimask_output">Whether to return multiple masks or a single</param>
		/// <returns>(batched predicted masks,batched predictions of mask quality)</returns>
		public override (Tensor, Tensor) forward(Tensor image_embeddings, Tensor image_pe, Tensor sparse_prompt_embeddings, Tensor dense_prompt_embeddings, bool multimask_output = false)
		{
			(Tensor masks, Tensor iou_pred) = this.predict_masks(image_embeddings: image_embeddings, image_pe: image_pe, sparse_prompt_embeddings: sparse_prompt_embeddings, dense_prompt_embeddings: dense_prompt_embeddings);
			// Select the correct mask or masks for output
			TensorIndex mask_slice = multimask_output ? TensorIndex.Slice(1, null) : TensorIndex.Slice(0, 1);
			masks = masks[.., mask_slice, .., ..];
			iou_pred = iou_pred[.., mask_slice];
			// Prepare output
			return (masks, iou_pred);
		}

		/// <summary>
		/// Predicts masks.
		/// </summary>
		/// <param name="image_embeddings"></param>
		/// <param name="image_pe"></param>
		/// <param name="sparse_prompt_embeddings"></param>
		/// <param name="dense_prompt_embeddings"></param>
		/// <returns></returns>
		private (Tensor, Tensor) predict_masks(Tensor image_embeddings, Tensor image_pe, Tensor sparse_prompt_embeddings, Tensor dense_prompt_embeddings)
		{
			using var _ = NewDisposeScope();
			// Concatenate output tokens
			Tensor output_tokens = torch.cat(new Tensor[] { this.iou_token.weight!, this.mask_tokens.weight! }, dim: 0);
			output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1);
			Tensor tokens = torch.cat(new Tensor[] { output_tokens, sparse_prompt_embeddings }, dim: 1);

			// Expand per-image data in batch direction to be per-mask
			Tensor src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim: 0);

			src = src + dense_prompt_embeddings;
			Tensor pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim: 0);

			long b = src.shape[0];
			long c = src.shape[1];
			long h = src.shape[2];
			long w = src.shape[3];

			// Run the Transformer
			(Tensor hs, src) = this.transformer.forward(src, pos_src, tokens);

			Tensor iou_token_out = hs[.., 0, ..];
			Tensor mask_tokens_out = hs[.., 1..(1 + this.num_mask_tokens), ..];

			// Upscale mask embeddings and predict masks using the mask tokens
			src = src.transpose(1, 2).view(b, c, h, w);

			Tensor upscaled_embedding = this.output_upscaling.forward(src);
			List<Tensor> hyper_in_list = new List<Tensor>();
			for (int i = 0; i < this.num_mask_tokens; i++)
			{
				hyper_in_list.Add(this.output_hypernetworks_mlps[i].forward(mask_tokens_out[.., i, ..]));
			}
			Tensor hyper_in = torch.stack(hyper_in_list, dim: 1);

			b = upscaled_embedding.shape[0];
			c = upscaled_embedding.shape[1];
			h = upscaled_embedding.shape[2];
			w = upscaled_embedding.shape[3];
			Tensor masks = (hyper_in.matmul(upscaled_embedding.view(b, c, h * w))).view(b, -1, h, w);

			// Generate mask quality predictions
			Tensor iou_pred = this.iou_prediction_head.forward(iou_token_out);

			return (masks.MoveToOuterDisposeScope(), iou_pred.MoveToOuterDisposeScope());

		}

		private class MLP : Module<Tensor, Tensor>
		{
			private readonly int num_layers;
			private readonly bool sigmoid_output;
			private readonly ModuleList<Linear> layers;

			public MLP(int input_dim, int hidden_dim, int output_dim, int num_layers, bool sigmoid_output = false) : base(nameof(MLP))
			{
				this.num_layers = num_layers;
				int[] h = Enumerable.Repeat(hidden_dim, num_layers - 1).ToArray();
				List<int> input_dims = new List<int> { input_dim };
				input_dims.AddRange(h);
				List<int> output_dims = h.ToList();
				output_dims.Add(output_dim);

				layers = new ModuleList<Linear>();
				for (int i = 0; i < num_layers; i++)
				{
					layers.Add(Linear(input_dims[i], output_dims[i]));
				}
				this.sigmoid_output = sigmoid_output;

				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				for (int i = 0; i < num_layers; i++)
				{
					x = layers[i].forward(x);
					if (i < num_layers - 1)
					{
						x = functional.relu(x);
					}
				}

				if (sigmoid_output)
				{
					x = functional.sigmoid(x);
				}

				return x;
			}

			protected override void Dispose(bool disposing)
			{
				if (disposing)
				{
					layers.Dispose();
				}
				base.Dispose(disposing);
			}

		}
	}
}
