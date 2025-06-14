using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SamSharp.Modeling
{
	internal class PromptEncoder : Module<(Tensor, Tensor)?, Tensor, Tensor, (Tensor, Tensor)>
	{
		private readonly int embed_dim;
		private readonly (int, int) image_embedding_size;
		private readonly (int, int) input_image_size;
		private readonly int mask_in_chans;
		private readonly PositionEmbeddingRandom pe_layer;
		private readonly int num_point_embeddings;
		private readonly ModuleList<Embedding> point_embeddings;
		private readonly Embedding not_a_point_embed;
		private readonly (int, int) mask_input_size;
		private readonly Sequential mask_downscaling;
		private readonly Embedding no_mask_embed;

		/// <summary>
		/// Encodes prompts for input to SAM's mask decoder.
		/// </summary>
		/// <param name="embed_dim">The prompts' embedding dimension</param>
		/// <param name="image_embedding_size">The spatial size of the image embedding, as (H, W).</param>
		/// <param name="input_image_size">The padded size of the image as input to the image encoder, as (H, W).</param>
		/// <param name="mask_in_chans">The number of hidden channels used for encoding input masks.</param>
		public PromptEncoder(int embed_dim, (int, int) image_embedding_size, (int, int) input_image_size, int mask_in_chans) : base(nameof(PromptEncoder))
		{
			this.embed_dim = embed_dim;
			this.image_embedding_size = image_embedding_size;
			this.input_image_size = input_image_size;
			this.mask_in_chans = mask_in_chans;
			this.pe_layer = new PositionEmbeddingRandom(embed_dim / 2, 0.1f);

			this.num_point_embeddings = 4;
			point_embeddings = new ModuleList<Embedding>();
			for (int i = 0; i < num_point_embeddings; i++)
			{
				point_embeddings.append(Embedding(1, embed_dim));
			}

			this.not_a_point_embed = nn.Embedding(1, embed_dim);
			this.mask_input_size = (4 * image_embedding_size.Item1, 4 * image_embedding_size.Item2);

			this.mask_downscaling = nn.Sequential(
				   nn.Conv2d(1, mask_in_chans / 4, kernel_size: 2, stride: 2),
				   new Common.LayerNorm2d(mask_in_chans / 4),
				   GELU(),
				   nn.Conv2d(mask_in_chans / 4, mask_in_chans, kernel_size: 2, stride: 2),
				   new Common.LayerNorm2d(mask_in_chans),
				   GELU(),
				   nn.Conv2d(mask_in_chans, embed_dim, kernel_size: 1));

			this.no_mask_embed = nn.Embedding(1, embed_dim);

			RegisterComponents();
		}

		/// <summary>
		/// Returns the positional encoding used to encode point prompts,
		/// applied to a dense set of points the shape of the image encoding.
		/// </summary>
		/// <returns>Positional encoding with shape 1x(embed_dim)X(embedding_h)X(embedding_w)</returns>
		public Tensor get_dense_pe()
		{
			return this.pe_layer.forward(this.image_embedding_size).unsqueeze(0);
		}

		private Tensor _embed_points(Tensor points, Tensor labels, bool pad)
		{
			points = points + 0.5f;  // Shift to center of pixel
			if (pad)
			{
				Tensor padding_point = torch.zeros(new long[] { points.shape[0], 1, 2 }, device: points.device, dtype: points.dtype);
				Tensor padding_label = -torch.ones(new long[] { labels.shape[0], 1 }, device: labels.device, dtype: labels.dtype);
				points = torch.cat(new Tensor[] { points, padding_point }, dim: 1);
				labels = torch.cat(new Tensor[] { labels, padding_label }, dim: 1);
			}
			Tensor point_embedding = this.pe_layer.forward_with_coords(points, this.input_image_size);
			point_embedding[labels == -1] = 0.0f;
			point_embedding[labels == -1] += this.not_a_point_embed.weight!;
			point_embedding[labels == 0] += this.point_embeddings[0].weight!;
			point_embedding[labels == 1] += this.point_embeddings[1].weight!;
			return point_embedding;
		}

		/// <summary>
		/// Embeds box prompts.
		/// </summary>
		/// <param name="boxes"></param>
		/// <returns></returns>
		private Tensor _embed_boxes(Tensor boxes)
		{
			boxes = boxes + 0.5f;  // Shift to center of pixel
			Tensor coords = boxes.reshape(-1, 2, 2);
			Tensor corner_embedding = this.pe_layer.forward_with_coords(coords, this.input_image_size);
			corner_embedding[.., 0, ..] += this.point_embeddings[2].weight!;
			corner_embedding[.., 1, ..] += this.point_embeddings[3].weight!;
			return corner_embedding;
		}

		/// <summary>
		/// Embeds mask inputs.
		/// </summary>
		/// <param name="masks"></param>
		/// <returns></returns>
		private Tensor _embed_masks(Tensor masks)
		{
			return this.mask_downscaling.forward(masks);
		}

		/// <summary>
		/// Gets the batch size of the output given the batch size of the input prompts.
		/// </summary>
		/// <param name="points"></param>
		/// <param name="boxes"></param>
		/// <param name="masks"></param>
		/// <returns></returns>
		private long _get_batch_size((Tensor, Tensor)? points, Tensor boxes, Tensor masks)
		{
			if (points is not null)
			{
				return points.Value.Item1.shape[0];
			}
			else if (boxes is not null)
			{
				return boxes.shape[0];
			}
			else if (masks is not null)
			{
				return masks.shape[0];
			}
			else
			{
				return 1;
			}
		}

		private Device _get_device()
		{
			return this.point_embeddings[0].weight!.device;
		}

		/// <summary>
		/// Embeds different types of prompts, returning both sparse and dense embeddings.
		/// </summary>
		/// <param name="points">point coordinates and labels to embed.</param>
		/// <param name="boxes">boxes to embed</param>
		/// <param name="masks">masks to embed</param>
		/// <returns>sparse embeddings for the points and boxes, with shape BxNx(embed_dim), where N is determined by the number of input points and boxes.<br/>
		/// dense embeddings for the masks, in the shape Bx(embed_dim)X(embed_H)X(embed_W)</returns>
		public override (Tensor, Tensor) forward((Tensor, Tensor)? points, Tensor boxes, Tensor masks)
		{
			using var _ = NewDisposeScope();
			long bs = this._get_batch_size(points, boxes, masks);
			Tensor sparse_embeddings = torch.empty(new long[] { bs, 0, this.embed_dim }, device: this._get_device());

			if (points is not null)
			{
				Tensor coords = points.Value.Item1;
				Tensor labels = points.Value.Item2;
				Tensor point_embeddings = this._embed_points(coords, labels, pad: (boxes is null));
				sparse_embeddings = torch.cat(new Tensor[] { sparse_embeddings, point_embeddings }, dim: 1);
			}

			if (boxes is not null)
			{
				Tensor box_embeddings = this._embed_boxes(boxes);
				sparse_embeddings = torch.cat(new Tensor[] { sparse_embeddings, box_embeddings }, dim: 1);
			}

			Tensor dense_embeddings = (masks is not null) ? this._embed_masks(masks) : this.no_mask_embed.weight!.reshape(1, -1, 1, 1).expand(bs, -1, this.image_embedding_size.Item1, this.image_embedding_size.Item2);
			return (sparse_embeddings.MoveToOuterDisposeScope(), dense_embeddings.MoveToOuterDisposeScope());
		}


		private class PositionEmbeddingRandom : Module<(int, int), Tensor>
		{
			private readonly Tensor positional_encoding_gaussian_matrix;
			public PositionEmbeddingRandom(int num_pos_feats = 64, float scale = 0) : base(nameof(PositionEmbeddingRandom))
			{
				scale = scale <= 0 ? 1.0f : scale;
				positional_encoding_gaussian_matrix = scale * torch.randn(new long[] { 2, num_pos_feats });
				RegisterComponents();
			}

			//Positionally encode points that are normalized to [0,1].
			private Tensor _pe_encoding(Tensor coords)
			{
				// assuming coords are in [0, 1]^2 square and have d_1 X ... X d_n X 2 shape
				coords = 2 * coords - 1;
				coords = coords.matmul(this.positional_encoding_gaussian_matrix);
				coords = 2 * Math.PI * coords;
				// outputs d_1 X ... X d_n X C shape
				return torch.cat(new Tensor[] { torch.sin(coords), torch.cos(coords) }, dim: -1);
			}

			// Generate positional encoding for a grid of the specified size.
			public override Tensor forward((int, int) size)
			{
				using var _ = NewDisposeScope();
				(int h, int w) = size;
				Device device = this.positional_encoding_gaussian_matrix.device;
				ScalarType dtype = this.positional_encoding_gaussian_matrix.dtype;
				Tensor grid = torch.ones(new long[] { h, w }, device: device, dtype: dtype);
				Tensor y_embed = grid.cumsum(dim: 0) - 0.5;
				Tensor x_embed = grid.cumsum(dim: 1) - 0.5;
				y_embed = y_embed / h;
				x_embed = x_embed / w;

				Tensor pe = this._pe_encoding(torch.stack(new Tensor[] { x_embed, y_embed }, dim: -1));

				return pe.permute(2, 0, 1).MoveToOuterDisposeScope();  // C X H X W
			}

			// Positionally encode points that are not normalized to [0,1].
			public Tensor forward_with_coords(Tensor coords_input, (int, int) image_size)
			{
				Tensor coords = coords_input.clone();
				coords[.., .., 0] = coords[.., .., 0] / image_size.Item2;
				coords[.., .., 1] = coords[.., .., 1] / image_size.Item1;
				//return this._pe_encoding(coords.to(torch.float32)); // B X N X C
				return this._pe_encoding(coords); // B X N X C
			}
		}
	}
}
