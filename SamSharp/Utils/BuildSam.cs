using SamSharp.Modeling;
using SamSharp.Tools;
using TorchSharp;
using static SamSharp.Modeling.Transformer;
using static TorchSharp.torch;

namespace SamSharp.Utils
{
	internal class BuildSam
	{
		internal static Sam BuildSamModel(string checkpoint, Device device, ScalarType dtype)
		{
			List<CommonTensor> commonTensors = PickleLoader.ReadTensorsInfoFromFile(checkpoint);
			return commonTensors.Count switch
			{
				314 => build_sam_vit_b(checkpoint, device, dtype),
				594 => build_sam_vit_h(checkpoint, device, dtype),
				482 => build_sam_vit_l(checkpoint, device, dtype),
				_ => throw new ArgumentException("Invalid SAM type specified.")
			};
		}

		private static Sam build_sam_vit_h(string checkpoint, Device device, ScalarType dtype)
		{
			return _build_sam(
				encoder_embed_dim: 1280,
				encoder_depth: 32,
				encoder_num_heads: 16,
				encoder_global_attn_indexes: new int[] { 7, 15, 23, 31 },
				checkpoint: checkpoint,
				device: device,
				dtype:dtype);
		}

		private static Sam build_sam_vit_l(string checkpoint, Device device, ScalarType dtype)
		{
			return _build_sam(
				encoder_embed_dim: 1024,
				encoder_depth: 24,
				encoder_num_heads: 16,
				encoder_global_attn_indexes: new int[] { 5, 11, 17, 23 },
				checkpoint: checkpoint,
				device: device,
				dtype: dtype);
		}


		private static Sam build_sam_vit_b(string checkpoint, Device device, ScalarType dtype)
		{
			return _build_sam(
				encoder_embed_dim: 768,
				encoder_depth: 12,
				encoder_num_heads: 12,
				encoder_global_attn_indexes: new int[] { 2, 5, 8, 11 },
				checkpoint: checkpoint,
				device: device,
				dtype: dtype);
		}

		private static Sam _build_sam(int encoder_embed_dim, int encoder_depth, int encoder_num_heads, int[] encoder_global_attn_indexes, string? checkpoint = null, Device? device = null, ScalarType dtype = ScalarType.Float32)
		{
			int prompt_embed_dim = 256;
			int image_size = 1024;
			int vit_patch_size = 16;
			int image_embedding_size = image_size / vit_patch_size;

			device = device ?? CPU;

			ImageEncoderViT imageEncoder = new ImageEncoderViT(depth: encoder_depth,
				embed_dim: encoder_embed_dim,
				img_size: image_size,
				mlp_ratio: 4,
				num_heads: encoder_num_heads,
				patch_size: vit_patch_size,
				qkv_bias: true,
				use_rel_pos: true,
				global_attn_indexes: encoder_global_attn_indexes,
				window_size: 14,
				out_chans: prompt_embed_dim).to(device, dtype);

			PromptEncoder promptEncoder = new PromptEncoder(
				embed_dim: prompt_embed_dim,
				image_embedding_size: (image_embedding_size, image_embedding_size),
				input_image_size: (image_size, image_size),
				mask_in_chans: 16).to(device, dtype);

			MaskDecoder maskDecoder = new MaskDecoder(
				num_multimask_outputs: 3,
				transformer: new TwoWayTransformer(
					depth: 2,
					embedding_dim: prompt_embed_dim,
					mlp_dim: 2048,
					num_heads: 8),
				transformer_dim: prompt_embed_dim,
				iou_head_depth: 3,
				iou_head_hidden_dim: 256).to(device, dtype);


			Sam sam = new Sam(
			image_encoder: imageEncoder,
			prompt_encoder: promptEncoder,
			mask_decoder: maskDecoder,
			pixel_mean: new float[] { 123.675f, 116.28f, 103.53f },
			pixel_std: new float[] { 58.395f, 57.12f, 57.375f });

			if (!string.IsNullOrEmpty(checkpoint))
			{
				Dictionary<string, Tensor> state_dict = PickleLoader.Load(checkpoint);
				(var error, var missing) = sam.load_state_dict(state_dict, strict: false);
				if (error.Count + missing.Count > 0)
				{
					throw new ArgumentException("Error loading state dict");
				}
			}
			return sam.to(device,dtype);
		}
	}
}
