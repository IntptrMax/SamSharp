using TorchSharp;
using static SamSharp.Utils.Classes;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SamSharp.Modeling
{
	internal class Sam : Module<List<BatchedInput>, bool, List<BatchedOutput>>
	{
		private readonly float mask_threshold = 0.0f;
		//private readonly string image_format = "RGB";

		public readonly ImageEncoderViT image_encoder;
		public readonly PromptEncoder prompt_encoder;
		public readonly MaskDecoder mask_decoder;
		private readonly float[] pixel_mean = new float[] { 123.675f, 116.28f, 103.53f };
		private readonly float[] pixel_std = new float[] { 58.395f, 57.12f, 57.375f };

		/// <summary>
		/// SAM predicts object masks from an image and input prompts.
		/// </summary>
		/// <param name="image_encoder">The backbone used to encode the image into image embeddings that allow for efficient mask prediction.</param>
		/// <param name="prompt_encoder">Encodes various types of input prompts.</param>
		/// <param name="mask_decoder">Predicts masks from the image embeddings and encoded prompts.</param>
		/// <param name="pixel_mean">Mean values for normalizing pixels in the input image.</param>
		/// <param name="pixel_std">Std values for normalizing pixels in the input image.</param>
		public Sam(ImageEncoderViT image_encoder, PromptEncoder prompt_encoder, MaskDecoder mask_decoder, float[] pixel_mean, float[] pixel_std) : base(nameof(Sam))
		{
			this.image_encoder = image_encoder;
			this.prompt_encoder = prompt_encoder;
			this.mask_decoder = mask_decoder;
			this.pixel_mean = pixel_mean;
			this.pixel_std = pixel_std;
			RegisterComponents();
		}

		/// <summary>
		/// Predicts masks end-to-end from provided images and prompts. If prompts are not known in advance, using SamPredictor is recommended over calling the model directly.
		/// </summary>
		/// <param name="batched_input">Batched inputs for SAM</param>
		/// <param name="multimask_output">Whether the model should predict multiple disambiguating masks, or return a single mask.</param>
		/// <returns></returns>
		public override List<BatchedOutput> forward(List<BatchedInput> batched_input, bool multimask_output)
		{
			using var _ = NewDisposeScope();
			List<Tensor> preprocessedImages = batched_input.Select(x => this.preprocess(x.Image)).ToList();
			Tensor input_images = torch.cat(preprocessedImages, dim: 0);
			Tensor image_embeddings = this.image_encoder.forward(input_images);

			List<BatchedOutput> outputs = new List<BatchedOutput>();
			for (int i = 0; i < batched_input.Count; i++)
			{
				BatchedInput image_record = batched_input[i];
				Tensor curr_embedding = image_embeddings[i];
				(Tensor, Tensor)? points = image_record.Point_coords is not null ? (image_record.Point_coords, image_record.Point_labels) : null;
				(Tensor sparse_embeddings, Tensor dense_embeddings) = this.prompt_encoder.forward(points: points, boxes: image_record.Boxes, masks: image_record.Mask_inputs);
				(Tensor low_res_masks, Tensor iou_predictions) = this.mask_decoder.forward(image_embeddings = curr_embedding.unsqueeze(0), image_pe: this.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings: sparse_embeddings, dense_prompt_embeddings: dense_embeddings, multimask_output: multimask_output);
				Tensor masks = this.postprocess_masks(low_res_masks, input_size: new long[] { image_record.Image.shape[2], image_record.Image.shape[3] }, original_size: image_record.Original_size);
				masks = masks > this.mask_threshold;

				outputs.Add(new BatchedOutput
				{
					Masks = masks.MoveToOuterDisposeScope(),
					Iou_predictions = iou_predictions.MoveToOuterDisposeScope(),
					Low_res_logits = low_res_masks.MoveToOuterDisposeScope()
				});
			}
			return outputs;
		}

		/// <summary>
		/// Remove padding and upscale masks to the original image size.
		/// </summary>
		/// <param name="masks"> Batched masks from the mask_decoder, in BxCxHxW format.</param>
		/// <param name="input_size">The size of the image input to the model, in (H, W) format.Used to remove padding.</param>
		/// <param name="original_size">The original size of the image before resizing for input to the model, in (H, W) format.</param>
		/// <returns>Batched masks in BxCxHxW format, where (H, W) is given by original_size.</returns>
		public Tensor postprocess_masks(Tensor masks, long[] input_size, long[] original_size)
		{
			masks = torch.nn.functional.interpolate(masks, new long[] { this.image_encoder.img_size, this.image_encoder.img_size }, mode: InterpolationMode.Bilinear, align_corners: false);
			masks = masks[TensorIndex.Ellipsis, ..(int)input_size[0], ..(int)input_size[1]];
			masks = torch.nn.functional.interpolate(masks, original_size, mode: InterpolationMode.Bilinear, align_corners: false);
			return masks;
		}


		/// <summary>
		/// Normalize pixel values and pad to a square input.
		/// </summary>
		public Tensor preprocess(Tensor x)
		{
			Tensor pixel_mean_tensor = tensor(this.pixel_mean).unsqueeze(-1).unsqueeze(-1).to(x.dtype,x.device);
			Tensor pixel_std_tensor = tensor(this.pixel_std).unsqueeze(-1).unsqueeze(-1).to(x.dtype, x.device);
			// Normalize colors
			x = (x - pixel_std_tensor) / pixel_std_tensor;
			// Pad
			long h = x.shape[2];
			long w = x.shape[3];

			long padh = this.image_encoder.img_size - h;
			long padw = this.image_encoder.img_size - w;
			x = torch.nn.functional.pad(x, new long[] { 0, padw, 0, padh }).to(x.dtype, x.device);
			return x;
		}

	}
}
