using SamSharp.Modeling;
using SkiaSharp;
using TorchSharp;
using static SamSharp.Utils.Classes;
using static TorchSharp.torch;

namespace SamSharp.Utils
{

	/// <summary>
	///  Predict masks for the given input prompts, using the currently set image. Input prompts are batched torch tensors and are expected to already be transformed to the input frame using ResizeLongestSide.
	/// </summary>
	/// <param name="point_coords">A BxNx2 array of point prompts to the model.Each point is in (X, Y) in pixels.</param>
	/// <param name="point_labels">A BxN array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point.</param>
	/// <param name="boxes">A Bx4 array given a box prompt to the model, in XYXY format.</param>
	/// <param name="mask_input">A low resolution mask input to the model, typically coming from a previous prediction iteration.Has form Bx1xHxW, where for SAM, H= W = 256.Masks returned by a previous iteration of the predict method do not need further transformation.</param>
	/// <param name="multimask_output">If true, the model will return three masks. For ambiguous input prompts(such as a single click), this will often produce better masks than a single prediction.If only a single mask is needed, the model's predicted quality score can be used to select the best mask.For non-ambiguous prompts, such as multiple input prompts, multimask_output=False can give better results.</param>
	/// <param name="return_logits">If true, returns un-thresholded masks logits instead of a binary mask.</param>
	/// <returns>(torch.Tensor): The output masks in BxCxHxW format, where C is the number of masks, and(H, W) is the original image size.<br/>
	/// (torch.Tensor): An array of shape BxC containing the model's predictions for the quality of each mask.<br/>
	/// (torch.Tensor): An array of shape BxCxHxW, where C is the number of masks and H = W = 256.These low res logits can be passed to	a subsequent iteration as mask input.</returns>

	public class SamPredictor
	{
		private readonly Sam model;
		private const float maxSize = 1024;

		public SamPredictor(string checkpointPath, SamDevice device = SamDevice.CPU)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);
			Device d = device is SamDevice.Cuda ? CUDA : new Device(DeviceType.CPU);
			model = BuildSam.BuildSamModel(checkpointPath, d);
		}

		public (List<SKBitmap>, List<float[]>) Predict(SKBitmap image, List<SamPoint> points)
		{
			using var _ = no_grad();
			model.eval();
			Tensor imgTensor = Tools.ImageTools.GetTensorFromImage(image);

			long w = imgTensor.shape[2];
			long h = imgTensor.shape[1];
			float scaleFactor = Math.Min(maxSize / w, maxSize / h);
			int newW = (int)Math.Ceiling(w * scaleFactor);
			int newH = (int)Math.Ceiling(h * scaleFactor);
			imgTensor = torchvision.transforms.functional.resize(imgTensor, newH, newW);
			long[] original_size = new long[] { h, w };

			Tensor pointsTensor = torch.zeros(new long[] { 1, points.Count, 2 });
			Tensor labelsTensor = torch.zeros(new long[] { 1, points.Count });

			for (int i = 0; i < points.Count; i++)
			{
				SamPoint point = points[i];
				pointsTensor[0, i, 0] = point.X * scaleFactor;
				pointsTensor[0, i, 1] = point.Y * scaleFactor;
				labelsTensor[0, i] = point.Label ? 1 : 0;
			}

			BatchedInput batchedInput = new BatchedInput
			{
				Image = imgTensor.unsqueeze(0),
				Point_coords = pointsTensor,
				Point_labels = labelsTensor,
				Original_size = original_size
			};
			batchedInput.to(CUDA);

			List<BatchedInput> inputs = new List<BatchedInput>
			{
				batchedInput
			};
			model.to(CUDA);

			List<BatchedOutput> outputs = model.forward(inputs, true);
			List<SKBitmap> outputBitmaps = new List<SKBitmap>();
			List<float[]> iou_predictions = new List<float[]>();
			foreach (BatchedOutput output in outputs)
			{
				Tensor msk = (output.Masks.@byte() * 255).clip(0, 255).cpu();
				SKBitmap bitmap = Tools.ImageTools.GetImageFromTensor(msk);
				outputBitmaps.Add(bitmap);
				iou_predictions.Add(output.Iou_predictions.data<float>().ToArray());
			}
			return (outputBitmaps,iou_predictions);
		}


	}
}
