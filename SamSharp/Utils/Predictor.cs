using SamSharp.Modeling;
using SkiaSharp;
using System.Linq;
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


		public SamPredictor(string checkpointPath, SamDevice device = SamDevice.CPU)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);
			Device d = device is SamDevice.Cuda ? CUDA : new Device(DeviceType.CPU);
			model = BuildSam.BuildSamModel(checkpointPath, d);
		}

		public List<PredictOutput> Predict(SKBitmap image, List<SamPoint> points = null, List<SamBox> boxes = null, int maxImageSize = 1024)
		{
			using var _ = no_grad();
			model.eval();
			Tensor imgTensor = Tools.ImageTools.GetTensorFromImage(image);

			long w = imgTensor.shape[2];
			long h = imgTensor.shape[1];
			float scaleFactor = Math.Min((float)maxImageSize / w, (float)maxImageSize / h);
			int newW = (int)Math.Ceiling(w * scaleFactor);
			int newH = (int)Math.Ceiling(h * scaleFactor);
			imgTensor = torchvision.transforms.functional.resize(imgTensor, newH, newW);
			long[] original_size = new long[] { h, w };

			Tensor pointsTensor = null;
			Tensor labelsTensor = null;
			Tensor boxesTensor = null;

			if (points.Count > 0)
			{
				pointsTensor = torch.zeros(new long[] { 1, points?.Count ?? 0, 2 });
				labelsTensor = torch.zeros(new long[] { 1, points?.Count ?? 0 });
				for (int i = 0; i < (points?.Count ?? 0); i++)
				{
					SamPoint point = points[i];
					pointsTensor[0, i, 0] = point.X * scaleFactor;
					pointsTensor[0, i, 1] = point.Y * scaleFactor;
					labelsTensor[0, i] = point.Label ? 1 : 0;
				}
			}

			if (boxes.Count > 0)
			{
				boxesTensor = torch.zeros(new long[] { boxes.Count, 4 });
				for (int i = 0; i < boxes.Count!; i++)
				{
					SamBox box = boxes[i];
					boxesTensor[i, 0] = box.Left * scaleFactor;   // Box Left
					boxesTensor[i, 1] = box.Top * scaleFactor;    // Box Top
					boxesTensor[i, 2] = box.Right * scaleFactor;  // Box Right
					boxesTensor[i, 3] = box.Bottom * scaleFactor; // Box Bottom
				}
			}

			BatchedInput batchedInput = new BatchedInput
			{
				Image = imgTensor.unsqueeze(0),
				Point_coords = pointsTensor,
				Point_labels = labelsTensor,
				Original_size = original_size,
				Boxes = boxesTensor
			};
			batchedInput.to(CUDA);

			List<BatchedInput> inputs = new List<BatchedInput>
			{
				batchedInput
			};
			model.to(CUDA);

			List<BatchedOutput> outputs = model.forward(inputs, false);

			List<PredictOutput> predictOutputs = new List<PredictOutput>();

			foreach (BatchedOutput output in outputs)
			{
				for (int i = 0; i < output.Masks.shape[0]; i++)
				{
					bool[,] maskArray = new bool[output.Masks.shape[3], output.Masks.shape[2]];
					var data = output.Masks.transpose(2, 3)[i].data<bool>().ToArray();
					Buffer.BlockCopy(data, 0, maskArray, 0, data.Length * sizeof(bool));

					predictOutputs.Add(new PredictOutput
					{
						Mask = maskArray,
						Precision = output.Iou_predictions[i].ToSingle(),
					});
				}
			}
			return predictOutputs;
		}


	}
}
