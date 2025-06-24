using System;
using TorchSharp;
using static TorchSharp.torch;

namespace SamSharp.Utils
{
	public class Classes
	{
		/// <summary>
		/// A class representing a batched input to the SAM model.
		/// </summary>
		public class BatchedInput : IDisposable
		{
			/// <summary>
			/// Batched point prompts for this image, with shape BxNx2.Already transformed to the input frame of the model.
			/// </summary>
			public Tensor Point_coords { get; set; }

			/// <summary>
			/// Batched labels for point prompts, with shape BxN.
			/// </summary>
			public Tensor Point_labels { get; set; }

			/// <summary>
			/// Batched box inputs, with shape Bx4. Already transformed to the input frame of the model.
			/// </summary>
			public Tensor? Boxes { get; set; } = null;

			/// <summary>
			/// Batched mask inputs to the model, in the form Bx1xHxW.
			/// </summary>
			public Tensor? Mask_inputs { get; set; } = null;

			/// <summary>
			///	The original size of the image before transformation, as (H, W).
			///	</summary>
			public long[] Original_size { get; set; }
			public long[] Input_size { get; set; }

			public void to(Device? device, ScalarType? dtype)
			{
				Point_coords = Point_coords?.to(dtype ?? Point_coords.dtype, device ?? Point_coords.device);
				Point_labels = Point_labels?.to(dtype ?? Point_labels.dtype, device ?? Point_labels.device);
				Boxes = Boxes?.to(dtype ?? Boxes.dtype, device ?? Boxes.device);
				Mask_inputs = Mask_inputs?.to(dtype ?? Mask_inputs.dtype, device ?? Mask_inputs.device);
			}

			public void Dispose()
			{
				Point_coords?.Dispose();
				Point_labels?.Dispose();
				Boxes?.Dispose();
				Mask_inputs?.Dispose();
			}
		}

		/// <summary>
		/// A class representing the output of the SAM model when processing a batch of inputs.
		/// </summary>
		public class BatchedOutput
		{
			/// <summary>
			/// Batched binary mask predictions, with shape BxCxHxW, where B is the number of input prompts, C is determined by multimask_output, and(H, W) is the original size of the image.
			/// </summary>
			public Tensor Masks { get; set; }

			/// <summary>
			/// The model's predictions of mask quality, in shape BxC.
			/// </summary>
			public Tensor Iou_predictions { get; set; }

			/// <summary>
			/// Low resolution logits with shape BxCxHxW, where H = W = 256.Can be passed as mask input to subsequent iterations of prediction.
			/// </summary>
			public Tensor Low_res_logits { get; set; }

			public void to(Device? device = null, ScalarType? dtype = null)
			{
				Masks = Masks?.to(dtype ?? Masks.dtype, device ?? Masks.device);
				Iou_predictions = Iou_predictions?.to(dtype ?? Iou_predictions.dtype, device ?? Iou_predictions.device);
				Low_res_logits = Low_res_logits?.to(dtype ?? Low_res_logits.dtype, device ?? Low_res_logits.device);
			}

			public void Dispose()
			{
				Masks?.Dispose();
				Iou_predictions?.Dispose();
				Low_res_logits?.Dispose();
			}
		}

		public enum SamType
		{
			VitB,
			VitL,
			VitH,
			VitT,
		}

		public enum OutputMode
		{
			BinaryMask,
			UncompressedRle,
			CocoRle,
		}

		public class SamPoint
		{
			public SamPoint(int x, int y, bool label)
			{
				this.X = x;
				this.Y = y;
				this.Label = label;
			}

			/// <summary>
			/// The X coordinates of the point in pixels.
			/// </summary>
			public int X { get; set; }

			/// <summary>
			/// The Y coordinates of the point in pixels.
			/// </summary>
			public int Y { get; set; }
			/// <summary>
			/// True is for foreground, false for background.
			/// </summary>
			public bool Label { get; set; }
		}

		public class SamBox
		{
			public SamBox(int left, int top, int right, int bottom)
			{
				this.Left = left;
				this.Top = top;
				this.Right = right;
				this.Bottom = bottom;
			}
			public int Left { get; set; }
			public int Top { get; set; }
			public int Right { get; set; }
			public int Bottom { get; set; }
		}

		public class PredictOutput
		{
			public bool[,] Mask { get; set; }
			public float Precision { get; set; }
		}

		public enum SamDevice
		{
			CPU = 0,
			CUDA = 1,
		}

		public enum SamScalarType
		{
			Float16 = 5,
			Float32 = 6,
			BFloat16 = 15
		}

		//public class MaskData : IDisposable
		//{
		//	public Tensor IouPreds { get; set; }

		//	public Tensor Points { get; set; }

		//	public Tensor StabilityScore { get; set; }
		//	public Tensor Boxes { get; set; }
		//	public List<Rle> Rles { get; set; }

		//	public void Concat(MaskData data)
		//	{
		//		IouPreds = IouPreds is null ? data.IouPreds : torch.concat(new Tensor[] { IouPreds, data.IouPreds });
		//		Points = Points is null ? data.Points : torch.concat(new Tensor[] { Points, data.Points });
		//		StabilityScore = StabilityScore is null ? data.StabilityScore : torch.concat(new Tensor[] { StabilityScore, data.StabilityScore });
		//		Boxes = Boxes is null ? data.Boxes : torch.concat(new Tensor[] { Boxes, data.Boxes });
		//		if (Rles is null)
		//		{
		//			Rles = data.Rles;
		//		}
		//		else
		//		{
		//			Rles.AddRange(data.Rles);
		//		}
		//	}

		//	public void Dispose()
		//	{
		//		IouPreds?.Dispose();
		//		Points?.Dispose();
		//		StabilityScore?.Dispose();
		//		Boxes?.Dispose();
		//		Rles.Clear();
		//	}

		//	public void Filter(Tensor index)
		//	{
		//		IouPreds = IouPreds[torch.as_tensor(index, device: IouPreds.device)];
		//		Points = Points[torch.as_tensor(index, device: Points.device)];
		//		StabilityScore = StabilityScore[torch.as_tensor(index, device: StabilityScore.device)];
		//		Boxes = Boxes[torch.as_tensor(index, device: Boxes.device)];
		//		long[] indexs = index.data<long>().ToArray();
		//		Rles = indexs.Where(i => i >= 0 && i < Rles.Count).Select(i => Rles[(int)i]).ToList();
		//	}
		//}

		public class MaskData : IDisposable
		{
			public Tensor IouPreds { get; set; }

			public Tensor Points { get; set; }

			public Tensor StabilityScore { get; set; }
			public Tensor Boxes { get; set; }
			public Tensor Masks { get; set; }

			public List<Rle> Rles { get; set; }

			public void Concat(MaskData data)
			{
				IouPreds = IouPreds is null ? data.IouPreds : torch.concat(new Tensor[] { IouPreds, data.IouPreds });
				Points = Points is null ? data.Points : torch.concat(new Tensor[] { Points, data.Points });
				StabilityScore = StabilityScore is null ? data.StabilityScore : torch.concat(new Tensor[] { StabilityScore, data.StabilityScore });
				Boxes = Boxes is null ? data.Boxes : torch.concat(new Tensor[] { Boxes, data.Boxes });
				Masks = Masks is null ? data.Masks : torch.concat(new Tensor[] { Masks, data.Masks });
				if (Rles is null)
				{
					Rles = data.Rles;
				}
				else
				{
					Rles.AddRange(data.Rles);
				}
			}

			public void Dispose()
			{
				IouPreds?.Dispose();
				Points?.Dispose();
				StabilityScore?.Dispose();
				Boxes?.Dispose();
				Masks?.Dispose();
				Rles.Clear();
			}

			public void Filter(Tensor index)
			{
				long[] indexes = index.data<long>().ToArray();
				List<Rle> newRles = new List<Rle>();
				foreach (long item in indexes)
				{
					if (Rles is not null)
					{
						newRles.Add(Rles[(int)item]);
					}
				}
				Rles = newRles;

				IouPreds = IouPreds[torch.as_tensor(index, device: IouPreds.device)];
				Points = Points[torch.as_tensor(index, device: Points.device)];
				StabilityScore = StabilityScore[torch.as_tensor(index, device: StabilityScore.device)];
				Boxes = Boxes[torch.as_tensor(index, device: Boxes.device)];
				if (Masks is not null)
				{
					Masks = Masks[torch.as_tensor(index, device: Masks.device)];
				}
			}
		}

		public class Rle
		{
			public int[] Size { get; set; }
			public List<long> Counts { get; set; }
		}

	}
}
