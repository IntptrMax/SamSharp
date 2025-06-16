using SkiaSharp;
using static TorchSharp.torch;

namespace SamSharp.Utils
{
	public class Classes
	{
		/// <summary>
		/// A class representing a batched input to the SAM model.
		/// </summary>
		public class BatchedInput
		{
			/// <summary>
			/// The image as a torch tensor in 3xHxW format, already transformed for input to the model.
			/// </summary>
			public Tensor Image { get; set; }

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

			public void to(Device? device = null)
			{
				if (device is not null)
				{
					Image = Image?.to(device);
					Point_coords = Point_coords?.to(device);
					Point_labels = Point_labels?.to(device);
					Boxes = Boxes?.to(device);
					Mask_inputs = Mask_inputs?.to(device);
				}
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
		}

		public enum SamType
		{
			VitB,
			VitL,
			VitH
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
			Cuda = 0,
			CPU,
		}
	}
}
