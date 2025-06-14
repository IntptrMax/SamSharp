using static TorchSharp.torch;

namespace SamSharp.Utils
{
	public class Classes
	{
		public class BatchedInput
		{
			public Tensor Image { get; set; }
			public Tensor Point_coords { get; set; }
			public Tensor Point_labels { get; set; }
			public Tensor? Boxes { get; set; } = null;
			public Tensor? Mask_inputs { get; set; } = null;
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

		public class BatchedOutput
		{
			public Tensor Masks { get; set; }
			public Tensor Iou_predictions { get; set; }
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


			public int X { get; set; }
			public int Y { get; set; }
			/// <summary>
			/// True is for foreground, false for background.
			/// </summary>
			public bool Label { get; set; }
		}

		public enum SamDevice
		{
			Cuda = 0,
			CPU,
		}
	}
}
