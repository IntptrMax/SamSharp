using SkiaSharp;
using static SamSharp.Utils.Classes;

namespace SamSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string checkpointPath = @"..\..\..\Assets\Weights\MobileSam.pt";
			SamDevice device = SamDevice.CUDA; // or SamDevice.Cpu if you want to run on CPU
			SamScalarType dtype = SamScalarType.Float32; // Maybe SamScalarType.Float16 or SamScalarType.BFloat16 will get None result.
			int imageSize = 512; // The maximum size of the image to process, can be adjusted based on your GPU memory

			SKBitmap image = SKBitmap.Decode(@"..\..\..\Assets\Images\truck.jpg");

			//// Use Automatic Mask Generator
			//SamSharp.Utils.SamAutomaticMaskGenerator generator = new SamSharp.Utils.SamAutomaticMaskGenerator(checkpointPath, device: device, dtype: dtype);
			//List<PredictOutput> outputs = generator.generate(image, maxImageSize: imageSize);

			// Use predictor
			SamSharp.Utils.SamPredictor predictor = new SamSharp.Utils.SamPredictor(checkpointPath, device, dtype);
			List<SamPoint> points = new List<SamPoint>
			{
				new SamPoint(500, 375, false),
				new SamPoint(1524,675, false),
			};
			List<SamBox> boxes = new List<SamBox>
			{
				new SamBox(75, 275, 1725, 850),
				new SamBox(425, 600, 700, 875),
				new SamBox(1375, 550, 1650, 800),
				new SamBox(1240, 675, 1400, 750),
			};

			predictor.SetImage(image);
			List<PredictOutput> outputs = predictor.Predict(null, boxes);

			Console.WriteLine("The predictions are :");

			using (SKCanvas canvas = new SKCanvas(image))
			{
				canvas.Clear(SKColors.Transparent);
				var random = new Random();

				for (int i = 0; i < outputs.Count; i++)
				{
					PredictOutput output = outputs[i];
					Console.WriteLine($"Mask {i}: Precision: {output.Precision * 100:F2}%");
					bool[,] mask = output.Mask;

					SKColor color = new SKColor((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256));
					using (var paint = new SKPaint { Color = color, BlendMode = SKBlendMode.Src })
					{
						int width = mask.GetLength(0);
						int height = mask.GetLength(1);

						using (var path = new SKPath())
						{
							for (int y = 0; y < height; y++)
							{
								for (int x = 0; x < width; x++)
								{
									if (mask[x, y])
									{
										path.AddRect(new SKRect(x, y, x + 1, y + 1));
									}
								}
							}
							canvas.DrawPath(path, paint);
						}
					}
				}
			}
			SKData data = image.Encode(SKEncodedImageFormat.Png, 100);
			using (var stream = File.OpenWrite($"mask.png"))
			{
				data.SaveTo(stream);
			}

		}
	}
}
