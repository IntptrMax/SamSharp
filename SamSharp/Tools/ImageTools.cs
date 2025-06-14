using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace SamSharp.Tools
{
	internal class ImageTools
	{
		internal static Tensor GetTensorFromImage(SKBitmap skBitmap)
		{
			using (MemoryStream stream = new MemoryStream())
			{
				skBitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
				stream.Position = 0;
				Tensor tensor = torchvision.io.read_image(stream, torchvision.io.ImageReadMode.RGB);
				return tensor;
			}
		}

		internal static SKBitmap GetImageFromTensor(Tensor tensor)
		{
			using (MemoryStream memoryStream = new MemoryStream())
			{
				torchvision.io.write_png(tensor.cpu(), memoryStream);
				memoryStream.Position = 0;
				SKBitmap skBitmap = SKBitmap.Decode(memoryStream);
				return skBitmap;
			}
		}
	}
}
