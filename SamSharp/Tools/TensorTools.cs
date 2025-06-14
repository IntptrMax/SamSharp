using System.IO.Compression;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace SamSharp.Tools
{
	internal class TensorTools
	{
		/// <summary>
		/// Load Python .pt tensor file
		/// </summary>
		/// <param name="path">tensor path</param>
		/// <returns>Tensor in TorchSharp</returns>
		public static Tensor LoadTensorFromPT(string path)
		{
			torch.ScalarType dtype = torch.ScalarType.Float32;
			List<long> shape = new List<long>();
			ZipArchive zip = ZipFile.OpenRead(path);
			ZipArchiveEntry headerEntry = zip.Entries.First(e => e.Name == "data.pkl");

			// Header is always small enough to fit in memory, so we can read it all at once
			using Stream headerStream = headerEntry.Open();
			byte[] headerBytes = new byte[headerEntry.Length];
			headerStream.Read(headerBytes, 0, headerBytes.Length);

			string headerStr = Encoding.Default.GetString(headerBytes);
			if (headerStr.Contains("HalfStorage"))
			{
				dtype = torch.ScalarType.Float16;
			}
			else if (headerStr.Contains("BFloat"))
			{
				dtype = torch.ScalarType.Float16;
			}
			else if (headerStr.Contains("FloatStorage"))
			{
				dtype = torch.ScalarType.Float32;
			}
			for (int i = 0; i < headerBytes.Length; i++)
			{
				if (headerBytes[i] == 81 && headerBytes[i + 1] == 75 && headerBytes[i + 2] == 0)
				{
					for (int j = i + 2; j < headerBytes.Length; j++)
					{
						if (headerBytes[j] == 75)
						{
							shape.Add(headerBytes[j + 1]);
							j++;
						}
						else if (headerBytes[j] == 77)
						{
							shape.Add(headerBytes[j + 1] + headerBytes[j + 2] * 256);
							j += 2;
						}
						else if (headerBytes[j] == 113)
						{
							break;
						}

					}
					break;
				}
			}

			Tensor tensor = torch.zeros(shape.ToArray(), dtype: dtype);
			ZipArchiveEntry dataEntry = zip.Entries.First(e => e.Name == "0");

			using Stream dataStream = dataEntry.Open();
			byte[] data = new byte[dataEntry.Length];
			dataStream.Read(data, 0, data.Length);
			tensor.bytes = data;
			return tensor;
		}

		/// <summary>
		/// Load Python .pt tensor file and change dtype and device the same as given tensor.
		/// </summary>
		/// <param name="path">tensor path</param>
		/// <param name="tensor">the given tensor</param>
		/// <returns>Tensor in TorchSharp</returns>
		public static Tensor LoadTensorFromPT(string path, Tensor tensor)
		{
			return LoadTensorFromPT(path).to(tensor.dtype, tensor.device);
		}

	}
}
