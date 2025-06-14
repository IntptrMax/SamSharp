using SkiaSharp;
using static SamSharp.Utils.Classes;

namespace SamSharp_WinFormDemo
{
	public partial class MainForm : Form
	{
		private SKBitmap image;
		private Image pictureBoxImage;
		private float scale = 1.0f;
		private int padX = 0;
		private int padY = 0;
		private SamSharp.Utils.SamPredictor predictor;
		private string modelPath;


		private List<SamPoint> samPoints = new List<SamPoint>();

		public MainForm()
		{
			InitializeComponent();
		}

		private void Button_ImageLoad_Click(object sender, EventArgs e)
		{
			OpenFileDialog openFileDialog = new OpenFileDialog
			{
				Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;",
				Title = "Select an Image"
			};
			if (openFileDialog.ShowDialog() == DialogResult.OK)
			{
				try
				{
					string filePath = openFileDialog.FileName;

					PictureBox_Image.Image = Image.FromFile(filePath);
					pictureBoxImage = Image.FromFile(filePath);
					image = SKBitmap.Decode(filePath);
					int imageWidth = image.Width;
					int imageHeight = image.Height;
					int pictureBoxWidth = PictureBox_Image.Width;
					int pictureBoxHeight = PictureBox_Image.Height;
					scale = Math.Min((float)pictureBoxWidth / imageWidth, (float)pictureBoxHeight / imageHeight);
					padX = (pictureBoxWidth - (int)(imageWidth * scale)) / 2;
					padY = (pictureBoxHeight - (int)(imageHeight * scale)) / 2;
					samPoints = new List<SamPoint>();
					DrawPoints();
				}
				catch (Exception ex)
				{
					MessageBox.Show($"Error loading image: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
			}
		}

		private void DrawPoints()
		{
			Image img = pictureBoxImage.Clone() as Image;
			Graphics graphics = Graphics.FromImage(img);
			samPoints.ForEach(point =>
			{
				graphics.FillEllipse(point.Label ? Brushes.Green : Brushes.Blue, point.X - 5, point.Y - 5, 10, 10);
			});
			//graphics.Save();
			graphics.Dispose();
			PictureBox_Image.Image = img;
		}

		private void PictureBox_Image_MouseClick(object sender, MouseEventArgs e)
		{
			if (e.Location.X < padX || e.Location.X > PictureBox_Image.Width - padX || e.Location.Y < padY || e.Location.Y > PictureBox_Image.Height - padY)
			{
				MessageBox.Show("Click is outside the image bounds.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				return;
			}

			int adjustedX = (int)((e.Location.X - padX) / scale);
			int adjustedY = (int)((e.Location.Y - padY) / scale);
			SamPoint point = new SamPoint(adjustedX, adjustedY, RadioButton_Foreground.Checked); // Assuming true for foreground
			samPoints.Add(point);
			DrawPoints();
		}

		private void Button_Run_Click(object sender, EventArgs e)
		{
			if (image is null)
			{
				MessageBox.Show("Please load an image first.");
				return;
			}
			if (samPoints.Count < 1)
			{
				MessageBox.Show("Please select several points on image.");
				return;
			}
			(List<SKBitmap> bitmaps, List<float[]> iou_predictions) = predictor.Predict(image, samPoints);
			PictureBox_Mask.Image = SKBitmapToBitmap(bitmaps[0]);
			GC.Collect();
		}

		private void MainForm_Load(object sender, EventArgs e)
		{
			ComboBox_Device.SelectedIndex = 0;
		}

		private Bitmap SKBitmapToBitmap(SKBitmap skBitmap)
		{
			using (var stream = new MemoryStream())
			{
				skBitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
				stream.Seek(0, SeekOrigin.Begin);
				return new Bitmap(stream);
			}
		}

		private void Button_ModelScan_Click(object sender, EventArgs e)
		{
			OpenFileDialog openFileDialog = new OpenFileDialog
			{
				Filter = ".pth|*.pth",
			};
			if (openFileDialog.ShowDialog() == DialogResult.OK)
			{
				TextBox_ModelPath.Text = openFileDialog.FileName;
				modelPath = openFileDialog.FileName;
			}
		}

		private void Button_ModelLoad_Click(object sender, EventArgs e)
		{
			if (string.IsNullOrEmpty(modelPath))
			{
				MessageBox.Show("Please Load a model first");
			}
			SamDevice device = (SamDevice)ComboBox_Device.SelectedIndex; // or SamDevice.Cpu if you want to run on CPU
			try
			{
				predictor = new SamSharp.Utils.SamPredictor(modelPath, device);
				MessageBox.Show("Model Loaded Done.");
			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.Message);
			}
		}

		private void Button_RemoveLastPoint_Click(object sender, EventArgs e)
		{
			int count = samPoints.Count;
			if (count > 0)
			{
				samPoints.RemoveAt(count - 1);
				DrawPoints();
			}

		}
	}
}
