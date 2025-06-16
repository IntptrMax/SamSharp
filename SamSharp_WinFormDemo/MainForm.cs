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
		private List<SamBox> samBoxes = new List<SamBox>();

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
					samBoxes = new List<SamBox>();
					DrawPointsAndBoxes();
				}
				catch (Exception ex)
				{
					MessageBox.Show($"Error loading image: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
			}
		}

		private void DrawPointsAndBoxes()
		{
			if (pictureBoxImage is null)
			{
				MessageBox.Show("Please load an image first.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				return;
			}
			Image img = pictureBoxImage.Clone() as Image;
			Graphics graphics = Graphics.FromImage(img);
			samPoints.ForEach(point =>
			{
				graphics.FillEllipse(point.Label ? Brushes.Green : Brushes.Blue, point.X - 10, point.Y - 10, 20, 20);
			});

			samBoxes.ForEach(box =>
			{
				graphics.DrawRectangle(new Pen(Color.Red, 4),
					box.Left,
					box.Top,
					box.Right - box.Left,
					box.Bottom - box.Top);
			});
			//graphics.Save();
			graphics.Dispose();
			PictureBox_Image.Image = img;
		}

		private void Button_Run_Click(object sender, EventArgs e)
		{
			if (image is null)
			{
				MessageBox.Show("Please load an image first.");
				return;
			}
			if (samPoints.Count + samBoxes.Count == 0)
			{
				MessageBox.Show("Please select several points or boxes on image.");
				return;
			}
			int imageSize = (int)NumericUpDown_ImageSize.Value;

			List<PredictOutput> outputs = predictor.Predict(image, samPoints, samBoxes,imageSize);

			SKBitmap resultBmp = image.Copy();
			using (SKCanvas canvas = new SKCanvas(resultBmp))
			{
				for (int i = 0; i < outputs.Count; i++)
				{
					SKImageInfo info = new SKImageInfo(image.Width, image.Height);
					using (SKBitmap maskBitmap = new SKBitmap(info))
					using (SKCanvas c = new SKCanvas(maskBitmap))
					{
						c.Clear(SKColors.Transparent);
						PredictOutput output = outputs[i];

						// Precision of the masks
						Console.WriteLine($"Mask {i}: Precision: {output.Precision * 100:F2}%");
						bool[,] mask = output.Mask;
						Random random = new Random();
						SKColor color = new SKColor((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256), 128);

						for (int y = 0; y < mask.GetLength(1); y++)
						{
							for (int x = 0; x < mask.GetLength(0); x++)
							{
								if (mask[x, y])
								{
									c.DrawPoint(x, y, color);
								}
							}
						}
						canvas.DrawBitmap(maskBitmap, new SKPoint(0, 0));
					}
				}
			}
			PictureBox_Mask.Image = SKBitmapToBitmap(resultBmp);
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

		private bool isDrawing = false;
		private Point startPoint;
		private Rectangle currentRect;

		private void PictureBox_Image_MouseDown(object sender, MouseEventArgs e)
		{
			if (e.Button != MouseButtons.Left)
			{
				return; // Only handle left mouse button clicks
			}
			if (e.Location.X < padX || e.Location.X > PictureBox_Image.Width - padX || e.Location.Y < padY || e.Location.Y > PictureBox_Image.Height - padY)
			{
				MessageBox.Show("Click is outside the image bounds.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				return;
			}

			int adjustedX = (int)((e.Location.X - padX) / scale);
			int adjustedY = (int)((e.Location.Y - padY) / scale);
			if (RadioButton_Foreground.Checked)
			{
				SamPoint point = new SamPoint(adjustedX, adjustedY, true); // Assuming true for foreground
				samPoints.Add(point);
				DrawPointsAndBoxes();
			}
			else if (RadioButton_Background.Checked)
			{
				SamPoint point = new SamPoint(adjustedX, adjustedY, false); // Assuming true for foreground
				samPoints.Add(point);
				DrawPointsAndBoxes();
			}
			else if (RadioButton_Box.Checked)
			{
				isDrawing = true;
				startPoint = e.Location;
				currentRect = new Rectangle(startPoint, new Size(0, 0));
			}
		}

		private void PictureBox_Image_MouseUp(object sender, MouseEventArgs e)
		{
			PictureBox_Image.Invalidate();
			if (e.Button == MouseButtons.Left && isDrawing)
			{
				isDrawing = false;

				int x = Math.Min(startPoint.X, e.X);
				int y = Math.Min(startPoint.Y, e.Y);
				int width = Math.Abs(startPoint.X - e.X);
				int height = Math.Abs(startPoint.Y - e.Y);

				if (e.Location.X < padX || e.Location.X > PictureBox_Image.Width - padX || e.Location.Y < padY || e.Location.Y > PictureBox_Image.Height - padY)
				{
					MessageBox.Show("Click is outside the image bounds.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
					return;
				}

				int adjustedLeft = (int)((x - padX) / scale);
				int adjustedTop = (int)((y - padY) / scale);
				int adjustedRight = (int)((x + width - padX) / scale);
				int adjustedBottom = (int)((y + height - padY) / scale);

				samBoxes.Add(new SamBox(adjustedLeft, adjustedTop, adjustedRight, adjustedBottom));
				DrawPointsAndBoxes();
			}
		}

		private void PictureBox_Image_MouseMove(object sender, MouseEventArgs e)
		{
			if (isDrawing)
			{
				int x = Math.Min(startPoint.X, e.X);
				int y = Math.Min(startPoint.Y, e.Y);
				int width = Math.Abs(startPoint.X - e.X);
				int height = Math.Abs(startPoint.Y - e.Y);

				currentRect = new Rectangle(x, y, width, height);
				PictureBox_Image.Invalidate();
			}
		}

		private void PictureBox_Image_Paint(object sender, PaintEventArgs e)
		{
			if (isDrawing)
			{
				if (currentRect != null && currentRect.Width > 0 && currentRect.Height > 0)
				{
					using (Pen pen = new Pen(Color.Red, 2))
					{
						e.Graphics.DrawRectangle(pen, currentRect);
					}
				}
			}
		}

		private void Button_RemoveLastPoint_Click(object sender, EventArgs e)
		{
			int count = samPoints.Count;
			if (count > 0)
			{
				samPoints.RemoveAt(count - 1);
				DrawPointsAndBoxes();
			}
		}

		private void Button_RemoveLastBox_Click(object sender, EventArgs e)
		{
			int count = samBoxes.Count;
			if (count > 0)
			{
				samBoxes.RemoveAt(count - 1);
				DrawPointsAndBoxes();
			}
		}
	}
}
