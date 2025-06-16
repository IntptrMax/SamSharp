namespace SamSharp_WinFormDemo
{
	partial class MainForm
	{
		/// <summary>
		///  Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		///  Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		///  Required method for Designer support - do not modify
		///  the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			groupBox1 = new GroupBox();
			PictureBox_Image = new PictureBox();
			Button_Run = new Button();
			RadioButton_Background = new RadioButton();
			RadioButton_Foreground = new RadioButton();
			Button_ImageLoad = new Button();
			groupBox3 = new GroupBox();
			PictureBox_Mask = new PictureBox();
			groupBox4 = new GroupBox();
			groupBox2 = new GroupBox();
			ComboBox_Device = new ComboBox();
			Button_ModelLoad = new Button();
			groupBox5 = new GroupBox();
			Button_ModelScan = new Button();
			TextBox_ModelPath = new TextBox();
			Button_RemoveLastPoint = new Button();
			groupBox6 = new GroupBox();
			label1 = new Label();
			NumericUpDown_ImageSize = new NumericUpDown();
			Button_RemoveLastBox = new Button();
			RadioButton_Box = new RadioButton();
			groupBox1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)PictureBox_Image).BeginInit();
			groupBox3.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)PictureBox_Mask).BeginInit();
			groupBox4.SuspendLayout();
			groupBox2.SuspendLayout();
			groupBox5.SuspendLayout();
			groupBox6.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_ImageSize).BeginInit();
			SuspendLayout();
			// 
			// groupBox1
			// 
			groupBox1.Controls.Add(PictureBox_Image);
			groupBox1.Location = new Point(12, 12);
			groupBox1.Name = "groupBox1";
			groupBox1.Size = new Size(416, 425);
			groupBox1.TabIndex = 0;
			groupBox1.TabStop = false;
			groupBox1.Text = "Input Image";
			// 
			// PictureBox_Image
			// 
			PictureBox_Image.Dock = DockStyle.Fill;
			PictureBox_Image.Location = new Point(3, 19);
			PictureBox_Image.Name = "PictureBox_Image";
			PictureBox_Image.Size = new Size(410, 403);
			PictureBox_Image.SizeMode = PictureBoxSizeMode.Zoom;
			PictureBox_Image.TabIndex = 0;
			PictureBox_Image.TabStop = false;
			PictureBox_Image.Paint += PictureBox_Image_Paint;
			PictureBox_Image.MouseDown += PictureBox_Image_MouseDown;
			PictureBox_Image.MouseMove += PictureBox_Image_MouseMove;
			PictureBox_Image.MouseUp += PictureBox_Image_MouseUp;
			// 
			// Button_Run
			// 
			Button_Run.Location = new Point(746, 18);
			Button_Run.Name = "Button_Run";
			Button_Run.Size = new Size(92, 25);
			Button_Run.TabIndex = 3;
			Button_Run.Text = "Run";
			Button_Run.UseVisualStyleBackColor = true;
			Button_Run.Click += Button_Run_Click;
			// 
			// RadioButton_Background
			// 
			RadioButton_Background.AutoSize = true;
			RadioButton_Background.Location = new Point(118, 22);
			RadioButton_Background.Name = "RadioButton_Background";
			RadioButton_Background.Size = new Size(102, 21);
			RadioButton_Background.TabIndex = 2;
			RadioButton_Background.TabStop = true;
			RadioButton_Background.Text = "Back Ground";
			RadioButton_Background.UseVisualStyleBackColor = true;
			// 
			// RadioButton_Foreground
			// 
			RadioButton_Foreground.AutoSize = true;
			RadioButton_Foreground.Checked = true;
			RadioButton_Foreground.Location = new Point(12, 22);
			RadioButton_Foreground.Name = "RadioButton_Foreground";
			RadioButton_Foreground.Size = new Size(100, 21);
			RadioButton_Foreground.TabIndex = 1;
			RadioButton_Foreground.TabStop = true;
			RadioButton_Foreground.Text = "Fore Ground";
			RadioButton_Foreground.UseVisualStyleBackColor = true;
			// 
			// Button_ImageLoad
			// 
			Button_ImageLoad.Location = new Point(736, 41);
			Button_ImageLoad.Name = "Button_ImageLoad";
			Button_ImageLoad.Size = new Size(102, 24);
			Button_ImageLoad.TabIndex = 0;
			Button_ImageLoad.Text = "Load Image";
			Button_ImageLoad.UseVisualStyleBackColor = true;
			Button_ImageLoad.Click += Button_ImageLoad_Click;
			// 
			// groupBox3
			// 
			groupBox3.Controls.Add(PictureBox_Mask);
			groupBox3.Location = new Point(440, 12);
			groupBox3.Name = "groupBox3";
			groupBox3.Size = new Size(416, 425);
			groupBox3.TabIndex = 2;
			groupBox3.TabStop = false;
			groupBox3.Text = "Mask";
			// 
			// PictureBox_Mask
			// 
			PictureBox_Mask.Dock = DockStyle.Fill;
			PictureBox_Mask.Location = new Point(3, 19);
			PictureBox_Mask.Name = "PictureBox_Mask";
			PictureBox_Mask.Size = new Size(410, 403);
			PictureBox_Mask.SizeMode = PictureBoxSizeMode.Zoom;
			PictureBox_Mask.TabIndex = 0;
			PictureBox_Mask.TabStop = false;
			// 
			// groupBox4
			// 
			groupBox4.Controls.Add(groupBox2);
			groupBox4.Controls.Add(Button_ModelLoad);
			groupBox4.Controls.Add(Button_ImageLoad);
			groupBox4.Controls.Add(groupBox5);
			groupBox4.Location = new Point(12, 443);
			groupBox4.Name = "groupBox4";
			groupBox4.Size = new Size(844, 90);
			groupBox4.TabIndex = 3;
			groupBox4.TabStop = false;
			// 
			// groupBox2
			// 
			groupBox2.Controls.Add(ComboBox_Device);
			groupBox2.Location = new Point(462, 22);
			groupBox2.Name = "groupBox2";
			groupBox2.Size = new Size(122, 57);
			groupBox2.TabIndex = 4;
			groupBox2.TabStop = false;
			groupBox2.Text = "Device";
			// 
			// ComboBox_Device
			// 
			ComboBox_Device.DropDownStyle = ComboBoxStyle.DropDownList;
			ComboBox_Device.FormattingEnabled = true;
			ComboBox_Device.Items.AddRange(new object[] { "Cuda", "CPU" });
			ComboBox_Device.Location = new Point(6, 19);
			ComboBox_Device.Name = "ComboBox_Device";
			ComboBox_Device.Size = new Size(106, 25);
			ComboBox_Device.TabIndex = 1;
			// 
			// Button_ModelLoad
			// 
			Button_ModelLoad.Location = new Point(605, 41);
			Button_ModelLoad.Name = "Button_ModelLoad";
			Button_ModelLoad.Size = new Size(102, 24);
			Button_ModelLoad.TabIndex = 2;
			Button_ModelLoad.Text = "Model Load";
			Button_ModelLoad.UseVisualStyleBackColor = true;
			Button_ModelLoad.Click += Button_ModelLoad_Click;
			// 
			// groupBox5
			// 
			groupBox5.Controls.Add(Button_ModelScan);
			groupBox5.Controls.Add(TextBox_ModelPath);
			groupBox5.Location = new Point(6, 20);
			groupBox5.Name = "groupBox5";
			groupBox5.Size = new Size(450, 57);
			groupBox5.TabIndex = 0;
			groupBox5.TabStop = false;
			groupBox5.Text = "Model Path";
			// 
			// Button_ModelScan
			// 
			Button_ModelScan.Location = new Point(369, 24);
			Button_ModelScan.Name = "Button_ModelScan";
			Button_ModelScan.Size = new Size(75, 23);
			Button_ModelScan.TabIndex = 1;
			Button_ModelScan.Text = "Scan";
			Button_ModelScan.UseVisualStyleBackColor = true;
			Button_ModelScan.Click += Button_ModelScan_Click;
			// 
			// TextBox_ModelPath
			// 
			TextBox_ModelPath.Location = new Point(6, 24);
			TextBox_ModelPath.Name = "TextBox_ModelPath";
			TextBox_ModelPath.ReadOnly = true;
			TextBox_ModelPath.Size = new Size(357, 23);
			TextBox_ModelPath.TabIndex = 0;
			// 
			// Button_RemoveLastPoint
			// 
			Button_RemoveLastPoint.Location = new Point(280, 18);
			Button_RemoveLastPoint.Name = "Button_RemoveLastPoint";
			Button_RemoveLastPoint.Size = new Size(126, 25);
			Button_RemoveLastPoint.TabIndex = 5;
			Button_RemoveLastPoint.Text = "Remove Last Point";
			Button_RemoveLastPoint.UseVisualStyleBackColor = true;
			Button_RemoveLastPoint.Click += Button_RemoveLastPoint_Click;
			// 
			// groupBox6
			// 
			groupBox6.Controls.Add(label1);
			groupBox6.Controls.Add(NumericUpDown_ImageSize);
			groupBox6.Controls.Add(Button_RemoveLastBox);
			groupBox6.Controls.Add(RadioButton_Box);
			groupBox6.Controls.Add(Button_RemoveLastPoint);
			groupBox6.Controls.Add(Button_Run);
			groupBox6.Controls.Add(RadioButton_Foreground);
			groupBox6.Controls.Add(RadioButton_Background);
			groupBox6.Location = new Point(12, 539);
			groupBox6.Name = "groupBox6";
			groupBox6.Size = new Size(844, 53);
			groupBox6.TabIndex = 4;
			groupBox6.TabStop = false;
			groupBox6.Text = "Points and boxes";
			// 
			// label1
			// 
			label1.AutoSize = true;
			label1.Location = new Point(560, 22);
			label1.Name = "label1";
			label1.Size = new Size(72, 17);
			label1.TabIndex = 8;
			label1.Text = "Image Size";
			// 
			// NumericUpDown_ImageSize
			// 
			NumericUpDown_ImageSize.Increment = new decimal(new int[] { 64, 0, 0, 0 });
			NumericUpDown_ImageSize.Location = new Point(638, 18);
			NumericUpDown_ImageSize.Maximum = new decimal(new int[] { 8192, 0, 0, 0 });
			NumericUpDown_ImageSize.Minimum = new decimal(new int[] { 128, 0, 0, 0 });
			NumericUpDown_ImageSize.Name = "NumericUpDown_ImageSize";
			NumericUpDown_ImageSize.Size = new Size(69, 23);
			NumericUpDown_ImageSize.TabIndex = 2;
			NumericUpDown_ImageSize.Value = new decimal(new int[] { 512, 0, 0, 0 });
			// 
			// Button_RemoveLastBox
			// 
			Button_RemoveLastBox.Location = new Point(412, 18);
			Button_RemoveLastBox.Name = "Button_RemoveLastBox";
			Button_RemoveLastBox.Size = new Size(126, 25);
			Button_RemoveLastBox.TabIndex = 7;
			Button_RemoveLastBox.Text = "Remove Last Box";
			Button_RemoveLastBox.UseVisualStyleBackColor = true;
			Button_RemoveLastBox.Click += Button_RemoveLastBox_Click;
			// 
			// RadioButton_Box
			// 
			RadioButton_Box.AutoSize = true;
			RadioButton_Box.Location = new Point(226, 22);
			RadioButton_Box.Name = "RadioButton_Box";
			RadioButton_Box.Size = new Size(48, 21);
			RadioButton_Box.TabIndex = 6;
			RadioButton_Box.TabStop = true;
			RadioButton_Box.Text = "Box";
			RadioButton_Box.UseVisualStyleBackColor = true;
			// 
			// MainForm
			// 
			AutoScaleDimensions = new SizeF(7F, 17F);
			AutoScaleMode = AutoScaleMode.Font;
			ClientSize = new Size(866, 603);
			Controls.Add(groupBox6);
			Controls.Add(groupBox4);
			Controls.Add(groupBox3);
			Controls.Add(groupBox1);
			Name = "MainForm";
			Text = "MainForm";
			Load += MainForm_Load;
			groupBox1.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)PictureBox_Image).EndInit();
			groupBox3.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)PictureBox_Mask).EndInit();
			groupBox4.ResumeLayout(false);
			groupBox2.ResumeLayout(false);
			groupBox5.ResumeLayout(false);
			groupBox5.PerformLayout();
			groupBox6.ResumeLayout(false);
			groupBox6.PerformLayout();
			((System.ComponentModel.ISupportInitialize)NumericUpDown_ImageSize).EndInit();
			ResumeLayout(false);
		}

		#endregion

		private GroupBox groupBox1;
		private PictureBox PictureBox_Image;
		private Button Button_ImageLoad;
		private RadioButton RadioButton_Foreground;
		private RadioButton RadioButton_Background;
		private Button Button_Run;
		private GroupBox groupBox3;
		private PictureBox PictureBox_Mask;
		private GroupBox groupBox4;
		private GroupBox groupBox5;
		private Button Button_ModelScan;
		private TextBox TextBox_ModelPath;
		private Button Button_ModelLoad;
		private ComboBox ComboBox_Device;
		private GroupBox groupBox2;
		private Button Button_RemoveLastPoint;
		private GroupBox groupBox6;
		private RadioButton RadioButton_Box;
		private Button Button_RemoveLastBox;
		private Label label1;
		private NumericUpDown NumericUpDown_ImageSize;
	}
}
