# 🎬 Nova Canvas & Reel Optimizer

A comprehensive Streamlit application for optimizing prompts and generating images/videos using Amazon's Nova Canvas and Reel models, featuring AI-powered suggestions and advanced outpainting capabilities.

![Nova Canvas & Reel Optimizer](assets/video_demo.gif)

## ✨ Features

### 🖼️ **Image Generation with Nova Canvas**
- **Text-to-Image Generation**: Create images from text prompts with optimization
- **Image-to-Image Outpainting**: Upload images and extend/edit them with AI
- **AI-Powered Suggestions**: Claude Sonnet 4 analyzes images and suggests:
  - Smart mask prompts for targeting specific areas
  - Creative themes for outpainting and enhancement
- **Intelligent Resizing**: Resize images to different dimensions with AI fill
- **Comparison Mode**: Generate and compare original vs optimized prompts
- **Advanced Controls**: Seed, CFG scale, aspect ratio, and quality settings

### 🎥 **Video Generation with Nova Reel**
- **Text-to-Video**: Generate videos from text descriptions
- **Image-to-Video**: Convert images to videos with motion
- **Multiple Model Support**: Nova Reel and Luma Ray models
- **Prompt Optimization**: AI-enhanced prompts for better video results
- **Comparison Videos**: Side-by-side original vs optimized results

### 🎬 **Long Video Generation**
- **Story-to-Video Pipeline**: Transform stories into multi-shot videos
- **Automatic Storyboarding**: AI breaks down stories into scenes
- **Shot Generation**: Create individual video clips for each scene
- **Video Stitching**: Combine clips into cohesive long-form videos
- **Caption Generation**: Add captions to final videos
- **Progress Tracking**: Real-time progress for complex workflows

### 🤖 **AI-Powered Intelligence**
- **Claude Sonnet 4 Integration**: Advanced image analysis and suggestions
- **Context-Aware Recommendations**: Understands image content and composition
- **Smart Prompt Generation**: Optimized prompts based on Amazon Nova guidelines
- **Automatic Error Recovery**: Intelligent handling of generation failures

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- AWS credentials configured with access to:
  - Amazon Bedrock (Nova Canvas, Nova Reel, Claude Sonnet 4)
  - S3 bucket for video output
- Sufficient AWS permissions for model access

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/buckylee2019/canvas_reel_optimizer.git
cd canvas_reel_optimizer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure AWS credentials**:
```bash
aws configure
# or set environment variables:
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

4. **Update configuration**:
   - Edit `config.py` to set your S3 bucket name
   - Verify model IDs and regions match your AWS setup

### Running the Application

**Streamlit Interface** (Recommended):
```bash
streamlit run streamlit_app.py
```

**Gradio Interface** (Legacy):
```bash
python app.py
```

Open your browser to the displayed URL (typically http://localhost:8501 for Streamlit)

## 📖 Usage Guide

### 🖼️ Image Generation

#### Text-to-Image Mode
1. Select "Text-to-Image" mode
2. Enter your prompt or choose from templates
3. Click "🔧 Optimize Prompt" for AI enhancement
4. Click "🎨 Generate Image" to create
5. Use "Copy to Video Gen" to transfer to video generation

#### Image-to-Image Outpainting Mode
1. Select "Image-to-Image (Outpainting)" mode
2. Upload your image
3. **Get AI Suggestions**:
   - Click "🎯 Suggest Mask Prompts" for targeting specific areas
   - Click "🎨 Suggest Themes" for creative enhancement ideas
   - Click "🚀 Get All Suggestions" for comprehensive analysis
4. Use suggested prompts or write your own
5. **Optional Resizing**:
   - Enable "Enable Resize" checkbox
   - Set target dimensions or use quick presets
   - Preview canvas layout
6. Generate your enhanced image

### 🎥 Video Generation
1. Enter video prompt or copy from image generation
2. Optionally upload an image for image-to-video
3. Choose video model (Nova Reel or Luma Ray)
4. Optimize prompt and generate video
5. Compare results if comparison mode is enabled

### 🎬 Long Video Generation
1. Enter your story or concept
2. Click "Generate Shots" to create storyboard
3. Review and edit generated shots
4. Click "Generate Video" for full pipeline:
   - Shot image generation
   - Video creation for each shot
   - Video stitching and captioning

## 🛠️ Configuration

### Model Settings
- **Optimization Model**: Choose between Nova Pro/Lite for prompt optimization
- **Canvas Model**: Nova Canvas v1:0 for image generation
- **Reel Model**: Nova Reel v1:0/v1:1 for video generation
- **Analysis Model**: Claude Sonnet 4 for AI suggestions

### S3 Configuration
Update `config.py` with your S3 bucket:
```python
DEFAULT_BUCKET = "your-s3-bucket-name"
```

### Advanced Settings
- **CFG Scale**: Controls creativity vs adherence (1.0-10.0)
- **Quality**: Standard vs Premium generation
- **Aspect Ratios**: Multiple preset ratios available
- **Seeds**: Reproducible generation with fixed seeds

## 📁 Project Structure

```
canvas_reel_optimizer/
├── streamlit_app.py          # Main Streamlit application
├── app.py                    # Legacy Gradio interface
├── config.py                 # Configuration settings
├── generation.py             # Core generation functions
├── shot_video.py             # Long video generation logic
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── assets/                   # Demo images and videos
├── generated_images/         # Output directory for images
├── generated_videos/         # Output directory for videos
└── .gitignore               # Git ignore rules
```

## 🎯 Key Features Deep Dive

### AI-Powered Suggestions
The application uses Claude Sonnet 4 to analyze uploaded images and provide:

**Mask Prompts**: Target specific areas
- "background" - for background modifications
- "person's clothing" - for clothing changes
- "sky and clouds" - for atmospheric edits
- "foreground objects" - for object manipulation

**Creative Themes**: Enhancement ideas
- "extend with serene mountain landscape"
- "add warm golden hour lighting"
- "transform to professional studio setting"
- "enhance with artistic bokeh effects"

### Intelligent Outpainting
- **Canvas Creation**: Automatically positions original image
- **Smart Masking**: Preserves original content, fills new areas
- **Dimension Validation**: Ensures Nova Canvas compatibility
- **Preview System**: Shows what areas will be AI-generated

### Advanced Video Pipeline
- **Multi-Model Support**: Nova Reel, Luma Ray integration
- **Prompt Optimization**: AI-enhanced descriptions
- **Error Recovery**: Handles generation failures gracefully
- **Progress Tracking**: Real-time status updates

## 🔧 Troubleshooting

### Common Issues

**"NovaCanvasResizer object has no attribute 'resize_image'"**
- Click "Clear Resizer Cache" in Debug Options
- Restart the Streamlit application

**AWS Permissions Errors**
- Verify Bedrock model access in your AWS account
- Check S3 bucket permissions
- Ensure correct AWS region configuration

**Generation Failures**
- Try lower CFG scale values (3.0-5.0)
- Use "standard" quality for testing
- Check image dimensions (320-2048 pixels)

### Performance Tips
- Use "standard" quality for faster testing
- Enable comparison mode only when needed
- Clear generated files periodically to save disk space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Amazon Nova**: Canvas and Reel models for image/video generation
- **Claude Sonnet 4**: Advanced AI analysis and suggestions
- **Streamlit**: Excellent web application framework
- **AWS Bedrock**: Managed AI service platform

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review AWS Bedrock documentation for model-specific guidance

---

**Built with ❤️ using Amazon Nova, Claude Sonnet 4, and Streamlit**