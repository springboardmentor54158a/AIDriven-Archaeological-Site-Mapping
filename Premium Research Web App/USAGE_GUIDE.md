# 🎯 Archaeological Site Analysis Platform - Usage Guide

## Quick Start

### Step 1: Landing Page
- View the animated hero section with project overview
- Click **"Explore Platform"** to go directly to dashboard
- Or click **"Login"** / **"Sign Up"** to access authentication

### Step 2: Authentication (Optional)
- **Sign Up**: Enter name, email, and password (minimum 6 characters)
- **Login**: Enter email and password
- Form validation provides instant feedback
- Note: Authentication is simulated - any valid credentials work

### Step 3: Main Dashboard
- **Overview**: See project status widgets
- **Upload Status**: Check if data is uploaded
- **Pipeline Progress**: Monitor analysis completion (0-100%)
- **System Status**: Confirm all systems operational

**Quick Actions:**
- Upload Data (if not done)
- Start Analysis (if data uploaded)
- View Metrics (if pipeline started)

### Step 4: Upload Data
Navigate to **Upload Data** from sidebar

**Images Section:**
- Drag & drop satellite/drone images OR click to browse
- Supports PNG, JPG, TIFF formats
- Upload multiple images at once
- Preview uploaded files with size info
- Remove individual files with X button

**CSV Section:**
- Drag & drop CSV terrain data OR click to browse
- Only one CSV file needed
- Preview file name and size
- Remove and re-upload if needed

Click **"Proceed to Analysis"** when both image(s) and CSV are uploaded.

### Step 5: AI Pipeline
Navigate to **AI Pipeline** from sidebar

**Pipeline Steps:**
1. **Image Preprocessing** - Normalizing and resizing
2. **Semantic Segmentation** - Identifying features
3. **Object Detection** - Detecting structures
4. **Erosion Prediction** - Analyzing degradation

**Controls:**
- **Start**: Begin analysis pipeline
- **Pause/Resume**: Temporarily halt processing
- **Reset**: Return to beginning

Watch the animated progress bars for each step!

### Step 6: Model Performance
Navigate to **Performance** from sidebar

**View Metrics:**
- **Segmentation Quality**: IoU and Dice scores
- **Detection Accuracy**: mAP, Precision, Recall
- **Prediction Quality**: RMSE and R² scores

**Interactive Features:**
- Model selector dropdown
- Confidence threshold slider
- Click **"Run Analysis"** to generate new metrics
- View erosion trend chart
- Compare metrics in bar chart

**Understanding Metrics:**
- **IoU** (Intersection over Union): 0-100%, higher is better
- **Dice Score**: 0-100%, measures segmentation overlap
- **mAP** (mean Average Precision): 0-100%, detection quality
- **RMSE** (Root Mean Square Error): Lower is better for predictions
- **R² Score**: 0-100%, how well model fits data

### Step 7: Results & Insights
Navigate to **Results** from sidebar

**Risk Assessment:**
- **Low Risk**: RMSE < 3 (Green)
- **Medium Risk**: RMSE 3-4.5 (Orange)
- **High Risk**: RMSE > 4.5 (Red)

**AI Insights:**
- Detailed segmentation analysis
- Object detection summary
- Erosion prediction insights
- Confidence percentage

**Recommendations:**
- Immediate actions based on risk level
- Data collection suggestions
- Model optimization tips

**Export Report:**
- Click **"Export Report"** button
- Downloads JSON file with all metrics

### Step 8: Review & Feedback
Navigate to **Feedback** from sidebar

**Rate Your Experience:**
- Click stars (1-5) to rate
- See instant feedback text

**Choose Category:**
- General
- Accuracy
- Usability
- Feature Request

**Write Comments:**
- Share thoughts and suggestions
- Report any issues
- Request new features

Click **"Submit Feedback"** to complete!

### Step 9: Success Screen
Automatically shown after feedback or click manually

**View Summary:**
- Overall performance percentage
- Segmentation quality
- Detection accuracy
- Prediction quality

**Next Steps:**
- Review results in dashboard
- Export reports
- Start new analysis

**Actions:**
- **Back to Dashboard**: Return to main view
- **Start New Analysis**: Restart entire workflow

## 🎨 Theme Toggle

Click the **Sun/Moon icon** (top-right corner) to switch between:
- **Dark Mode**: Better for low-light environments
- **Light Mode**: Easier reading in bright conditions

Theme persists throughout the session!

## 🔔 Notifications

Watch for floating notifications (top-right):
- **Green**: Success messages
- **Red**: Error messages
- **Orange**: Warning messages
- **Blue**: Info messages

Auto-dismiss after 5 seconds.

## 💡 Tips & Tricks

### Best Practices:
1. Upload high-quality images for better analysis
2. Ensure CSV data is properly formatted
3. Let pipeline complete fully before viewing results
4. Review all metrics before making decisions
5. Export reports for documentation

### Keyboard Shortcuts:
- No keyboard shortcuts implemented (use mouse/touch)

### Performance:
- Upload recommended: 5-10 images max
- CSV file size: Under 10MB recommended
- Charts may lag with very large datasets

### Troubleshooting:
- **Data not uploading?** Check file format
- **Pipeline stuck?** Try reset button
- **No metrics showing?** Run analysis first
- **Charts not loading?** Refresh page

## 🌟 Features Overview

### ✅ Completed Features:
- Full authentication flow
- Drag-and-drop file upload
- Animated AI pipeline
- Real-time metrics generation
- Interactive charts
- Risk assessment
- Feedback system
- Export functionality
- Dark/light theme
- Responsive design

### 🎯 Simulated Features:
- AI model processing (frontend simulation)
- Metrics calculation (random realistic values)
- File analysis (client-side only)
- Authentication (no backend)

## 📱 Mobile Experience

The platform is fully responsive:
- **Desktop**: Full sidebar navigation
- **Tablet**: Collapsible sidebar
- **Mobile**: Horizontal scrolling navigation

All features work on mobile devices!

## 🔒 Privacy & Security

**Note:** This is a demonstration platform:
- No data is sent to servers
- Files stay in browser memory
- No actual AI processing occurs
- Metrics are simulated
- Authentication is frontend-only

**Do not upload sensitive data!**

## 🆘 Need Help?

1. Check this usage guide
2. Review PROJECT_INFO.md for technical details
3. Explore each section methodically
4. Use feedback form to report issues

---

**Enjoy exploring the Archaeological Site Analysis Platform!** 🏛️
